"""
Stage 1 (Pre-Filter & Post-Generation Check): Context-Window Feasibility Filter.

Evaluates whether a sequential candidate group satisfies the prototype threshold:
    P + R + O <= 0.70 * C
where:
- C = selected model context-window capacity
- P = combined static prompt tokens (or generated unified prompt tokens post-Stage 4)
- R = estimated runtime dynamic input tokens (without double-counting static prompts)
- O = reserved output tokens summed across group agents
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from optimization_agent.model_selection.groq_catalog import MODEL_CATALOG

MAX_CONTEXT_UTILIZATION_RATIO = 0.70

MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    model_id: int(meta["context_window"])
    for model_id, meta in MODEL_CATALOG.items()
    if isinstance(meta.get("context_window"), (int, float)) and meta["context_window"] > 0
}
MODEL_CONTEXT_WINDOWS.update({
    "llama-3.3-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "mixtral-8x7b-32768": 32768,
    "gemma2-9b-it": 8192,
})


def resolve_model_context_window(
    model_id: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[int], str, Optional[str]]:
    """
    Resolve model context window C from state override or baseline model metadata.
    Never guesses capacity for unknown models.
    """
    resolved_model = (model_id or "").strip() if isinstance(model_id, str) else None
    if not resolved_model and state:
        meta = state.get("workflow_metadata") or {}
        recon = meta.get("model_reconciliation") or {}
        resolved_model = (
            recon.get("runtime_observed_model")
            or recon.get("declared_model")
            or state.get("target_model")
        )
        if isinstance(resolved_model, str):
            resolved_model = resolved_model.strip() or None

    if state and state.get("model_context_window") is not None:
        raw_cap = state.get("model_context_window")
        if isinstance(raw_cap, (int, float)) and raw_cap > 0:
            return int(raw_cap), "state_model_context_window", resolved_model
        return None, "invalid_state_model_context_window", resolved_model

    if resolved_model and resolved_model in MODEL_CONTEXT_WINDOWS:
        return int(MODEL_CONTEXT_WINDOWS[resolved_model]), "catalog_lookup", resolved_model

    return None, "context_window_unavailable", resolved_model


def count_tokens(text: Optional[str], model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate tokens using tiktoken when installed, or a deterministic
    conservative BPE heuristic otherwise.
    """
    if not text:
        return {"tokens": 0, "method": "empty", "is_exact_tokenizer": True}

    try:
        import tiktoken  # type: ignore

        model_lower = (model_id or "").lower()
        enc_name = "o200k_base" if ("gpt-oss" in model_lower or "gpt-4o" in model_lower) else "cl100k_base"
        enc = tiktoken.get_encoding(enc_name)
        return {
            "tokens": len(enc.encode(text)),
            "method": f"tiktoken:{enc_name}",
            "is_exact_tokenizer": True,
        }
    except Exception:
        # Conservative upper-bound BPE estimate (~3.5 chars/token or ~1.15x regex pieces)
        char_est = math.ceil(len(text) / 3.5)
        piece_est = math.ceil(len(re.findall(r"\w+|[^\w\s]", text)) * 1.15)
        return {
            "tokens": max(char_est, piece_est),
            "method": "conservative_bpe_heuristic",
            "is_exact_tokenizer": False,
        }


def extract_agent_static_prompt(
    agent_id: str,
    prompt_metrics_map: Dict[str, Dict[str, Any]],
    profile_map: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    """Retrieve full static prompt text for an agent, falling back to profile task specs."""
    pm = prompt_metrics_map.get(agent_id, {})
    full_text = pm.get("prompt_text")
    if isinstance(full_text, str) and full_text.strip():
        return full_text, "mcp_prompt_metrics_full"

    sample_text = pm.get("sample_prompt_text")
    if isinstance(sample_text, str) and sample_text.strip() and not pm.get("is_truncated", False):
        return sample_text, "mcp_prompt_metrics_sample"

    prof = profile_map.get(agent_id, {})
    parts: List[str] = []
    for key in ("role", "goal", "backstory", "declared_responsibilities"):
        val = prof.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    for task in prof.get("tasks", []) or []:
        if isinstance(task, dict):
            desc = task.get("description")
            if isinstance(desc, str) and desc.strip():
                parts.append(desc.strip())
    if parts:
        return "\n".join(parts), "mcp_profile_fallback"

    return "", "unavailable"


def evaluate_context_budget(
    agent_ids: List[str],
    model_id: Optional[str],
    prompt_metrics_map: Dict[str, Dict[str, Any]],
    token_metrics_map: Dict[str, Dict[str, Any]],
    profile_map: Dict[str, Dict[str, Any]],
    dep_parents_map: Optional[Dict[str, Set[str]]] = None,
    unified_prompt_text: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate P + R + O <= 0.70 * C for a sequential candidate group.
    Never treats mean_tokens as prompt tokens and avoids double-counting P inside R.
    """
    stage_check = "post_generation" if unified_prompt_text is not None else "pre_filter"
    C, cap_source, resolved_model = resolve_model_context_window(model_id, state)

    if C is None or C <= 0:
        return {
            "passed": False,
            "evidence_status": "insufficient_evidence",
            "stage_check": stage_check,
            "rejection_reason": "context_window_unavailable",
            "model_id": resolved_model,
            "context_window_capacity": None,
            "context_window_source": cap_source,
            "prompt_tokens_P": None,
            "original_combined_prompt_tokens": None,
            "runtime_input_tokens_R": None,
            "reserved_output_tokens_O": None,
            "total_projected_tokens": None,
            "budget_limit_tokens": None,
            "max_utilization_pct": round(MAX_CONTEXT_UTILIZATION_RATIO * 100.0, 1),
            "utilization_pct": None,
            "double_counting_avoided": True,
            "tokenization_method": None,
            "is_exact_tokenizer": None,
            "per_agent_breakdown": {},
        }

    # Validate explicit mean_prompt_tokens and mean_completion_tokens per agent
    missing_agents: List[str] = []
    mean_in_map: Dict[str, float] = {}
    mean_out_map: Dict[str, float] = {}
    for aid in agent_ids:
        tm = token_metrics_map.get(aid) or {}
        mean_in = tm.get("mean_prompt_tokens")
        mean_out = tm.get("mean_completion_tokens")
        obs = tm.get("observation_count", 0)
        if (
            not isinstance(mean_in, (int, float))
            or not isinstance(mean_out, (int, float))
            or isinstance(mean_in, bool)
            or isinstance(mean_out, bool)
            or mean_in < 0
            or mean_out < 0
            or not isinstance(obs, (int, float))
            or obs <= 0
        ):
            missing_agents.append(aid)
        else:
            mean_in_map[aid] = float(mean_in)
            mean_out_map[aid] = float(mean_out)

    budget_limit = round(MAX_CONTEXT_UTILIZATION_RATIO * C, 4)
    if missing_agents:
        return {
            "passed": False,
            "evidence_status": "insufficient_evidence",
            "stage_check": stage_check,
            "rejection_reason": f"missing_or_unreliable_token_evidence:{','.join(missing_agents)}",
            "model_id": resolved_model,
            "context_window_capacity": C,
            "context_window_source": cap_source,
            "prompt_tokens_P": None,
            "original_combined_prompt_tokens": None,
            "runtime_input_tokens_R": None,
            "reserved_output_tokens_O": None,
            "total_projected_tokens": None,
            "budget_limit_tokens": budget_limit,
            "max_utilization_pct": round(MAX_CONTEXT_UTILIZATION_RATIO * 100.0, 1),
            "utilization_pct": None,
            "double_counting_avoided": True,
            "tokenization_method": None,
            "is_exact_tokenizer": None,
            "per_agent_breakdown": {},
        }

    per_agent_breakdown: Dict[str, Dict[str, Any]] = {}
    static_tok_map: Dict[str, int] = {}
    dynamic_input_map: Dict[str, float] = {}
    tok_method = "empty"
    is_exact = True

    for aid in agent_ids:
        p_text, p_source = extract_agent_static_prompt(aid, prompt_metrics_map, profile_map)
        tok_info = count_tokens(p_text, resolved_model)
        static_tok = tok_info["tokens"]
        if tok_info["method"] != "empty":
            tok_method = tok_info["method"]
            is_exact = is_exact and bool(tok_info["is_exact_tokenizer"])

        # Remove static prompt portion already counted in P to prevent double-counting
        dynamic_in = max(0.0, mean_in_map[aid] - float(static_tok))
        static_tok_map[aid] = static_tok
        dynamic_input_map[aid] = dynamic_in

        per_agent_breakdown[aid] = {
            "prompt_source": p_source,
            "static_prompt_tokens": static_tok,
            "mean_input_tokens": round(mean_in_map[aid], 2),
            "mean_output_tokens": round(mean_out_map[aid], 2),
            "estimated_dynamic_input_tokens": round(dynamic_in, 2),
        }

    original_combined_p = sum(static_tok_map.values())
    if unified_prompt_text is not None:
        unified_tok_info = count_tokens(unified_prompt_text, resolved_model)
        P = unified_tok_info["tokens"]
        tok_method = unified_tok_info["method"]
        is_exact = bool(unified_tok_info["is_exact_tokenizer"])
    else:
        P = original_combined_p

    # External runtime input: entry agent dynamic input + any external fan-in inputs
    group_set = set(agent_ids)
    parents_map = dep_parents_map or {}
    entry_plus_external = dynamic_input_map[agent_ids[0]] if agent_ids else 0.0
    for aid in agent_ids[1:]:
        ext_parents = parents_map.get(aid, set()) - group_set
        if ext_parents:
            entry_plus_external += dynamic_input_map[aid]

    max_single_dynamic = max(dynamic_input_map.values()) if dynamic_input_map else 0.0
    R = math.ceil(max(entry_plus_external, max_single_dynamic))
    O = math.ceil(sum(mean_out_map.values()))

    total_projected = P + R + O
    utilization_pct = round((total_projected / C) * 100.0, 2)
    passed = total_projected <= (MAX_CONTEXT_UTILIZATION_RATIO * C + 1e-9)
    rejection_reason = (
        None
        if passed
        else f"context_utilization_{utilization_pct}pct_exceeds_70pct_limit"
    )

    return {
        "passed": passed,
        "evidence_status": "sufficient",
        "stage_check": stage_check,
        "rejection_reason": rejection_reason,
        "model_id": resolved_model,
        "context_window_capacity": C,
        "context_window_source": cap_source,
        "prompt_tokens_P": P,
        "original_combined_prompt_tokens": original_combined_p,
        "runtime_input_tokens_R": R,
        "reserved_output_tokens_O": O,
        "total_projected_tokens": total_projected,
        "budget_limit_tokens": budget_limit,
        "max_utilization_pct": round(MAX_CONTEXT_UTILIZATION_RATIO * 100.0, 1),
        "utilization_pct": utilization_pct,
        "double_counting_avoided": True,
        "tokenization_method": tok_method,
        "is_exact_tokenizer": is_exact,
        "per_agent_breakdown": per_agent_breakdown,
    }
