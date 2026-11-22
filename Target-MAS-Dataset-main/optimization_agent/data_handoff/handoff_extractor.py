"""
Stage 1 (Dimension G): Deterministic Inter-Agent Handoff Extraction.

Extracts verified producer-consumer handoff records strictly from declared
workflow dependencies (never inferring dependencies solely from observed
execution order) and preserves explicit runtime vs static evidence provenance.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from optimization_agent.consolidation.segmentation import _normalize_agent_id
from optimization_agent.consolidation.token_budget import count_tokens

_GENERIC_IO_MARKERS = {"all_previous_stages", "previous_stage", "input", "context", ""}


def _parse_payload_dict(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse a dictionary payload from execution evidence if available."""
    if isinstance(raw, dict) and raw:
        return dict(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict) and parsed:
                    return parsed
            except Exception:
                return None
    return None


def _extract_agent_runtime_payloads(ev_record: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return (runtime_input_dict, runtime_output_dict) from Stage 10 execution evidence."""
    if not isinstance(ev_record, dict):
        return None, None
    in_dict = _parse_payload_dict(ev_record.get("input")) or _parse_payload_dict(ev_record.get("sample_inputs"))
    out_dict = _parse_payload_dict(ev_record.get("output")) or _parse_payload_dict(ev_record.get("sample_outputs"))
    return in_dict, out_dict


def extract_verified_handoffs(
    state: Dict[str, Any],
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract verified producer-consumer handoff records from workflow_graph and MCP state.
    Never infers dependencies purely from observed execution order and never fabricates sizes.
    """
    discovered_agents = list(state.get("discovered_agents") or [])
    valid_agents: Set[str] = set(discovered_agents)
    workflow_graph = dict(state.get("workflow_graph") or {})
    edges_raw = workflow_graph.get("edges") or []
    agent_profiles = dict(state.get("agent_profiles") or {})
    token_metrics = dict((state.get("token_metrics") or {}).get("agent_token_metrics") or {})
    exec_evidence = dict(state.get("execution_evidence") or {})
    run_comparisons = dict(state.get("agent_run_comparisons") or {})

    profile_map: Dict[str, Dict[str, Any]] = {
        p["agent_id"]: p
        for p in (agent_profiles.get("profiles") or [])
        if isinstance(p, dict) and p.get("agent_id")
    }

    warnings: List[str] = []
    verified_edges: List[Tuple[str, str]] = []
    seen_edges: Set[Tuple[str, str]] = set()
    observed_edges: Set[Tuple[str, str]] = set()
    malformed_count = 0

    for edge in edges_raw:
        if not isinstance(edge, dict):
            continue
        rel = edge.get("relationship")
        u = _normalize_agent_id(edge.get("source"))
        v = _normalize_agent_id(edge.get("target"))

        if rel in ("declared_dependency", None):
            if u in valid_agents and v in valid_agents and u != v:
                if (u, v) not in seen_edges:
                    seen_edges.add((u, v))
                    verified_edges.append((u, v))
            elif rel == "declared_dependency" and (not u or not v):
                malformed_count += 1
        elif rel == "observed_execution_order":
            if u in valid_agents and v in valid_agents and u != v:
                observed_edges.add((u, v))

    if malformed_count:
        warnings.append(
            f"Ignored {malformed_count} malformed declared_dependency edge(s) with missing agent IDs."
        )

    if not verified_edges:
        warnings.append(
            "No verified producer-consumer declared dependencies found in workflow_graph; observed execution order alone is never treated as a handoff dependency."
        )
        return {
            "handoffs": [],
            "dependency_evidence_status": "insufficient_dependency_evidence",
            "warnings": warnings,
        }

    handoffs: List[Dict[str, Any]] = []
    for producer, consumer in verified_edges:
        p_prof = profile_map.get(producer, {})
        c_prof = profile_map.get(consumer, {})
        producer_declared_outputs = [
            str(x).strip()
            for x in (p_prof.get("declared_outputs") or [])
            if isinstance(x, str) and str(x).strip()
        ]
        consumer_declared_inputs = [
            str(x).strip()
            for x in (c_prof.get("declared_inputs") or [])
            if isinstance(x, str) and str(x).strip()
        ]

        _, prod_out_dict = _extract_agent_runtime_payloads(exec_evidence.get(producer) or {})
        cons_in_dict, _ = _extract_agent_runtime_payloads(exec_evidence.get(consumer) or {})

        # Prefer explicit consumer runtime input when recorded; otherwise producer runtime output
        if cons_in_dict:
            actual_payload = cons_in_dict
            payload_source = "runtime_consumer_input"
        elif prod_out_dict:
            actual_payload = prod_out_dict
            payload_source = "runtime_producer_output"
        else:
            actual_payload = None
            payload_source = "statically_inferred_declared_outputs"

        if actual_payload is not None:
            transmitted_fields = list(actual_payload.keys())
            serialized = json.dumps(actual_payload, sort_keys=True, ensure_ascii=False)
            payload_size_bytes: Optional[int] = len(serialized.encode("utf-8"))
            tok_info = count_tokens(serialized, model_id)
            payload_tokens: Optional[int] = int(tok_info["tokens"])
            tok_method: Optional[str] = tok_info["method"]
            is_exact_tok: Optional[bool] = bool(tok_info["is_exact_tokenizer"])
            evidence_quality = "runtime_observed"
            is_static = False
        else:
            transmitted_fields = list(producer_declared_outputs)
            payload_size_bytes = None
            payload_tokens = None
            tok_method = None
            is_exact_tok = None
            evidence_quality = "statically_inferred"
            is_static = True

        obs_count = int(
            (token_metrics.get(producer) or {}).get("observation_count")
            or (run_comparisons.get(producer) or {}).get("runs_compared_count")
            or 0
        )

        handoffs.append({
            "handoff_id": f"{producer}->{consumer}",
            "producer_agent": producer,
            "consumer_agent": consumer,
            "producer_declared_outputs": producer_declared_outputs,
            "consumer_declared_inputs": consumer_declared_inputs,
            "producer_runtime_output_fields": list(prod_out_dict.keys()) if prod_out_dict else [],
            "consumer_runtime_input_fields": list(cons_in_dict.keys()) if cons_in_dict else [],
            "transmitted_fields": transmitted_fields,
            "actual_payload": actual_payload,
            "payload_size_bytes": payload_size_bytes,
            "payload_tokens": payload_tokens,
            "tokenization_method": tok_method,
            "is_exact_tokenizer": is_exact_tok,
            "observation_count": obs_count,
            "dependency_relationship": "declared_dependency",
            "is_statically_inferred": is_static,
            "evidence_quality": evidence_quality,
            "payload_source": payload_source,
            "confidence": "high" if not is_static else "moderate_static",
            "evidence_provenance": [
                "stage_3_graph",
                "stage_4_profiles",
                "stage_10_evidence" if not is_static else "static_schema_inference",
            ],
        })

    return {
        "handoffs": handoffs,
        "dependency_evidence_status": "verified",
        "warnings": warnings,
    }
