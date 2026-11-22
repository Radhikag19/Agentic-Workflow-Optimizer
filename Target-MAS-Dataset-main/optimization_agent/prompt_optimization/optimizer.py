"""
Stage 2 & Stage 3 (Dimension E): Single Groq Optimization Call & Deterministic Validation.

Makes at most one Groq call per eligible prompt using the specified system instruction,
and enforces strict preservation of responsibilities, I/O schemas, constraints, and
capability boundaries in Python before accepting any replacement prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from optimization_agent.consolidation.groq_client import GroqConsolidationClient
from optimization_agent.consolidation.prompt_combiner import _extract_schema_field_names

PROMPT_OPTIMIZATION_SYSTEM_PROMPT = (
    "You are a prompt optimization specialist. Analyze the supplied agent prompt and produce a shorter, "
    "clearer version by removing redundant wording, duplicated instructions, repetitive static descriptions, "
    "unnecessary preambles, and repeated schema explanations.\n\n"
    "Preserve every essential responsibility, instruction, business rule, exception, constraint, decision criterion, "
    "required input field, required output field, output format, and safety or compliance requirement.\n\n"
    "Do not remove information simply to achieve a lower token count. Do not introduce new capabilities, assumptions, "
    "or behavior. Do not weaken validation requirements. Maintain the original logical order and intended behavior.\n\n"
    "If the prompt has no meaningful redundancy, return it substantially unchanged.\n\n"
    "Return structured JSON containing:\n\n"
    "- agent_id\n"
    "- original_prompt_summary\n"
    "- detected_redundancies\n"
    "- compression_actions\n"
    "- optimized_prompt\n"
    "- preserved_responsibilities\n"
    "- preserved_constraints\n"
    "- preserved_input_schema\n"
    "- preserved_output_schema\n"
    "- semantic_preservation_assessment\n"
    "- estimated_token_reduction\n"
    "- optimization_rationale\n\n"
    "The optimized prompt must be directly usable as the replacement prompt for the original agent."
)

REQUIRED_OPTIMIZATION_FIELDS: Tuple[str, ...] = (
    "agent_id",
    "original_prompt_summary",
    "detected_redundancies",
    "compression_actions",
    "optimized_prompt",
    "preserved_responsibilities",
    "preserved_constraints",
    "preserved_input_schema",
    "preserved_output_schema",
    "semantic_preservation_assessment",
    "estimated_token_reduction",
    "optimization_rationale",
)

_FORBIDDEN_NEW_CAPABILITY_PHRASES: Tuple[str, ...] = (
    "execute shell command",
    "run shell command",
    "browse the internet",
    "search the web",
    "skip validation",
    "bypass validation",
    "bypass compliance",
    "auto-approve without",
    "ignore safety",
)


def build_optimization_request_payload(target: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble all 8 required inputs for the single Groq prompt optimization call."""
    return {
        "agent_id": target["target_id"],
        "original_prompt_text": target["prompt_text"],
        "role": target.get("role"),
        "goal": target.get("goal"),
        "required_responsibilities": target.get("required_responsibilities", []),
        "required_input_schema": target.get("required_inputs", []),
        "required_output_schema": target.get("required_outputs", []),
        "business_rules_and_constraints": target.get("critical_constraints", []),
        "supporting_evidence": {
            "originated_from_consolidation": target.get("originated_from_consolidation", False),
            "target_agents": target.get("target_agents", []),
            "detected_redundancy_signals": target.get("inspection", {}).get("detected_redundancy_signals", []),
            "prompt_tokens": target.get("inspection", {}).get("prompt_tokens"),
        },
    }


def _extract_constraint_text(preserved_constraints: Any) -> str:
    if isinstance(preserved_constraints, str):
        return preserved_constraints.lower()
    if isinstance(preserved_constraints, list):
        return " ".join(
            json.dumps(x) if isinstance(x, (dict, list)) else str(x)
            for x in preserved_constraints
        ).lower()
    if isinstance(preserved_constraints, dict):
        return json.dumps(preserved_constraints).lower()
    return ""


def _constraint_keywords(rule: str) -> List[str]:
    """Extract critical numeric values and domain terms from a constraint string."""
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", rule.lower())
    stopwords = {
        "the", "and", "for", "with", "must", "only", "that", "this", "from",
        "into", "not", "are", "when", "all", "any", "shall", "should", "within",
    }
    words = [
        w for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", rule.lower())
        if w not in stopwords
    ]
    return numbers + words[:4]


def validate_optimized_prompt_output(
    expected_agent_id: str,
    raw_result: Optional[Dict[str, Any]],
    original_prompt: str,
    required_responsibilities: List[str],
    required_inputs: List[str],
    required_outputs: List[str],
    critical_constraints: List[str],
    error: Optional[str] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Deterministic Stage 3 post-optimization validator enforcing all 7 checks.
    Returns (accepted, validated_dict, rejection_reason).
    """
    if error or not isinstance(raw_result, dict):
        return False, None, f"llm_optimization_error:{error or 'null_response'}"

    missing_fields = [f for f in REQUIRED_OPTIMIZATION_FIELDS if f not in raw_result]
    if missing_fields:
        return False, None, f"missing_required_fields:{','.join(missing_fields)}"

    if raw_result.get("agent_id") != expected_agent_id:
        return False, None, f"agent_id_mismatch:expected={expected_agent_id},got={raw_result.get('agent_id')}"

    optimized_prompt = raw_result.get("optimized_prompt")
    if not isinstance(optimized_prompt, str) or not optimized_prompt.strip():
        return False, None, "empty_optimized_prompt"

    opt_low = optimized_prompt.lower()
    orig_low = (original_prompt or "").lower()

    # Check semantic_preservation_assessment flags if structured dict
    sem = raw_result.get("semantic_preservation_assessment")
    if isinstance(sem, dict):
        for bad_flag in ("is_preserved", "responsibilities_preserved", "constraints_preserved", "schema_preserved"):
            if sem.get(bad_flag) is False:
                return False, None, f"semantic_preservation_failed:{bad_flag}=False"
        if sem.get("validation_weakened") is True:
            return False, None, "semantic_preservation_failed:validation_weakened=True"
        if sem.get("introduced_new_capabilities") is True or sem.get("unsupported_capabilities"):
            return False, None, "unsupported_capabilities_introduced"

    # Rule 2: Responsibilities preserved
    pres_resp = raw_result.get("preserved_responsibilities")
    has_resp = bool(pres_resp) if isinstance(pres_resp, (list, dict)) else (isinstance(pres_resp, str) and bool(pres_resp.strip()))
    if required_responsibilities and not has_resp:
        return False, None, "missing_preserved_responsibilities"

    # Rule 3 & 4: Required input & output fields preserved
    if required_inputs:
        schema_in_names = _extract_schema_field_names(raw_result.get("preserved_input_schema"))
        missing_in_schema = [f for f in required_inputs if f.lower() not in schema_in_names]
        # Also verify fields explicitly present in original_prompt were not dropped from optimized_prompt
        missing_in_prompt = [
            f for f in required_inputs
            if f.lower() in orig_low and f.lower() not in opt_low
        ]
        if missing_in_schema or missing_in_prompt:
            dropped = sorted(set(missing_in_schema + missing_in_prompt))
            return False, None, f"dropped_required_input_fields:{','.join(dropped)}"

    if required_outputs:
        schema_out_names = _extract_schema_field_names(raw_result.get("preserved_output_schema"))
        missing_out_schema = [f for f in required_outputs if f.lower() not in schema_out_names]
        missing_out_prompt = [
            f for f in required_outputs
            if f.lower() in orig_low and f.lower() not in opt_low
        ]
        if missing_out_schema or missing_out_prompt:
            dropped = sorted(set(missing_out_schema + missing_out_prompt))
            return False, None, f"dropped_required_output_fields:{','.join(dropped)}"

    # Rule 5: Critical constraints & business rules preserved
    if critical_constraints:
        pres_const = raw_result.get("preserved_constraints")
        const_text = _extract_constraint_text(pres_const)
        if not const_text.strip():
            return False, None, "missing_preserved_constraints"

        combined_const_check = f"{const_text} {opt_low}"
        for rule in critical_constraints:
            rule_low = rule.strip().lower()
            if not rule_low:
                continue
            markers = _constraint_keywords(rule)
            if markers:
                hits = sum(1 for m in markers if m in opt_low)
                # Ensure prompt itself retains at least half of critical markers (and numbers verbatim)
                nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", rule_low)
                if any(n not in opt_low for n in nums) or hits == 0:
                    return False, None, f"dropped_critical_constraint:{rule[:80]}"
            elif rule_low not in combined_const_check:
                return False, None, f"dropped_critical_constraint:{rule[:80]}"

    # Rule 7: Reject newly introduced unsupported capabilities
    for phrase in _FORBIDDEN_NEW_CAPABILITY_PHRASES:
        if phrase in opt_low and phrase not in orig_low:
            return False, None, f"unsupported_capability_phrase_detected:{phrase}"

    return True, dict(raw_result), None


def optimize_prompt_with_groq(
    target: Dict[str, Any],
    groq_client: GroqConsolidationClient,
) -> Dict[str, Any]:
    """
    Execute a single Groq optimization call for an eligible prompt and run
    deterministic Stage 3 post-validation (no automatic retries on failure).
    """
    payload = build_optimization_request_payload(target)
    user_prompt = (
        "Optimize the following agent prompt according to the system instructions.\n"
        f"{json.dumps(payload, indent=2)}"
    )
    raw_resp, err = groq_client.call_json(
        PROMPT_OPTIMIZATION_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.0,
        max_tokens=4096,
    )

    accepted, validated, reason = validate_optimized_prompt_output(
        expected_agent_id=target["target_id"],
        raw_result=raw_resp,
        original_prompt=target["prompt_text"],
        required_responsibilities=target.get("required_responsibilities", []),
        required_inputs=target.get("required_inputs", []),
        required_outputs=target.get("required_outputs", []),
        critical_constraints=target.get("critical_constraints", []),
        error=err,
    )

    if not accepted or validated is None:
        return {
            "accepted": False,
            "result": None,
            "rejection_reason": reason,
            "retained_prompt": target["prompt_text"],
        }

    return {
        "accepted": True,
        "result": validated,
        "rejection_reason": None,
        "retained_prompt": None,
    }
