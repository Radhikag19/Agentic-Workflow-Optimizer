"""
Stage 4 (Dimension G): Optional Groq Semantic Analysis.

Invoked ONLY when deterministic Python mapping leaves semantically ambiguous
uncertain fields. Never overrides Python hard constraints (required, used,
or conditional fields).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from optimization_agent.consolidation.groq_client import GroqConsolidationClient

MIN_SEMANTIC_CONFIDENCE = 0.75

REQUIRED_GROQ_HANDOFF_FIELDS: Tuple[str, ...] = (
    "producer_agent",
    "consumer_agent",
    "field_assessments",
    "redundant_fields",
    "duplicate_fields",
    "required_fields",
    "uncertain_fields",
    "proposed_compact_payload_schema",
    "semantic_rationale",
    "confidence",
    "requires_human_or_empirical_validation",
)

HANDOFF_SEMANTIC_SYSTEM_PROMPT = (
    "You are analyzing an inter-agent data handoff in a multi-agent workflow.\n\n"
    "Determine which transmitted fields are essential for the downstream consumer and which fields "
    "may be redundant, duplicated, or replaceable with a smaller representation.\n\n"
    "Consider the consumer's actual responsibilities, declared input schema, task requirements, "
    "and workflow context.\n\n"
    "Do not recommend removing a field simply because it appears large or is not mentioned explicitly "
    "in one historical execution.\n\n"
    "Preserve all required business information, identifiers, decision evidence, exceptions, and constraints.\n\n"
    "If a field's necessity cannot be determined confidently, classify it as uncertain and recommend retaining it.\n\n"
    "Identify whether a large document or object can be replaced by a compact representation, "
    "such as selected excerpts, document references, changed fields, or typed delta objects.\n\n"
    "Return structured JSON with:\n\n"
    "- producer_agent\n"
    "- consumer_agent\n"
    "- field_assessments\n"
    "- redundant_fields\n"
    "- duplicate_fields\n"
    "- required_fields\n"
    "- uncertain_fields\n"
    "- proposed_compact_payload_schema\n"
    "- semantic_rationale\n"
    "- confidence\n"
    "- requires_human_or_empirical_validation"
)


def _as_field_list(raw: Any) -> Optional[List[str]]:
    if not isinstance(raw, list):
        return None
    out: List[str] = []
    for x in raw:
        if not isinstance(x, str):
            return None
        if x.strip():
            out.append(x.strip())
    return out


def validate_groq_semantic_output(
    expected_producer: str,
    expected_consumer: str,
    transmitted_fields: List[str],
    python_mapping: Dict[str, Any],
    raw_result: Optional[Dict[str, Any]],
    error: Optional[str] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Deterministically validate Groq's Stage 4 JSON response and enforce that
    Groq cannot override Python hard constraints (required/used/conditional fields).
    """
    if error or not isinstance(raw_result, dict):
        return False, None, f"groq_semantic_error:{error or 'null_response'}"

    missing = [k for k in REQUIRED_GROQ_HANDOFF_FIELDS if k not in raw_result]
    if missing:
        return False, None, f"missing_required_keys:{','.join(missing)}"

    if raw_result.get("producer_agent") != expected_producer or raw_result.get("consumer_agent") != expected_consumer:
        return False, None, "producer_or_consumer_agent_mismatch"

    conf_raw = raw_result.get("confidence")
    if not isinstance(conf_raw, (int, float)) or isinstance(conf_raw, bool) or conf_raw < MIN_SEMANTIC_CONFIDENCE:
        return False, None, f"low_confidence_or_invalid:{conf_raw}"

    req_list = _as_field_list(raw_result.get("required_fields"))
    red_list = _as_field_list(raw_result.get("redundant_fields"))
    dup_list = _as_field_list(raw_result.get("duplicate_fields"))
    unc_list = _as_field_list(raw_result.get("uncertain_fields"))
    if req_list is None or red_list is None or dup_list is None or unc_list is None:
        return False, None, "invalid_field_list_types"

    allowed: Set[str] = set(transmitted_fields)
    all_mentioned = set(req_list) | set(red_list) | set(dup_list) | set(unc_list)
    unsupported = sorted(all_mentioned - allowed)
    if unsupported:
        return False, None, f"unsupported_fields:{','.join(unsupported)}"

    # Contradiction checks
    if set(req_list).intersection(set(red_list) | set(dup_list)):
        return False, None, "contradictory_required_and_excluded_overlap"
    if set(unc_list).intersection(set(red_list) | set(dup_list)):
        return False, None, "contradictory_uncertain_and_excluded_overlap"

    # Hard constraint protection: Python's required, used, and conditional fields CANNOT be dropped
    protected: Set[str] = (
        set(python_mapping.get("required_fields") or [])
        | set(python_mapping.get("used_fields") or [])
        | set(python_mapping.get("conditional_fields_preserved") or [])
    )
    violations = sorted(protected.intersection(set(red_list) | set(dup_list)))
    if violations:
        return False, None, f"hard_constraint_violation_protected_fields:{','.join(violations)}"

    validated = dict(raw_result)
    validated["confidence"] = round(float(conf_raw), 4)
    validated["requires_human_or_empirical_validation"] = True
    return True, validated, None


def analyze_handoff_semantically_with_groq(
    handoff: Dict[str, Any],
    python_mapping: Dict[str, Any],
    profile_map: Dict[str, Dict[str, Any]],
    groq_client: GroqConsolidationClient,
) -> Dict[str, Any]:
    """Build Stage 4 payload, make one Groq call, and validate deterministically."""
    producer = handoff["producer_agent"]
    consumer = handoff["consumer_agent"]
    p_prof = profile_map.get(producer, {})
    c_prof = profile_map.get(consumer, {})
    pa = handoff.get("payload_analysis") or {}

    payload_snippet: Any = None
    if isinstance(handoff.get("actual_payload"), dict):
        payload_snippet = {
            k: (v[:300] + "..." if isinstance(v, str) and len(v) > 300 else v)
            for k, v in handoff["actual_payload"].items()
        }

    request_data = {
        "producer_agent": producer,
        "producer_role": p_prof.get("role"),
        "producer_goal": p_prof.get("goal"),
        "producer_output_schema": handoff.get("producer_declared_outputs", []),
        "consumer_agent": consumer,
        "consumer_role": c_prof.get("role"),
        "consumer_goal": c_prof.get("goal"),
        "consumer_input_schema": handoff.get("consumer_declared_inputs", []),
        "consumer_task_and_responsibilities": {
            "declared_responsibilities": c_prof.get("declared_responsibilities"),
            "tasks": [t.get("description") for t in (c_prof.get("tasks") or []) if isinstance(t, dict)],
        },
        "original_handoff_fields": handoff.get("transmitted_fields", []),
        "python_field_classification": python_mapping.get("field_classifications", {}),
        "execution_evidence_snippet": payload_snippet,
        "suspected_redundant_duplicate_or_uncertain_fields": {
            "potentially_redundant": python_mapping.get("potentially_redundant_fields", []),
            "duplicate": python_mapping.get("duplicate_fields", []),
            "uncertain": python_mapping.get("uncertain_fields", []),
            "repeated_across_handoffs": pa.get("repeated_across_handoffs", []),
        },
    }

    user_prompt = (
        "Analyze the following inter-agent data handoff and return structured JSON.\n"
        f"{json.dumps(request_data, indent=2)}"
    )
    raw_resp, err = groq_client.call_json(
        HANDOFF_SEMANTIC_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.0,
        max_tokens=2048,
    )

    accepted, validated, reason = validate_groq_semantic_output(
        expected_producer=producer,
        expected_consumer=consumer,
        transmitted_fields=handoff.get("transmitted_fields", []),
        python_mapping=python_mapping,
        raw_result=raw_resp,
        error=err,
    )
    if not accepted or validated is None:
        return {
            "used_groq": True,
            "accepted": False,
            "fallback_to_conservative_python": True,
            "rejection_reason": reason,
            "validated_result": None,
        }

    return {
        "used_groq": True,
        "accepted": True,
        "fallback_to_conservative_python": False,
        "rejection_reason": None,
        "validated_result": validated,
    }
