"""
Stage 5 (Dimension G): Compact Payload Recommendation & Token Savings Math.

Builds compact typed payload recommendations from safe Python + validated Groq
decisions without mutating global workflow state, and calculates single-tokenizer
payload savings cleanly distinguishing runtime measurements from static schemas.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from optimization_agent.consolidation.token_budget import count_tokens
from optimization_agent.data_handoff.payload_analyzer import _DOCUMENT_FIELD_HINTS


def build_handoff_recommendation(
    handoff: Dict[str, Any],
    python_mapping: Dict[str, Any],
    groq_result: Optional[Dict[str, Any]],
    model_id: Optional[str],
    allocated_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Generate an `OptimizationCandidate` for `communication_data_handoff` if at least
    one safely removable/compactable field exists and achieves positive reduction.
    Returns None when all fields must be conservatively retained.
    """
    producer = handoff["producer_agent"]
    consumer = handoff["consumer_agent"]
    transmitted = list(handoff.get("transmitted_fields") or [])
    pa = handoff.get("payload_analysis") or {}
    cls_map = python_mapping.get("field_classifications") or {}

    protected_set = (
        set(python_mapping.get("required_fields") or [])
        | set(python_mapping.get("used_fields") or [])
        | set(python_mapping.get("conditional_fields_preserved") or [])
    )
    python_redundant = set(python_mapping.get("potentially_redundant_fields") or [])
    python_duplicate = set(python_mapping.get("duplicate_fields") or [])
    python_uncertain = set(python_mapping.get("uncertain_fields") or [])

    groq_redundant: set = set()
    groq_duplicate: set = set()
    if groq_result and groq_result.get("accepted") and groq_result.get("validated_result"):
        vres = groq_result["validated_result"]
        # Groq may ONLY resolve Stage 3 uncertain fields (never protected fields)
        groq_redundant = (set(vres.get("redundant_fields") or []) & python_uncertain) - protected_set
        groq_duplicate = (set(vres.get("duplicate_fields") or []) & python_uncertain) - protected_set

    excluded_set = (python_redundant | python_duplicate | groq_redundant | groq_duplicate) - protected_set
    if not excluded_set:
        return None

    fields_retained: List[str] = []
    fields_excluded: List[str] = []
    fields_replaced_with_ref: List[Dict[str, Any]] = []
    field_change_reasons: Dict[str, str] = {}

    large_doc_set = set(pa.get("large_text_or_document_fields") or [])
    for f in transmitted:
        if f in excluded_set:
            fields_excluded.append(f)
            if f in large_doc_set or f.strip().lower() in _DOCUMENT_FIELD_HINTS:
                fields_replaced_with_ref.append({
                    "field": f,
                    "reference_pointer": f"shared_workflow_store:{producer}/{f}",
                    "retained_in_shared_state": True,
                })
                field_change_reasons[f] = (
                    "Full document payload excluded from direct inter-agent transmission "
                    "(original remains accessible in shared workflow store)."
                )
            elif f in (python_duplicate | groq_duplicate):
                field_change_reasons[f] = "Duplicate value already carried by retained canonical field."
            else:
                field_change_reasons[f] = (
                    (cls_map.get(f) or {}).get("reason")
                    or "Confirmed redundant by downstream consumer requirement mapping."
                )
        else:
            fields_retained.append(f)
            if f in python_uncertain:
                field_change_reasons[f] = "Conservatively retained (uncertain necessity)."
            else:
                field_change_reasons[f] = (cls_map.get(f) or {}).get("reason") or "Required/used by consumer."

    if not fields_retained:
        # Never recommend an empty payload; preserve safely
        return None

    actual_payload = handoff.get("actual_payload")
    obs_count = int(handoff.get("observation_count") or 0)
    proposed_schema = {
        "retained_fields": fields_retained,
        "excluded_fields": fields_excluded,
        "shared_store_references": fields_replaced_with_ref,
        "typed_delta_projection": True,
    }

    if isinstance(actual_payload, dict) and actual_payload:
        proposed_payload = {k: actual_payload[k] for k in fields_retained if k in actual_payload}
        orig_json = json.dumps(actual_payload, sort_keys=True, ensure_ascii=False)
        prop_json = json.dumps(proposed_payload, sort_keys=True, ensure_ascii=False)

        orig_tok_info = count_tokens(orig_json, model_id)
        prop_tok_info = count_tokens(prop_json, model_id)
        if orig_tok_info["method"] != prop_tok_info["method"] and orig_tok_info["method"] != "empty" and prop_tok_info["method"] != "empty":
            raise ValueError("Tokenizer method mismatch between original and proposed payloads.")

        orig_tokens: Optional[int] = int(orig_tok_info["tokens"])
        prop_tokens: Optional[int] = int(prop_tok_info["tokens"])
        tokens_saved: Optional[int] = orig_tokens - prop_tokens
        if tokens_saved <= 0:
            return None

        pct_saved: Optional[float] = round((tokens_saved / orig_tokens) * 100.0, 2) if orig_tokens > 0 else 0.0
        orig_bytes: Optional[int] = len(orig_json.encode("utf-8"))
        prop_bytes: Optional[int] = len(prop_json.encode("utf-8"))
        bytes_saved: Optional[int] = orig_bytes - prop_bytes
        byte_pct_saved: Optional[float] = round((bytes_saved / orig_bytes) * 100.0, 2) if orig_bytes > 0 else 0.0
        tok_method: Optional[str] = orig_tok_info["method"]
        is_exact_tok = bool(orig_tok_info["is_exact_tokenizer"] and prop_tok_info["is_exact_tokenizer"])
        measurement_type = "exact_tokenizer" if is_exact_tok else "estimated_heuristic_tokenizer"
        evidence_basis = "runtime_payload_observed"
        is_estimate = not is_exact_tok
    else:
        proposed_payload = None
        orig_tokens = None
        prop_tokens = None
        tokens_saved = None
        pct_saved = None
        orig_bytes = None
        prop_bytes = None
        bytes_saved = None
        byte_pct_saved = None
        tok_method = None
        is_exact_tok = None
        measurement_type = "static_schema_only"
        evidence_basis = "statically_inferred_schema_only"
        is_estimate = True

    remaining_uncertain = [f for f in python_uncertain if f not in excluded_set]

    observed_problem_parts = [
        f"Verified handoff '{producer} -> {consumer}' transmits {len(transmitted)} field(s) ({transmitted}), "
        f"of which {len(fields_excluded)} ({fields_excluded}) are redundant or duplicate for downstream consumption."
    ]
    if orig_tokens is not None:
        observed_problem_parts.append(
            f"Observed single-execution payload measures {orig_tokens} tokens ({orig_bytes} bytes, {evidence_basis}); "
            f"compact projection reduces payload to {prop_tokens} tokens ({prop_bytes} bytes, -{tokens_saved} tokens / {pct_saved}%)."
        )
    else:
        observed_problem_parts.append(
            "Runtime payload content is unavailable; field exclusions are statically inferred from declared schemas."
        )

    candidate: Dict[str, Any] = {
        "candidate_id": f"HANDOFF-{producer.upper()}-TO-{consumer.upper()}",
        "dimension": "communication_data_handoff",
        "title": f"Compact Handoff Payload Schema for {producer} -> {consumer}",
        "target_agents": [producer, consumer],
        "observed_problem": " ".join(observed_problem_parts),
        "supporting_evidence": {
            "mcp_stage_source": handoff.get("evidence_provenance", ["stage_3_graph", "stage_4_profiles", "stage_10_evidence"]),
            "producer_agent": producer,
            "consumer_agent": consumer,
            "verified_dependency_relationship": handoff.get("dependency_relationship", "declared_dependency"),
            "original_transmitted_fields": transmitted,
            "consumer_required_fields": list(python_mapping.get("required_fields") or []),
            "consumer_used_fields": list(python_mapping.get("used_fields") or []),
            "conditional_fields_preserved": list(python_mapping.get("conditional_fields_preserved") or []),
            "potentially_redundant_fields": sorted(python_redundant | groq_redundant),
            "duplicate_fields": sorted(python_duplicate | groq_duplicate),
            "uncertain_fields": remaining_uncertain,
            "fields_retained": fields_retained,
            "fields_excluded": fields_excluded,
            "fields_replaced_with_reference": fields_replaced_with_ref,
            "field_change_reasons": field_change_reasons,
            "original_payload_tokens": orig_tokens,
            "proposed_payload_tokens": prop_tokens,
            "tokens_saved": tokens_saved,
            "savings_percentage": pct_saved,
            "original_payload_bytes": orig_bytes,
            "proposed_payload_bytes": prop_bytes,
            "bytes_saved": bytes_saved,
            "byte_savings_percentage": byte_pct_saved,
            "observation_count": obs_count,
            "repeated_across_handoffs": pa.get("repeated_across_handoffs", []),
            "tokenization_method": tok_method,
            "is_exact_tokenizer": is_exact_tok,
            "measurement_type": measurement_type,
            "evidence_quality": handoff.get("evidence_quality"),
            "evidence_basis": evidence_basis,
            "original_payload_sample": actual_payload,
            "proposed_compact_payload_sample": proposed_payload,
            "python_analysis_results": {
                "payload_analysis": pa,
                "consumer_mapping": python_mapping,
            },
            "groq_semantic_analysis": groq_result,
            "evidence_provenance": handoff.get("evidence_provenance", []),
        },
        "proposed_compact_payload_schema": proposed_schema,
        "proposed_solution": (
            f"Transmit compact projected fields {fields_retained} from '{producer}' to '{consumer}' "
            f"(excluding {fields_excluded} from the inter-agent wire payload while preserving all global state)."
        ),
        "expected_benefit": {
            "original_payload_tokens_per_handoff": orig_tokens,
            "proposed_payload_tokens_per_handoff": prop_tokens,
            "tokens_saved_per_handoff": tokens_saved,
            "token_savings_pct": pct_saved,
            "original_payload_bytes_per_handoff": orig_bytes,
            "proposed_payload_bytes_per_handoff": prop_bytes,
            "bytes_saved_per_handoff": bytes_saved,
            "byte_savings_pct": byte_pct_saved,
            "fields_removed_count": len(fields_excluded),
            "fields_retained_count": len(fields_retained),
            "observation_count": obs_count,
            "estimated_repeated_transmissions_eliminated_per_run": len(pa.get("repeated_across_handoffs") or []) and len([f for f in fields_excluded if f in (pa.get("repeated_across_handoffs") or [])]) or 0,
            "evidence_basis": evidence_basis,
            "measurement_type": measurement_type,
            "is_estimate": is_estimate,
            "quality_impact": "Requires shadow validation to confirm consumer accuracy with compact payload.",
        },
        "tradeoffs_and_risks": {
            "latency_tradeoff": "Lower payload serialization/prompt-prefill volume; runtime delta requires validation.",
            "token_cost_tradeoff": (
                f"Projects {tokens_saved} fewer payload tokens per execution ({pct_saved}% reduction)."
                if tokens_saved is not None
                else "Projected token reduction requires runtime payload telemetry."
            ),
            "output_quality_risk": "Moderate (changing inter-agent handoff schemas requires regression verification).",
            "implementation_complexity": "Low",
            "shared_storage_reference_dependency": bool(fields_replaced_with_ref),
        },
        "tier": "Tier 2",
        "requires_validation": True,
        "allocated_key": allocated_key,
    }
    return candidate
