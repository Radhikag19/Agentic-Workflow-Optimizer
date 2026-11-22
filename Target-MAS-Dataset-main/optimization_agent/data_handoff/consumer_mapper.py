"""
Stage 3 (Dimension G): Downstream Consumer Requirement Mapping (Deterministic Python).

Maps each transmitted field to: required | used | duplicate | potentially_redundant | uncertain
strictly preserving declared consumer inputs, conditional/fallback fields, and
task-required context while isolating genuine semantic ambiguity for optional Groq review.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from optimization_agent.consolidation.token_budget import extract_agent_static_prompt
from optimization_agent.data_handoff.payload_analyzer import (
    _DOCUMENT_FIELD_HINTS,
    _METADATA_FIELD_HINTS,
)

_CONDITIONAL_MARKER_RE = re.compile(
    r"\b(?:if|when|whenever|unless|optional|fallback|exception|warning|missing|conditional|in case)\b",
    re.IGNORECASE,
)


def _tokenize_identifier(name: str) -> Set[str]:
    parts = [p.lower() for p in re.split(r"[^a-zA-Z0-9]+", name) if len(p) >= 3]
    # Include simple singular/plural stems for overlap checks
    stems = set(parts)
    for p in parts:
        if p.endswith("s") and len(p) >= 4:
            stems.add(p[:-1])
    return stems


def _is_field_explicitly_referenced(field: str, text: str) -> bool:
    if not text:
        return False
    f_low = field.strip().lower()
    if not f_low:
        return False
    pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(f_low)}(?![a-z0-9_])", re.IGNORECASE)
    return bool(pattern.search(text))


def _is_field_in_conditional_context(field: str, text: str) -> bool:
    """Return True if field appears in a conditional/exception/fallback sentence or line."""
    if not text:
        return False
    f_low = field.strip().lower()
    pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(f_low)}(?![a-z0-9_])", re.IGNORECASE)
    for segment in re.split(r"[\n.!?]+", text):
        if pattern.search(segment) and _CONDITIONAL_MARKER_RE.search(segment):
            return True
    return False


def map_consumer_requirements(
    handoff: Dict[str, Any],
    profile_map: Dict[str, Dict[str, Any]],
    prompt_metrics_map: Dict[str, Dict[str, Any]],
    execution_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classify every transmitted field in a handoff against downstream consumer needs.
    Determines whether Groq semantic disambiguation is genuinely required.
    """
    consumer_id = handoff["consumer_agent"]
    c_prof = profile_map.get(consumer_id, {})
    pa = handoff.get("payload_analysis") or {}
    transmitted = list(handoff.get("transmitted_fields") or [])

    c_declared_inputs = [
        str(x).strip()
        for x in (c_prof.get("declared_inputs") or [])
        if isinstance(x, str) and str(x).strip()
    ]
    c_declared_inputs_low = {x.lower() for x in c_declared_inputs}
    c_declared_outputs = [
        str(x).strip()
        for x in (c_prof.get("declared_outputs") or [])
        if isinstance(x, str) and str(x).strip()
    ]

    c_prompt_text, prompt_source = extract_agent_static_prompt(consumer_id, prompt_metrics_map, profile_map)
    task_texts = " ".join(
        str(t.get("description") or "")
        for t in (c_prof.get("tasks") or [])
        if isinstance(t, dict)
    )
    role_goal_resp = " ".join(
        str(c_prof.get(k) or "")
        for k in ("role", "goal", "backstory", "declared_responsibilities")
    )
    real_prompt_text = c_prompt_text if prompt_source != "mcp_profile_fallback" else ""
    combined_prompt_and_task = f"{real_prompt_text}\n{task_texts}".strip()
    combined_full_context = f"{role_goal_resp}\n{combined_prompt_and_task}\n{' '.join(c_declared_outputs)}".strip()
    context_tokens = _tokenize_identifier(combined_full_context)

    c_ev = execution_evidence.get(consumer_id) if isinstance(execution_evidence, dict) else {}
    c_out = (c_ev or {}).get("output") or (c_ev or {}).get("sample_outputs")
    c_out_keys_low = {str(k).strip().lower() for k in c_out.keys()} if isinstance(c_out, dict) else set()

    has_explicit_consumer_schema = len(c_declared_inputs_low) > 0
    has_consumer_prompt_evidence = bool(combined_prompt_and_task.strip())

    dup_lookup: Dict[str, str] = {}
    for pair in (pa.get("duplicate_value_pairs") or []):
        dup_lookup[pair["duplicate_field"]] = pair["canonical_field"]

    doc_fields_set = set(pa.get("large_text_or_document_fields") or [])
    meta_fields_set = set(pa.get("repeated_metadata_fields") or [])
    large_json_set = set(pa.get("large_json_objects") or [])
    repeated_set = set(pa.get("repeated_across_handoffs") or [])

    # Check if consumer explicitly expects structured extracted items while full doc is also sent
    has_structured_extracts_required = any(
        f.lower() in c_declared_inputs_low or _is_field_explicitly_referenced(f, combined_prompt_and_task)
        for f in transmitted
        if f.lower() not in _DOCUMENT_FIELD_HINTS and f.lower() not in _METADATA_FIELD_HINTS
    )

    classifications: Dict[str, Dict[str, Any]] = {}
    required_fields: List[str] = []
    used_fields: List[str] = []
    duplicate_fields: List[str] = []
    potentially_redundant_fields: List[str] = []
    uncertain_fields: List[str] = []
    conditional_fields: List[str] = []
    ambiguity_reasons: List[str] = []

    for f in transmitted:
        f_low = f.strip().lower()
        is_cond = _is_field_in_conditional_context(f, combined_prompt_and_task)
        in_decl_in = f_low in c_declared_inputs_low
        explicit_placeholder = bool(
            re.search(
                rf"(?:\{{{re.escape(f_low)}\}}|`{re.escape(f_low)}`|'{re.escape(f_low)}'|\"{re.escape(f_low)}\")",
                combined_prompt_and_task,
                re.IGNORECASE,
            )
        )
        in_prompt_or_task = _is_field_explicitly_referenced(f, combined_prompt_and_task)
        in_goal_or_resp = _is_field_explicitly_referenced(f, role_goal_resp)
        in_c_outputs = _is_field_explicitly_referenced(f, " ".join(c_declared_outputs)) or (f_low in c_out_keys_low)

        if in_decl_in or explicit_placeholder or is_cond:
            cls = "required"
            reason = (
                "Explicitly listed in consumer declared_inputs"
                if in_decl_in
                else ("Conditional/fallback reference in consumer task instructions" if is_cond else "Explicit input field placeholder in consumer prompt")
            )
            required_fields.append(f)
            if is_cond:
                conditional_fields.append(f)
        elif in_prompt_or_task or in_goal_or_resp or in_c_outputs:
            cls = "used"
            reason = "Referenced by consumer prompt, task/goal responsibilities, or propagated outputs."
            used_fields.append(f)
        elif f in dup_lookup:
            cls = "duplicate"
            reason = f"Exact serialized value duplicate of retained field '{dup_lookup[f]}'."
            duplicate_fields.append(f)
        else:
            f_stems = _tokenize_identifier(f)
            stem_overlap = f_stems.intersection(context_tokens)
            is_doc_hint = f_low in _DOCUMENT_FIELD_HINTS or f in doc_fields_set
            is_meta_hint = f_low in _METADATA_FIELD_HINTS or f in meta_fields_set or "metadata" in f_low or "debug" in f_low

            if not has_consumer_prompt_evidence and not has_explicit_consumer_schema:
                cls = "uncertain"
                reason = "Insufficient consumer schema and prompt evidence; treating necessity conservatively."
                uncertain_fields.append(f)
                ambiguity_reasons.append(f"Field '{f}': missing consumer input schema and prompt evidence.")
            elif has_explicit_consumer_schema and has_consumer_prompt_evidence:
                if (is_doc_hint and has_structured_extracts_required) or is_meta_hint or not stem_overlap:
                    cls = "potentially_redundant"
                    reason = (
                        "Unreferenced full-document payload alongside required structured fields."
                        if is_doc_hint
                        else (
                            "Unreferenced metadata/debug field not required by consumer schema or prompt."
                            if is_meta_hint
                            else "Not in consumer declared_inputs and unreferenced in consumer task/prompt."
                        )
                    )
                    potentially_redundant_fields.append(f)
                else:
                    cls = "uncertain"
                    reason = (
                        f"Absent from declared_inputs, but shares domain stem(s) {sorted(stem_overlap)} "
                        f"with consumer instructions/goal."
                    )
                    uncertain_fields.append(f)
                    ambiguity_reasons.append(f"Field '{f}': semantic stem overlap {sorted(stem_overlap)} despite missing from declared_inputs.")
            else:
                # Partial evidence (e.g., prompt exists but declared_inputs is empty, or declared_inputs without prompt)
                if is_meta_hint and has_consumer_prompt_evidence and not stem_overlap:
                    cls = "potentially_redundant"
                    reason = "Unreferenced metadata/debug field with no overlap in consumer prompt."
                    potentially_redundant_fields.append(f)
                else:
                    cls = "uncertain"
                    reason = "Partial consumer contract metadata leaves field necessity semantically ambiguous."
                    uncertain_fields.append(f)
                    ambiguity_reasons.append(f"Field '{f}': incomplete consumer schema/prompt boundary.")

        classifications[f] = {
            "field": f,
            "classification": cls,
            "is_conditional": is_cond,
            "reason": reason,
            "bytes": (pa.get("per_field_metrics") or {}).get(f, {}).get("bytes"),
            "tokens": (pa.get("per_field_metrics") or {}).get(f, {}).get("tokens"),
        }

    # Trigger Groq ONLY when semantic ambiguity exists AND consumer has real prompt/goal/responsibility evidence
    has_consumer_semantic_evidence = bool(
        has_consumer_prompt_evidence
        or str(c_prof.get("goal") or "").strip()
        or str(c_prof.get("declared_responsibilities") or "").strip()
    )
    has_impactful_uncertain = any(
        u in doc_fields_set
        or u in large_json_set
        or u in meta_fields_set
        or u in repeated_set
        or ((pa.get("per_field_metrics") or {}).get(u, {}).get("tokens") or 0) >= 25
        or not has_explicit_consumer_schema
        for u in uncertain_fields
    )
    needs_groq = bool(uncertain_fields and has_impactful_uncertain and has_consumer_semantic_evidence)

    return {
        "producer_agent": handoff["producer_agent"],
        "consumer_agent": consumer_id,
        "consumer_declared_inputs": c_declared_inputs,
        "consumer_prompt_source": prompt_source,
        "field_classifications": classifications,
        "required_fields": required_fields,
        "used_fields": used_fields,
        "potentially_redundant_fields": potentially_redundant_fields,
        "duplicate_fields": duplicate_fields,
        "uncertain_fields": uncertain_fields,
        "conditional_fields_preserved": conditional_fields,
        "needs_groq_semantic_analysis": needs_groq,
        "semantic_ambiguity_reasons": ambiguity_reasons,
    }
