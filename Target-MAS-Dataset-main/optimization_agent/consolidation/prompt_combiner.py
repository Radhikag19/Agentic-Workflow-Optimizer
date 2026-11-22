"""
Stage 4: Groq LLM-Based Prompt Combination & Deterministic Validation.

Generates unified system/task prompts only for compatible sequential execution groups
that already survived Stages 1-3, verifies strict schema/agent-ID/compatibility preservation,
and reapplies the deterministic 70% context-window constraint on the generated prompt.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from optimization_agent.consolidation.groq_client import GroqConsolidationClient
from optimization_agent.consolidation.token_budget import (
    count_tokens,
    evaluate_context_budget,
    extract_agent_static_prompt,
)

REQUIRED_COMBINATION_FIELDS: Tuple[str, ...] = (
    "group_id",
    "source_agent_ids",
    "combined_agent_name",
    "combined_role",
    "combined_goal",
    "unified_prompt",
    "preserved_responsibilities",
    "merged_responsibilities",
    "removed_redundancies",
    "preserved_input_schema",
    "preserved_output_schema",
    "compatibility_assessment",
    "combination_rationale",
    "estimated_combined_prompt_tokens",
    "requires_validation",
)

PROMPT_COMBINATION_SYSTEM_PROMPT = """You are an expert Multi-Agent System Prompt Architect performing Dimension B (Agent Consolidation).
Given a strictly ordered sequence of compatible non-validator agents, produce a unified prompt that consolidates their tasks.

MANDATORY REQUIREMENTS:
1. Preserve EVERY essential task and responsibility across all source agents.
2. Preserve EVERY required external input field and required downstream output field.
3. Preserve ALL business rules, thresholds, exceptions, compliance guardrails, and decision criteria.
4. Preserve important domain-specific instructions verbatim or faithfully.
5. Maintain the exact logical execution order of the original agents' steps.
6. Eliminate duplicated role descriptions, repeated preamble text, and redundant boilerplate.
7. Remove unnecessary intermediate handoff formatting between consolidated internal steps.
8. Do NOT introduce any new capabilities, tools, or ungrounded assumptions.
9. Do NOT weaken validation, compliance, safety, or formatting constraints.
10. Explicitly report any incompatible responsibilities or conflicting instructions in `compatibility_assessment`.
11. `source_agent_ids` MUST match the exact ordered list provided.
12. `requires_validation` MUST be true.

Return valid JSON ONLY containing ALL of these exact keys:
{
  "group_id": "<string>",
  "source_agent_ids": ["<agent_1>", "<agent_2>", "..."],
  "combined_agent_name": "<string>",
  "combined_role": "<string>",
  "combined_goal": "<string>",
  "unified_prompt": "<complete consolidated instruction prompt>",
  "preserved_responsibilities": ["<string>"],
  "merged_responsibilities": ["<string>"],
  "removed_redundancies": ["<string>"],
  "preserved_input_schema": ["<required input field or schema spec>"],
  "preserved_output_schema": ["<required output field or schema spec>"],
  "compatibility_assessment": {
    "is_compatible": true,
    "incompatible_responsibilities": [],
    "notes": "<string>"
  },
  "combination_rationale": "<string>",
  "estimated_combined_prompt_tokens": <integer>,
  "requires_validation": true
}"""


def check_group_io_compatibility(
    group_agents: List[str],
    dep_parents_map: Dict[str, Set[str]],
    dep_children_map: Dict[str, Set[str]],
    profile_map: Dict[str, Dict[str, Any]],
    classifications: Dict[str, Dict[str, Any]],
) -> Tuple[bool, str]:
    """Deterministic pre-generation check of Stage 3 exclusions and dependency ordering."""
    if len(group_agents) < 2:
        return False, "group_requires_at_least_two_agents"

    for aid in group_agents:
        cls = classifications.get(aid)
        if not cls:
            return False, f"missing_stage3_classification:{aid}"
        if cls.get("excluded_from_consolidation", True):
            return False, f"stage3_excluded:{aid}:{cls.get('exclusion_category')}"

    pos = {aid: idx for idx, aid in enumerate(group_agents)}
    for aid in group_agents:
        for parent in dep_parents_map.get(aid, set()):
            if parent in pos and pos[parent] >= pos[aid]:
                return False, f"dependency_order_conflict:{parent}_must_precede_{aid}"
        for child in dep_children_map.get(aid, set()):
            if child in pos and pos[child] <= pos[aid]:
                return False, f"dependency_order_conflict:{aid}_must_precede_{child}"

    return True, "compatible_sequential_execution_group"


def compute_required_group_schemas(
    group_agents: List[str],
    profile_map: Dict[str, Dict[str, Any]],
    dep_parents_map: Dict[str, Set[str]],
    dep_children_map: Dict[str, Set[str]],
) -> Tuple[List[str], List[str]]:
    """
    Compute external input fields and terminal/downstream output fields that the
    consolidated agent MUST preserve.
    """
    produced_so_far: Set[str] = set()
    required_inputs: List[str] = []
    seen_in: Set[str] = set()

    for idx, aid in enumerate(group_agents):
        prof = profile_map.get(aid, {})
        for field in (prof.get("declared_inputs") or []):
            if isinstance(field, str) and field.strip():
                norm = field.strip()
                if (idx == 0 or norm.lower() not in produced_so_far) and norm.lower() not in seen_in:
                    required_inputs.append(norm)
                    seen_in.add(norm.lower())
        for out_field in (prof.get("declared_outputs") or []):
            if isinstance(out_field, str) and out_field.strip():
                produced_so_far.add(out_field.strip().lower())

    group_set = set(group_agents)
    downstream_external_inputs: Set[str] = set()
    for aid in group_agents:
        for child in dep_children_map.get(aid, set()):
            if child not in group_set:
                c_prof = profile_map.get(child, {})
                for ci in (c_prof.get("declared_inputs") or []):
                    if isinstance(ci, str) and ci.strip():
                        downstream_external_inputs.add(ci.strip().lower())

    required_outputs: List[str] = []
    seen_out: Set[str] = set()
    last_agent = group_agents[-1]
    for aid in group_agents:
        prof = profile_map.get(aid, {})
        for out_field in (prof.get("declared_outputs") or []):
            if isinstance(out_field, str) and out_field.strip():
                norm = out_field.strip()
                if (aid == last_agent or norm.lower() in downstream_external_inputs) and norm.lower() not in seen_out:
                    required_outputs.append(norm)
                    seen_out.add(norm.lower())

    return required_inputs, required_outputs


def _extract_schema_field_names(schema_val: Any) -> Set[str]:
    """Extract normalized field names from list/dict/string schema structures."""
    names: Set[str] = set()
    if isinstance(schema_val, dict):
        props = schema_val.get("properties")
        if isinstance(props, dict):
            names.update(str(k).strip().lower() for k in props.keys() if str(k).strip())
        for k, v in schema_val.items():
            if k not in ("type", "required", "description", "properties"):
                names.add(str(k).strip().lower())
            if isinstance(v, (dict, list)):
                names.update(_extract_schema_field_names(v))
    elif isinstance(schema_val, list):
        for item in schema_val:
            if isinstance(item, str) and item.strip():
                # Handle "field: description" strings safely
                token = item.strip().split(":", 1)[0].split(" ", 1)[0].strip()
                if token:
                    names.add(token.lower())
                names.add(item.strip().lower())
            elif isinstance(item, dict):
                if isinstance(item.get("name"), str):
                    names.add(item["name"].strip().lower())
                elif isinstance(item.get("field"), str):
                    names.add(item["field"].strip().lower())
                else:
                    names.update(_extract_schema_field_names(item))
    elif isinstance(schema_val, str) and schema_val.strip():
        names.add(schema_val.strip().lower())
    return names


def validate_combined_prompt_output(
    group_id: str,
    expected_agent_ids: List[str],
    raw_result: Optional[Dict[str, Any]],
    required_inputs: List[str],
    required_outputs: List[str],
    error: Optional[str] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str], str]:
    """
    Validate LLM prompt combination JSON output against all mandatory schema, ordering,
    responsibility-compatibility, and required-field constraints.

    Returns (is_valid, validated_result, rejection_reason, rejection_category).
    """
    if error or not isinstance(raw_result, dict):
        return (
            False,
            None,
            f"llm_combination_error:{error or 'null_response'}",
            "llm_or_schema_error",
        )

    missing_fields = [k for k in REQUIRED_COMBINATION_FIELDS if k not in raw_result]
    if missing_fields:
        return (
            False,
            None,
            f"missing_required_fields:{','.join(missing_fields)}",
            "llm_or_schema_error",
        )

    if raw_result.get("source_agent_ids") != list(expected_agent_ids):
        return (
            False,
            None,
            f"unsupported_or_reordered_agent_ids:expected={expected_agent_ids},got={raw_result.get('source_agent_ids')}",
            "llm_or_schema_error",
        )

    unified_prompt = raw_result.get("unified_prompt")
    if not isinstance(unified_prompt, str) or not unified_prompt.strip():
        return False, None, "empty_unified_prompt", "llm_or_schema_error"

    compat = raw_result.get("compatibility_assessment")
    if isinstance(compat, dict):
        incompat_items = compat.get("incompatible_responsibilities") or []
        if compat.get("is_compatible") is False or (isinstance(incompat_items, list) and len(incompat_items) > 0):
            return (
                False,
                None,
                f"incompatible_responsibilities:{incompat_items}",
                "incompatible_responsibilities",
            )
    elif isinstance(compat, str) and "incompatible" in compat.lower() and "no incompatible" not in compat.lower():
        return (
            False,
            None,
            f"incompatible_responsibilities:{compat[:160]}",
            "incompatible_responsibilities",
        )

    # Check required I/O field preservation
    if required_inputs:
        in_names = _extract_schema_field_names(raw_result.get("preserved_input_schema"))
        prompt_low = unified_prompt.lower()
        dropped_in = [f for f in required_inputs if f.lower() not in in_names and f.lower() not in prompt_low]
        if dropped_in:
            return (
                False,
                None,
                f"dropped_required_input_fields:{','.join(dropped_in)}",
                "llm_or_schema_error",
            )

    if required_outputs:
        out_names = _extract_schema_field_names(raw_result.get("preserved_output_schema"))
        prompt_low = unified_prompt.lower()
        dropped_out = [f for f in required_outputs if f.lower() not in out_names and f.lower() not in prompt_low]
        if dropped_out:
            return (
                False,
                None,
                f"dropped_required_output_fields:{','.join(dropped_out)}",
                "llm_or_schema_error",
            )

    validated = dict(raw_result)
    validated["group_id"] = group_id
    validated["source_agent_ids"] = list(expected_agent_ids)
    validated["requires_validation"] = True
    return True, validated, None, ""


def generate_and_validate_unified_prompt(
    group_id: str,
    group_agents: List[str],
    groq_client: GroqConsolidationClient,
    model_id: Optional[str],
    profile_map: Dict[str, Dict[str, Any]],
    prompt_metrics_map: Dict[str, Dict[str, Any]],
    token_metrics_map: Dict[str, Dict[str, Any]],
    dep_parents_map: Dict[str, Set[str]],
    dep_children_map: Dict[str, Set[str]],
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Invoke Groq to combine prompts for a compatible group, validate all required JSON fields
    and schema preservation, and reapply the 70% context-window budget on the output.
    """
    required_inputs, required_outputs = compute_required_group_schemas(
        group_agents, profile_map, dep_parents_map, dep_children_map
    )

    ordered_agent_specs: List[Dict[str, Any]] = []
    for aid in group_agents:
        prof = profile_map.get(aid, {})
        prompt_text, prompt_source = extract_agent_static_prompt(aid, prompt_metrics_map, profile_map)
        ordered_agent_specs.append({
            "agent_id": aid,
            "role": prof.get("role"),
            "goal": prof.get("goal"),
            "backstory": prof.get("backstory"),
            "declared_responsibilities": prof.get("declared_responsibilities"),
            "declared_inputs": prof.get("declared_inputs", []),
            "declared_outputs": prof.get("declared_outputs", []),
            "tasks": prof.get("tasks", []),
            "prompt_source": prompt_source,
            "prompt_text": prompt_text,
        })

    request_payload = {
        "group_id": group_id,
        "execution_order": list(group_agents),
        "required_preserved_inputs": required_inputs,
        "required_preserved_outputs": required_outputs,
        "agents": ordered_agent_specs,
    }
    user_prompt = (
        "Combine the following ordered sequential execution agents into a single unified prompt.\n"
        f"{json.dumps(request_payload, indent=2)}"
    )

    raw_resp, err = groq_client.call_json(
        PROMPT_COMBINATION_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.0,
        max_tokens=4096,
    )

    is_valid, validated, reject_reason, reject_cat = validate_combined_prompt_output(
        group_id=group_id,
        expected_agent_ids=group_agents,
        raw_result=raw_resp,
        required_inputs=required_inputs,
        required_outputs=required_outputs,
        error=err,
    )

    if not is_valid or validated is None:
        return {
            "status": "rejected_incompatible" if reject_cat == "incompatible_responsibilities" else "rejected_llm_or_schema",
            "rejection_reason": reject_reason,
            "rejection_category": reject_cat,
            "combination_result": None,
            "post_budget": None,
            "prompt_generated": False,
        }

    # Independent deterministic post-generation tokenization & 70% context check
    post_budget = evaluate_context_budget(
        agent_ids=group_agents,
        model_id=model_id,
        prompt_metrics_map=prompt_metrics_map,
        token_metrics_map=token_metrics_map,
        profile_map=profile_map,
        dep_parents_map=dep_parents_map,
        unified_prompt_text=validated["unified_prompt"],
        state=state,
    )
    tok_count = count_tokens(validated["unified_prompt"], post_budget.get("model_id"))
    validated["measured_unified_prompt_tokens"] = tok_count["tokens"]
    validated["measured_tokenization_method"] = tok_count["method"]

    if not post_budget.get("passed", False):
        return {
            "status": "rejected_post_tokenization",
            "rejection_reason": post_budget.get("rejection_reason"),
            "rejection_category": "post_generation_context_exceeded",
            "combination_result": validated,
            "post_budget": post_budget,
            "prompt_generated": True,
        }

    return {
        "status": "accepted",
        "rejection_reason": None,
        "rejection_category": "",
        "combination_result": validated,
        "post_budget": post_budget,
        "prompt_generated": True,
    }
