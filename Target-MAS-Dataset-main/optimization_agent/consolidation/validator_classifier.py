"""
Stage 3: Groq LLM-Based Independent-Validation Responsibility Detection.

Classifies remaining candidate agents by inspecting actual responsibilities, tasks,
I/O contracts, dependencies, prompts, and execution evidence (NOT just agent names).
Deterministically prevents independent validators and uncertain agents from
being consolidated with execution agents, while retaining them in the workflow.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from optimization_agent.consolidation.groq_client import GroqConsolidationClient
from optimization_agent.consolidation.token_budget import extract_agent_static_prompt

VALID_RESPONSIBILITY_TYPES: Set[str] = {
    "independent_validation_verification",
    "auditing_compliance_checking",
    "quality_assurance_independent_review",
    "execution_extraction_transformation_analysis_synthesis",
    "mixed_or_unclear",
}

VALIDATOR_TYPES: Set[str] = {
    "independent_validation_verification",
    "auditing_compliance_checking",
    "quality_assurance_independent_review",
}

MIN_CLASSIFICATION_CONFIDENCE = 0.70

CLASSIFICATION_SYSTEM_PROMPT = """You are an autonomous Multi-Agent System responsibility classifier for Dimension B (Agent Consolidation).
Analyze the agent strictly from its actual responsibilities, goal, backstory, task descriptions, input/output contracts, dependencies, prompt instructions, and runtime execution evidence.

CRITICAL RULES:
1. Do NOT rely solely on agent names or keywords such as "validation", "review", or "audit". A neutrally named agent (e.g., "step_four_agent") may still act as an independent validator, while an agent with "validation" in its name might only do initial field extraction or normalization. Determine responsibility from actual semantics and evidence.
2. Classify `responsibility_type` as EXACTLY one of:
   - "independent_validation_verification": performs independent checking, verification, or gating of upstream agent outputs.
   - "auditing_compliance_checking": performs independent compliance auditing or policy violation verification.
   - "quality_assurance_independent_review": performs independent QA, peer review, or sign-off gatekeeping.
   - "execution_extraction_transformation_analysis_synthesis": performs primary extraction, parsing, normalization, domain analysis, transformation, drafting, or synthesis.
   - "mixed_or_unclear": mixes independent validation with primary execution, or evidence is insufficient/ambiguous.
3. Set `is_independent_validator` to true if and only if the agent performs independent validation, auditing, compliance verification, or QA independent review.
4. Return valid JSON ONLY with EXACTLY these keys:
{
  "agent_id": "<exact agent_id>",
  "responsibility_type": "<one of the 5 allowed types>",
  "is_independent_validator": <boolean>,
  "classification_confidence": <float between 0.0 and 1.0>,
  "evidence": ["<concrete evidence point referencing responsibilities/prompts/tasks/outputs>"],
  "reasoning": "<concise grounded explanation>",
  "consolidation_restrictions": ["<any restriction, or empty list>"]
}"""


def build_agent_classification_payload(
    agent_id: str,
    profile_map: Dict[str, Dict[str, Any]],
    prompt_metrics_map: Dict[str, Dict[str, Any]],
    dep_parents_map: Dict[str, Set[str]],
    dep_children_map: Dict[str, Set[str]],
    execution_evidence: Dict[str, Any],
    agent_run_comparisons: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble grounded MCP evidence payload for Stage 3 LLM responsibility detection."""
    prof = profile_map.get(agent_id, {})
    prompt_text, prompt_source = extract_agent_static_prompt(agent_id, prompt_metrics_map, profile_map)

    tasks_raw = prof.get("tasks", []) or []
    task_descriptions = [
        {
            "task_name": t.get("task_name"),
            "description": (t.get("description") or "")[:1500],
            "expected_output": (t.get("expected_output") or "")[:600],
        }
        for t in tasks_raw
        if isinstance(t, dict)
    ]

    ee = execution_evidence.get(agent_id, {}) if isinstance(execution_evidence, dict) else {}
    sample_out = ee.get("sample_outputs")
    if isinstance(sample_out, (dict, list)):
        sample_out_str = json.dumps(sample_out)[:1000]
    elif isinstance(sample_out, str):
        sample_out_str = sample_out[:1000]
    else:
        sample_out_str = None

    sample_in = ee.get("sample_inputs")
    if isinstance(sample_in, (dict, list)):
        sample_in_str = json.dumps(sample_in)[:600]
    elif isinstance(sample_in, str):
        sample_in_str = sample_in[:600]
    else:
        sample_in_str = None

    comp = agent_run_comparisons.get(agent_id, {}) if isinstance(agent_run_comparisons, dict) else {}

    return {
        "agent_id": agent_id,
        "role": prof.get("role"),
        "goal": prof.get("goal"),
        "backstory": prof.get("backstory"),
        "declared_responsibilities": prof.get("declared_responsibilities"),
        "task_description": task_descriptions,
        "input_output_contracts": {
            "declared_inputs": prof.get("declared_inputs", []),
            "declared_outputs": prof.get("declared_outputs", []),
        },
        "dependencies": {
            "upstream": sorted(dep_parents_map.get(agent_id, set())),
            "downstream": sorted(dep_children_map.get(agent_id, set())),
        },
        "prompt_text": prompt_text[:2500] if prompt_text else None,
        "prompt_source": prompt_source,
        "execution_evidence": {
            "observed_execution_orders": ee.get("observed_execution_orders"),
            "sample_inputs_snippet": sample_in_str,
            "sample_outputs_snippet": sample_out_str,
            "runs_compared_count": comp.get("runs_compared_count"),
        },
    }


def validate_classification_output(
    expected_agent_id: str,
    raw_result: Optional[Dict[str, Any]],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministic post-validation for Stage 3 LLM responsibility output.
    Conservatively excludes any uncertain, contradictory, or unsupported result.
    """
    required_keys = {
        "agent_id",
        "responsibility_type",
        "is_independent_validator",
        "classification_confidence",
        "evidence",
        "reasoning",
        "consolidation_restrictions",
    }

    if error or not isinstance(raw_result, dict):
        return {
            "agent_id": expected_agent_id,
            "responsibility_type": "mixed_or_unclear",
            "is_independent_validator": False,
            "classification_confidence": 0.0,
            "evidence": [],
            "reasoning": f"Conservative exclusion due to Groq classification failure ({error or 'null_response'}).",
            "consolidation_restrictions": ["conservatively_excluded_llm_failure"],
            "excluded_from_consolidation": True,
            "exclusion_category": "conservatively_excluded_uncertain",
            "retain_separately": True,
            "validation_passed": False,
        }

    missing_keys = sorted(k for k in required_keys if k not in raw_result)
    if missing_keys or raw_result.get("agent_id") != expected_agent_id:
        return {
            "agent_id": expected_agent_id,
            "responsibility_type": "mixed_or_unclear",
            "is_independent_validator": False,
            "classification_confidence": 0.0,
            "evidence": [],
            "reasoning": (
                f"Conservative exclusion: missing required keys {missing_keys} or agent_id mismatch "
                f"(got '{raw_result.get('agent_id')}')."
            ),
            "consolidation_restrictions": ["conservatively_excluded_malformed_output"],
            "excluded_from_consolidation": True,
            "exclusion_category": "conservatively_excluded_uncertain",
            "retain_separately": True,
            "validation_passed": False,
        }

    resp_type = raw_result.get("responsibility_type")
    is_val = raw_result.get("is_independent_validator")
    conf_raw = raw_result.get("classification_confidence")
    evidence = raw_result.get("evidence")
    reasoning = raw_result.get("reasoning")
    restrictions = raw_result.get("consolidation_restrictions")

    conf = float(conf_raw) if isinstance(conf_raw, (int, float)) and not isinstance(conf_raw, bool) else 0.0
    if isinstance(evidence, str):
        ev_list = [evidence.strip()] if evidence.strip() else []
    elif isinstance(evidence, list):
        ev_list = [str(x).strip() for x in evidence if str(x).strip()]
    else:
        ev_list = []

    if isinstance(restrictions, list):
        rest_list = [str(x).strip() for x in restrictions if str(x).strip()]
    else:
        rest_list = []

    contradictory = (
        not isinstance(is_val, bool)
        or (resp_type in VALIDATOR_TYPES and is_val is False)
        or (resp_type == "execution_extraction_transformation_analysis_synthesis" and is_val is True)
    )
    uncertain_or_unsupported = (
        resp_type not in VALID_RESPONSIBILITY_TYPES
        or resp_type == "mixed_or_unclear"
        or conf < MIN_CLASSIFICATION_CONFIDENCE
        or len(ev_list) == 0
        or not isinstance(reasoning, str)
        or not reasoning.strip()
    )

    if contradictory:
        return {
            "agent_id": expected_agent_id,
            "responsibility_type": resp_type if resp_type in VALID_RESPONSIBILITY_TYPES else "mixed_or_unclear",
            "is_independent_validator": bool(is_val) if isinstance(is_val, bool) else False,
            "classification_confidence": round(conf, 4),
            "evidence": ev_list,
            "reasoning": f"Conservative exclusion due to contradictory responsibility flags: {reasoning}",
            "consolidation_restrictions": rest_list or ["conservatively_excluded_contradictory_flags"],
            "excluded_from_consolidation": True,
            "exclusion_category": "conservatively_excluded_uncertain",
            "retain_separately": True,
            "validation_passed": False,
        }

    if resp_type in VALIDATOR_TYPES and is_val is True and conf >= MIN_CLASSIFICATION_CONFIDENCE and ev_list:
        return {
            "agent_id": expected_agent_id,
            "responsibility_type": resp_type,
            "is_independent_validator": True,
            "classification_confidence": round(conf, 4),
            "evidence": ev_list,
            "reasoning": reasoning.strip(),
            "consolidation_restrictions": rest_list or ["never_consolidate_with_execution_agent"],
            "excluded_from_consolidation": True,
            "exclusion_category": "independent_validator",
            "retain_separately": True,
            "validation_passed": True,
        }

    if uncertain_or_unsupported:
        return {
            "agent_id": expected_agent_id,
            "responsibility_type": resp_type if resp_type in VALID_RESPONSIBILITY_TYPES else "mixed_or_unclear",
            "is_independent_validator": bool(is_val) if isinstance(is_val, bool) else False,
            "classification_confidence": round(conf, 4),
            "evidence": ev_list,
            "reasoning": (
                f"Conservative exclusion due to uncertain/low-confidence/unsupported classification "
                f"(type={resp_type}, confidence={conf:.2f}, evidence_count={len(ev_list)}): {reasoning or ''}"
            ).strip(),
            "consolidation_restrictions": rest_list or ["conservatively_excluded_uncertain_responsibility"],
            "excluded_from_consolidation": True,
            "exclusion_category": "conservatively_excluded_uncertain",
            "retain_separately": True,
            "validation_passed": False,
        }

    # Eligible execution/extraction/transformation/analysis/synthesis agent
    return {
        "agent_id": expected_agent_id,
        "responsibility_type": resp_type,
        "is_independent_validator": False,
        "classification_confidence": round(conf, 4),
        "evidence": ev_list,
        "reasoning": reasoning.strip(),
        "consolidation_restrictions": rest_list,
        "excluded_from_consolidation": False,
        "exclusion_category": None,
        "retain_separately": False,
        "validation_passed": True,
    }


def classify_agents_with_groq(
    agent_ids: List[str],
    groq_client: GroqConsolidationClient,
    profile_map: Dict[str, Dict[str, Any]],
    prompt_metrics_map: Dict[str, Dict[str, Any]],
    dep_parents_map: Dict[str, Set[str]],
    dep_children_map: Dict[str, Set[str]],
    execution_evidence: Dict[str, Any],
    agent_run_comparisons: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Run Stage 3 Groq classification and deterministic validation on unique candidate agents."""
    results: Dict[str, Dict[str, Any]] = {}
    seen: Set[str] = set()

    for aid in agent_ids:
        if aid in seen:
            continue
        seen.add(aid)

        payload = build_agent_classification_payload(
            agent_id=aid,
            profile_map=profile_map,
            prompt_metrics_map=prompt_metrics_map,
            dep_parents_map=dep_parents_map,
            dep_children_map=dep_children_map,
            execution_evidence=execution_evidence,
            agent_run_comparisons=agent_run_comparisons,
        )
        user_prompt = (
            "Classify the operational responsibility of the following MAS agent using the evidence JSON below.\n"
            f"{json.dumps(payload, indent=2)}"
        )
        raw_resp, err = groq_client.call_json(CLASSIFICATION_SYSTEM_PROMPT, user_prompt, temperature=0.0)
        results[aid] = validate_classification_output(aid, raw_resp, error=err)

    return results
