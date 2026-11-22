"""
JEV Agent 2: LLM Model Selector and Post-Selection Validator.

Constructs JEV Choice evaluation over deterministically eligible Groq models,
executes model selection, and rigorously validates the recommendation against safety invariants.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from optimization_agent.model_selection.jev_client import call_jev
from optimization_agent.model_selection.eligibility_filter import (
    agent_requires_tools,
    estimate_context_requirement,
)

logger = logging.getLogger("optimization_agent.model_selection.agent2")


def build_agent2_state(
    agent1_requirements: Dict[str, Any],
    agent_context: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Construct the contextual evaluation state for JEV Agent 2."""
    requires_tools = agent_requires_tools(agent_context)
    estimated_tokens = estimate_context_requirement(agent_context)

    return {
        "agent_requirements": {
            "task_complexity": agent1_requirements.get("task_complexity", {}).get("class"),
            "prompt_complexity": agent1_requirements.get("prompt_complexity", {}).get("class"),
            "token_consumption": agent1_requirements.get("token_consumption", {}).get("class"),
        },
        "agent_context": {
            "agent_id": agent_context.get("agent_definition", {}).get("agent_id"),
            "role": agent_context.get("agent_definition", {}).get("role"),
            "responsibilities": agent_context.get("agent_definition", {}).get("responsibilities"),
            "requires_tool_calling": requires_tools,
            "estimated_context_tokens_rough": estimated_tokens,
        },
        "selection_objective": (
            "Select a model for this general-purpose workflow agent. First satisfy hard "
            "requirements: general-purpose suitability, context/input capacity and required "
            "tool calling. Then compare reasoning and instruction-following capability with "
            "the classified task and prompt complexity. Among models that satisfy requirements, "
            "consider total cost and model size. Do not choose a specialized safety or prompt-"
            "injection classifier as the primary contract-review agent. Choose no_suitable_model "
            "if none is defensible. Qualitative capability labels are heuristic metadata."
        ),
        "available_llms": candidates,
    }


def validate_model_selection(
    selected: Optional[str],
    candidates: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, str]:
    """
    Rigorously validate JEV's chosen model ID against eligible candidates.

    Returns:
        (validated_model_id, validated_metadata, is_validated, reason)
    """
    candidate_ids = {m["model_id"] for m in candidates}

    if selected == "no_suitable_model":
        return None, None, True, "JEV selected no_suitable_model."

    if selected and selected in candidate_ids:
        chosen_meta = next(m for m in candidates if m["model_id"] == selected)
        if chosen_meta.get("suitable_for_general_agent_tasks") is not True:
            return (
                None,
                None,
                False,
                "JEV selection failed the general-purpose suitability rule.",
            )

        reason = (
            "Selected from the deterministically eligible general-purpose candidates. "
            f"Role={chosen_meta.get('model_role')}; "
            f"reasoning={chosen_meta.get('reasoning_capability')}; "
            f"instruction-following={chosen_meta.get('instruction_following')}; "
            f"context-window={chosen_meta.get('context_window')}; "
            f"tool-calling={chosen_meta.get('tool_calling_support')}."
        )
        return selected, chosen_meta, True, reason

    # Unknown or ineligible model ID
    return (
        None,
        None,
        False,
        "JEV returned an unknown or ineligible model ID. The recommendation was rejected by post-validation.",
    )


def run_agent2_model_selector(
    agent1_requirements: Dict[str, Any],
    agent_context: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    custom_call_jev: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute JEV Agent 2: LLM Model Selector.

    Args:
        agent1_requirements: The requirements classification from Agent 1.
        agent_context: The compact agent-specific evidence context.
        candidates: List of deterministically eligible candidate models.
        custom_call_jev: Optional injected call_jev function for testing/mocking.

    Returns:
        Structured recommendation dictionary.
    """
    if not candidates:
        return {
            "selected_model": None,
            "selected_model_metadata": None,
            "selection_validated": False,
            "reason": "No general-purpose Groq models passed deterministic eligibility checks.",
            "choice_probabilities": {},
            "choice_confidence": None,
            "jev_model": None,
            "jev_usage": {},
            "raw_answer": None,
        }

    # Safety check immediately before dispatching to JEV
    if any(m.get("suitable_for_general_agent_tasks") is not True for m in candidates):
        raise ValueError("A specialized model was passed to Agent 2. Aborting selection.")

    state = build_agent2_state(agent1_requirements, agent_context, candidates)

    options: Dict[str, str] = {
        "no_suitable_model": (
            "No listed model can defensibly satisfy the agent requirements, or the "
            "available capability metadata is insufficient."
        )
    }

    for model in candidates:
        mid = model["model_id"]
        options[mid] = (
            f"Role: {model.get('model_role')}; "
            f"context window: {model.get('context_window')}; "
            f"max input planning limit: {model.get('max_input_tokens')}; "
            f"max output: {model.get('max_output_tokens')}; "
            f"reasoning: {model.get('reasoning_capability')}; "
            f"instruction following: {model.get('instruction_following')}; "
            f"tool calling: {model.get('tool_calling_support')}; "
            f"input price per 1M tokens: {model.get('input_price_per_million_usd')}; "
            f"output price per 1M tokens: {model.get('output_price_per_million_usd')}; "
            f"metadata: {model.get('metadata_source')}"
        )

    questions = {
        "recommended_model": {
            "type": "choice",
            "instructions": (
                "Choose the most suitable candidate for the supplied agent. Respect hard "
                "requirements first. Task complexity informs reasoning capability; prompt "
                "complexity informs instruction-following; observed token consumption and "
                "estimated context inform capacity. Among candidates that meet requirements, "
                "prefer lower total cost only when the supplied pricing is comparable. Do not "
                "infer capabilities that are not stated. Choose no_suitable_model if evidence "
                "does not support a defensible choice."
            ),
            "criteria": options,
        }
    }

    caller = custom_call_jev or call_jev
    response = caller(state, questions)

    answers = response.get("answers") or {}
    answer = answers.get("recommended_model") or {}
    raw_choice = answer.get("choice") or answer.get("value")

    selected, chosen_meta, validated, reason = validate_model_selection(raw_choice, candidates)

    return {
        "selected_model": selected,
        "selected_model_metadata": chosen_meta,
        "selection_validated": validated,
        "reason": reason,
        "choice_probabilities": answer.get("probabilities", {}),
        "choice_confidence": answer.get("confidence"),
        "jev_model": response.get("model"),
        "jev_usage": response.get("usage", {}),
        "raw_answer": answer,
    }

