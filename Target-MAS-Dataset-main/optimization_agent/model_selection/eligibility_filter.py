"""
Deterministic Hard-Constraint Eligibility Filter for Model Selection.

Enforces non-negotiable architectural safety invariants before candidate models
reach JEV Agent 2. Specialized classifiers and under-resourced models are strictly excluded.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("optimization_agent.model_selection.filter")


def agent_requires_tools(context: Dict[str, Any]) -> bool:
    """
    Determine whether an agent strictly requires tool-calling capabilities.

    Inspects declared_tools in agent_definition and tool assignments across task_information.
    """
    declared_tools = context.get("agent_definition", {}).get("registered_tools") or []
    task_tools: List[Any] = []
    for task in context.get("task_information", []):
        tools = task.get("tools") or []
        if isinstance(tools, list):
            task_tools.extend(tools)
        elif tools:
            task_tools.append(tools)
    return bool(declared_tools or task_tools)


def estimate_context_requirement(context: Dict[str, Any]) -> int:
    """
    Rough character-based heuristic for context token requirements.

    Formula: max(1, (character_count + 3) // 4)
    """
    serialized = json.dumps(context, ensure_ascii=False)
    return max(1, (len(serialized) + 3) // 4)


def filter_eligible_models(
    available_llms: List[Dict[str, Any]],
    agent_context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter available models using deterministic hard constraints.

    Hard Constraints Enforced:
      1. General-purpose suitability: suitable_for_general_agent_tasks == True.
      2. Tool-calling support: If agent requires tools, model must support tool-calling.
      3. Context window capacity: context_window >= estimated_context_tokens.
      4. Maximum input capacity: max_input_tokens >= estimated_context_tokens (if specified).
      5. Metadata completeness: Reasoning, instruction following, and output tokens must be valid.

    Returns:
        (eligible_llms, excluded_llms)
    """
    requires_tools = agent_requires_tools(agent_context)
    estimated_tokens = estimate_context_requirement(agent_context)

    eligible_llms: List[Dict[str, Any]] = []
    excluded_llms: List[Dict[str, Any]] = []

    for model in available_llms:
        reasons: List[str] = []
        mid = model.get("model_id", "UNKNOWN_MODEL")

        # 1. Hard Rule: Specialized classifiers cannot replace general-purpose agents
        if model.get("suitable_for_general_agent_tasks") is not True:
            reasons.append(
                f"Specialized model ({model.get('model_role')}); not approved for general-purpose agent tasks."
            )

        # 2. Tool-calling support
        if requires_tools and model.get("tool_calling_support") is not True:
            reasons.append("Agent requires tool calling, but this model is not marked as tool-capable.")

        # 3. Context window capacity
        context_window = model.get("context_window")
        if isinstance(context_window, (int, float)) and context_window < estimated_tokens:
            reasons.append(
                f"Context window ({context_window}) is below the rough estimated context requirement ({estimated_tokens})."
            )

        # 4. Maximum input planning limit
        max_input = model.get("max_input_tokens")
        if isinstance(max_input, (int, float)) and max_input < estimated_tokens:
            reasons.append(
                f"Maximum input planning limit ({max_input}) is below the rough estimated context requirement ({estimated_tokens})."
            )

        # 5. Metadata completeness for general-purpose models
        if model.get("suitable_for_general_agent_tasks") is True:
            if model.get("reasoning_capability") not in ("Low", "Medium", "High"):
                reasons.append("Reasoning capability metadata is missing or invalid.")
            if model.get("instruction_following") not in ("Low", "Medium", "High"):
                reasons.append("Instruction-following metadata is missing or invalid.")
            if not isinstance(model.get("max_output_tokens"), (int, float)):
                reasons.append("Maximum output-token metadata is missing.")

        if reasons:
            excluded_llms.append({"model_id": mid, "reasons": reasons})
        else:
            eligible_llms.append(model)

    # Invariant safety check
    assert all(m.get("suitable_for_general_agent_tasks") is True for m in eligible_llms), (
        "Eligibility error: a specialized model entered the general-purpose candidate list."
    )

    return eligible_llms, excluded_llms

