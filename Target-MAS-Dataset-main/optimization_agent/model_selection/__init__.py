"""
JEV-Based Agentic Model Selection Package for AWO.

Provides autonomous, evidence-grounded model selection recommendations
using TypeSafe JEV System One classification and Groq model discovery
orchestrated through a sequential LangGraph StateGraph.
"""

from optimization_agent.model_selection.orchestrator import (
    run_model_selection,
    build_model_selection_graph,
    AWOModelSelectionState,
)
from optimization_agent.model_selection.mcp_context_builder import (
    build_agent_context_from_mcp,
    calculate_workflow_token_reference,
)
from optimization_agent.model_selection.groq_catalog import (
    MODEL_CATALOG,
    get_live_groq_models,
    resolve_available_llms,
)
from optimization_agent.model_selection.eligibility_filter import (
    filter_eligible_models,
    agent_requires_tools,
    estimate_context_requirement,
)
from optimization_agent.model_selection.agent1_requirement_analyzer import (
    run_agent1_requirement_analyzer,
    extract_score_answer,
    AGENT1_QUESTIONS,
)
from optimization_agent.model_selection.agent2_model_selector import (
    run_agent2_model_selector,
    validate_model_selection,
)
from optimization_agent.model_selection.jev_client import call_jev

__all__ = [
    "run_model_selection",
    "build_model_selection_graph",
    "AWOModelSelectionState",
    "build_agent_context_from_mcp",
    "calculate_workflow_token_reference",
    "MODEL_CATALOG",
    "get_live_groq_models",
    "resolve_available_llms",
    "filter_eligible_models",
    "agent_requires_tools",
    "estimate_context_requirement",
    "run_agent1_requirement_analyzer",
    "extract_score_answer",
    "AGENT1_QUESTIONS",
    "run_agent2_model_selector",
    "validate_model_selection",
    "call_jev",
]

