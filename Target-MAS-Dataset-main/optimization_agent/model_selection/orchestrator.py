"""
LangGraph Orchestrator for JEV-Based Agentic Model Selection.

Builds and executes the sequential StateGraph:
START -> requirement_analyzer (Agent 1) -> model_selector (Agent 2) -> END
and compiles the standardized final structured recommendation report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from optimization_agent.model_selection.agent1_requirement_analyzer import (
    run_agent1_requirement_analyzer,
)
from optimization_agent.model_selection.agent2_model_selector import (
    run_agent2_model_selector,
)
from optimization_agent.model_selection.eligibility_filter import (
    filter_eligible_models,
)
from optimization_agent.model_selection.groq_catalog import (
    resolve_available_llms,
)
from optimization_agent.model_selection.mcp_context_builder import (
    build_agent_context_from_mcp,
)

logger = logging.getLogger("optimization_agent.model_selection.orchestrator")


class AWOModelSelectionState(TypedDict, total=False):
    """LangGraph State representation for Agentic Model Selection."""
    workflow_id: str
    selected_agent_id: str
    agent_context: Dict[str, Any]
    eligible_models: List[Dict[str, Any]]
    agent1_requirements: Dict[str, Any]
    agent2_recommendation: Dict[str, Any]
    custom_call_jev: Optional[Any]


def requirement_analyzer_node(state: AWOModelSelectionState) -> Dict[str, Any]:
    """LangGraph Node 1: Execute JEV Agent 1 (Requirement Analyzer)."""
    context = state["agent_context"]
    custom_caller = state.get("custom_call_jev")
    reqs = run_agent1_requirement_analyzer(context, custom_call_jev=custom_caller)
    return {"agent1_requirements": reqs}


def model_selector_node(state: AWOModelSelectionState) -> Dict[str, Any]:
    """LangGraph Node 2: Execute JEV Agent 2 (LLM Model Selector)."""
    reqs = state["agent1_requirements"]
    context = state["agent_context"]
    candidates = state["eligible_models"]
    custom_caller = state.get("custom_call_jev")
    recom = run_agent2_model_selector(reqs, context, candidates, custom_call_jev=custom_caller)
    return {"agent2_recommendation": recom}


def build_model_selection_graph() -> Any:
    """Build and compile the 2-node sequential LangGraph StateGraph."""
    graph_builder = StateGraph(AWOModelSelectionState)
    graph_builder.add_node("requirement_analyzer", requirement_analyzer_node)
    graph_builder.add_node("model_selector", model_selector_node)

    graph_builder.add_edge(START, "requirement_analyzer")
    graph_builder.add_edge("requirement_analyzer", "model_selector")
    graph_builder.add_edge("model_selector", END)

    return graph_builder.compile()


def run_model_selection(
    workflow_id: str,
    agent_id: str,
    evidence_dir: Optional[str] = None,
    custom_call_jev: Optional[Any] = None,
    live_groq_models: Optional[Dict[str, Any]] = None,
    export_report: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-End Execution of JEV-Based Agentic Model Selection using MCP Evidence.

    Args:
        workflow_id: The target workflow (e.g., 'contract_review').
        agent_id: The specific agent to evaluate (e.g., 'clause_mapping_agent').
        evidence_dir: Optional custom evidence directory path.
        custom_call_jev: Optional injected JEV call function (for testing/mocking).
        live_groq_models: Optional pre-fetched Groq models mapping.
        export_report: Whether to persist the structured JSON result to disk.
        output_dir: Optional output directory for saving the result.

    Returns:
        Structured JSON-compatible dictionary following the Colab prototype schema.
    """
    # 1. MCP Evidence Retrieval and Context Building
    agent_context = build_agent_context_from_mcp(
        workflow_id=workflow_id,
        agent_id=agent_id,
        evidence_dir=evidence_dir,
    )

    # 2. Groq Model Discovery and Capability Merging
    available_llms, missing_from_api = resolve_available_llms(live_models=live_groq_models)

    # 3. Deterministic Eligibility Filtering
    eligible_llms, excluded_llms = filter_eligible_models(
        available_llms=available_llms,
        agent_context=agent_context,
    )

    # 4. LangGraph Sequential Execution
    graph = build_model_selection_graph()
    initial_state: AWOModelSelectionState = {
        "workflow_id": workflow_id,
        "selected_agent_id": agent_id,
        "agent_context": agent_context,
        "eligible_models": eligible_llms,
        "custom_call_jev": custom_call_jev,
    }

    final_state = graph.invoke(initial_state)

    requirements = final_state.get("agent1_requirements", {})
    recommendation = final_state.get("agent2_recommendation", {})

    workflow_info = agent_context.get("workflow", {})
    baseline_model = workflow_info.get("baseline_model") or "openai/gpt-oss-120b"
    selected_model = recommendation.get("selected_model")

    token_ref = agent_context.get("token_consumption_reference", {})
    low_cut = token_ref.get("low_upper_boundary_approx")
    high_cut = token_ref.get("medium_upper_boundary_approx")
    scope = token_ref.get("comparison_scope")

    # 5. Build Standardized Recommendation Output
    final_output: Dict[str, Any] = {
        "workflow_id": workflow_id,
        "workflow_name": workflow_info.get("workflow_name", workflow_id),
        "selected_agent_id": agent_id,
        "baseline_model": baseline_model,
        "agent_requirements": requirements,
        "model_recommendation": recommendation,
        "eligible_model_ids": [m["model_id"] for m in eligible_llms],
        "excluded_models": excluded_llms,
        "token_reference": {
            "low_upper_boundary_approx": low_cut,
            "medium_upper_boundary_approx": high_cut,
            "scope": scope,
        },
        "selection_validation": {
            "validated_against_eligibility_rules": recommendation.get("selection_validated", False),
            "is_downgrade_candidate": bool(
                selected_model and baseline_model and selected_model != baseline_model
            ),
            "requires_empirical_quality_validation": True,
        },
        "limitations": [
            "JEV classifications and selections depend on the supplied evidence and prompt.",
            "Reasoning and instruction-following ratings are qualitative project heuristics.",
            "Derived maximum input token values are planning estimates, not separate provider-published limits.",
            "A capability-fit recommendation does not establish output-quality equivalence.",
        ],
    }

    # 6. Result Persistence
    if export_report:
        target_dir = Path(output_dir) if output_dir else Path.cwd() / "reports"
        target_dir.mkdir(parents=True, exist_ok=True)
        out_file = target_dir / f"awo_jev_recommendation_{agent_id}.json"
        out_file.write_text(json.dumps(final_output, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved AWO JEV Model Selection result to: {out_file}")

    return final_output

