"""
LangGraph Workflow Orchestration for TraceProfiler Autonomous Optimization Discovery Agent.

Orchestrates the 13 sequential execution stages:
Stages 1 to 10: TraceProfiler MCP Evidence Server retrieval
Stage 11: Systematic multi-dimensional optimization discovery
Stage 12: Feasibility and trade-off assessment
Stage 13: Structured JSON and human-readable Markdown report compilation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from optimization_agent.discovery import discovery_analysis
from optimization_agent.key_manager import get_default_key_manager
from optimization_agent.pipeline import (
    stage_1_discovery,
    stage_2_overview,
    stage_3_graph,
    stage_4_profiles,
    stage_5_prompts,
    stage_6_latency,
    stage_7_tokens,
    stage_8_cost,
    stage_9_comparison,
    stage_10_evidence,
)
from optimization_agent.report import generate_report
from optimization_agent.state import OptimizationAgentState
from optimization_agent.tradeoff import tradeoff_assessment

logger = logging.getLogger("optimization_agent.agent")


def build_optimization_graph() -> Any:
    """Build and compile the 13-stage sequential LangGraph StateGraph."""
    builder = StateGraph(OptimizationAgentState)

    # Register all 13 nodes
    builder.add_node("stage_1_discovery", stage_1_discovery)
    builder.add_node("stage_2_overview", stage_2_overview)
    builder.add_node("stage_3_graph", stage_3_graph)
    builder.add_node("stage_4_profiles", stage_4_profiles)
    builder.add_node("stage_5_prompts", stage_5_prompts)
    builder.add_node("stage_6_latency", stage_6_latency)
    builder.add_node("stage_7_tokens", stage_7_tokens)
    builder.add_node("stage_8_cost", stage_8_cost)
    builder.add_node("stage_9_comparison", stage_9_comparison)
    builder.add_node("stage_10_evidence", stage_10_evidence)
    builder.add_node("discovery_analysis", discovery_analysis)
    builder.add_node("tradeoff_assessment", tradeoff_assessment)
    builder.add_node("generate_report", generate_report)

    # Strictly sequential execution order
    builder.add_edge(START, "stage_1_discovery")
    builder.add_edge("stage_1_discovery", "stage_2_overview")
    builder.add_edge("stage_2_overview", "stage_3_graph")
    builder.add_edge("stage_3_graph", "stage_4_profiles")
    builder.add_edge("stage_4_profiles", "stage_5_prompts")
    builder.add_edge("stage_5_prompts", "stage_6_latency")
    builder.add_edge("stage_6_latency", "stage_7_tokens")
    builder.add_edge("stage_7_tokens", "stage_8_cost")
    builder.add_edge("stage_8_cost", "stage_9_comparison")
    builder.add_edge("stage_9_comparison", "stage_10_evidence")
    builder.add_edge("stage_10_evidence", "discovery_analysis")
    builder.add_edge("discovery_analysis", "tradeoff_assessment")
    builder.add_edge("tradeoff_assessment", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile()


class OptimizationDiscoveryAgent:
    """
    Autonomous Optimization Discovery Agent using LangGraph and LangChain.
    Systematically inspects target multi-agent systems via MCP evidence tools,
    discovers optimization candidates across 8 dimensions, and produces reports.
    """

    def __init__(self, key_manager=None):
        self.key_manager = key_manager or get_default_key_manager()
        self.graph = build_optimization_graph()

    def run(
        self,
        workflow_id: str,
        evidence_dir: Optional[str] = None,
        target_model: str = "openai/gpt-oss-120b",
    ) -> OptimizationAgentState:
        """
        Execute the end-to-end optimization discovery pipeline for a target workflow.
        """
        initial_state: OptimizationAgentState = {
            "workflow_id": workflow_id,
            "evidence_dir": evidence_dir,
            "target_model": target_model,
            "stage_status": {},
            "retrieval_errors": [],
            "analysis_limitations": [],
            "agent_run_comparisons": {},
            "execution_evidence": {},
            "optimization_candidates": [],
            "dimension_analyses": {},
            "tradeoff_assessments": [],
            "tier_classifications": {"tier_1": [], "tier_2": [], "tier_3": [], "tier_4": []},
            "final_report": {},
            "final_markdown_report": "",
            "key_usage_log": [],
        }

        logger.info(f"Starting Autonomous Optimization Discovery for workflow '{workflow_id}'...")
        final_state = self.graph.invoke(initial_state)
        logger.info(f"Completed optimization discovery pipeline for workflow '{workflow_id}'.")
        return final_state

