"""
State Definition for TraceProfiler Autonomous Optimization Discovery Agent.

Defines the typed dictionary state passed between LangGraph pipeline nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class OptimizationCandidate(TypedDict, total=False):
    """Schema for a discovered optimization candidate."""
    candidate_id: str
    dimension: str  # e.g., 'agent_pruning', 'workflow_parallelization', etc.
    title: str
    target_agents: List[str]
    observed_problem: str
    supporting_evidence: Dict[str, Any]
    proposed_solution: str
    expected_benefit: Dict[str, Any]  # latency_savings, token_savings, cost_savings, etc.
    tradeoffs_and_risks: Dict[str, Any]
    tier: str  # 'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'
    requires_validation: bool
    dependency_verified: Optional[bool]


class OptimizationAgentState(TypedDict, total=False):
    """
    Complete state object tracked and transformed throughout the 13-stage
    LangGraph execution pipeline.
    """
    # Inputs
    workflow_id: str
    evidence_dir: Optional[str]
    target_model: str

    # Execution tracking
    stage_status: Dict[str, str]
    retrieval_errors: List[Dict[str, Any]]
    analysis_limitations: List[str]

    # MCP Evidence Data (Stages 1 - 10)
    available_workflows: List[Dict[str, Any]]
    workflow_metadata: Dict[str, Any]
    workflow_overview: Dict[str, Any]
    workflow_graph: Dict[str, Any]
    agent_profiles: Dict[str, Any]
    discovered_agents: List[str]
    prompt_metrics: Dict[str, Any]
    latency_metrics: Dict[str, Any]
    token_metrics: Dict[str, Any]
    cost_metrics: Dict[str, Any]
    agent_run_comparisons: Dict[str, Any]  # agent_id -> compare_agent_runs dict
    execution_evidence: Dict[str, Any]     # agent_id -> get_execution_evidence dict
    sample_run_ids: List[str]

    # Analysis & Synthesis (Stages 11 - 13)
    dimension_analyses: Dict[str, Any]     # dimension_key -> raw analysis output
    optimization_candidates: List[Dict[str, Any]]
    tradeoff_assessments: List[Dict[str, Any]]
    tier_classifications: Dict[str, List[Dict[str, Any]]]

    # Generated Reports
    final_report: Dict[str, Any]
    final_markdown_report: str
    key_usage_log: List[Dict[str, Any]]

