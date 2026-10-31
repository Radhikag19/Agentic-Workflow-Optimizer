"""
Comprehensive Validation Test Suite for TraceProfiler Autonomous Optimization Discovery Agent.

Validates Checkpoints 1 through 6:
- Checkpoint 1: MCP Integration Check (all 10 tools, path traversal, custom dir validation, full prompt, isolated nodes)
- Checkpoint 2: LangGraph Pipeline Stage Sequencing (10 stages in order, Stage 9 & 10 for ALL agents)
- Checkpoint 3: Optimization Discovery Across All 8 Mandatory Dimensions (concrete evidence, dependency verification, model validation flag)
- Checkpoint 4: Feasibility and Trade-Off Assessment (Tiers 1-4, benefits & risks evaluated)
- Checkpoint 5: Groq Key Distribution & Masking (20 keys, distinct allocation per dimension, no leaked secrets)
- Checkpoint 6: End-to-End Execution on Target MAS (contract_review JSON & MD reports with 12 sections)
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from mcp_server import tools as evidence_tools
from optimization_agent.agent import OptimizationDiscoveryAgent, build_optimization_graph
from optimization_agent.discovery import (
    OptimizationDiscoveryEngine,
    verify_dependency_independence,
    _build_graph_reachability,
)
from optimization_agent.key_manager import (
    GroqKeyManager,
    TASK_KEY_MAPPING,
    mask_key,
)
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
from optimization_agent.report import generate_json_report, generate_markdown_report
from optimization_agent.state import OptimizationAgentState
from optimization_agent.tradeoff import TradeoffEvaluator, TIER_1_LABEL, TIER_2_LABEL


# =============================================================================
# Checkpoint 1: MCP Integration Check
# =============================================================================

def test_checkpoint_1_all_10_mcp_tools():
    """Verify all 10 MCP tools return valid data for contract_review."""
    wf_id = "contract_review"

    # 1. list_workflows
    res1 = evidence_tools.list_workflows()
    assert "workflows" in res1
    assert any(w.get("workflow_id") == wf_id for w in res1["workflows"])

    # 2. get_workflow_overview
    res2 = evidence_tools.get_workflow_overview(wf_id)
    assert res2.get("workflow_id") == wf_id
    assert "latency_summary" in res2 or "total_declared_agents" in res2

    # 3. get_workflow_graph with isolated nodes
    res3 = evidence_tools.get_workflow_graph(wf_id, include_isolated_nodes=True)
    assert "nodes" in res3 and "edges" in res3
    assert len(res3["nodes"]) > 0

    # 4. get_agent_profiles
    res4 = evidence_tools.get_agent_profiles(wf_id)
    assert "profiles" in res4
    profiles = res4["profiles"]
    assert len(profiles) >= 10
    agent_ids = [p["agent_id"] for p in profiles]
    sample_agent = agent_ids[0]

    # 5. get_prompt_metrics with full_prompt=True
    res5 = evidence_tools.get_prompt_metrics(wf_id, full_prompt=True)
    assert "prompt_metrics" in res5
    assert res5.get("full_prompt_requested") is True

    # 6. get_latency_metrics
    res6 = evidence_tools.get_latency_metrics(wf_id)
    assert "workflow_latency" in res6 or "agent_latency_metrics" in res6

    # 7. get_token_metrics
    res7 = evidence_tools.get_token_metrics(wf_id)
    assert "agent_token_metrics" in res7

    # 8. get_cost_metrics
    res8 = evidence_tools.get_cost_metrics(wf_id)
    assert "agent_cost_metrics" in res8

    # 9. compare_agent_runs
    res9 = evidence_tools.compare_agent_runs(wf_id, agent_id=sample_agent, limit=5)
    assert "agent_runs" in res9
    assert res9.get("agent_id") == sample_agent

    # 10. get_execution_evidence
    res10 = evidence_tools.get_execution_evidence(wf_id, agent_id=sample_agent)
    assert res10.get("agent_id") == sample_agent
    assert "sample_inputs" in res10 or "input" in res10


def test_checkpoint_1_security_and_edge_cases():
    """Verify path traversal prevention and custom directory validation."""
    # Path traversal protection
    res_traversal = evidence_tools.get_workflow_overview("../../../etc/passwd")
    assert "error" in res_traversal
    assert "traversal" in res_traversal["error"].lower()

    # Invalid custom directory
    res_invalid_dir = evidence_tools.list_workflows(evidence_dir="C:/non_existent_directory_xyz123")
    assert "error" in res_invalid_dir
    assert "does not exist" in res_invalid_dir["error"].lower()

    # Invalid agent reporting
    res_invalid_agent = evidence_tools.get_agent_profiles("contract_review", agent_ids=["phantom_agent_id"])
    assert "non_existent_agents" in res_invalid_agent or "error" in res_invalid_agent


# =============================================================================
# Checkpoint 2: LangGraph Pipeline Stage Sequencing
# =============================================================================

def test_checkpoint_2_pipeline_sequencing_and_coverage():
    """Execute Stages 1-10 sequentially and verify Stages 9 & 10 query EVERY agent."""
    state: OptimizationAgentState = {
        "workflow_id": "contract_review",
        "stage_status": {},
        "retrieval_errors": [],
        "analysis_limitations": [],
    }

    # Step through stages 1 to 10
    state.update(stage_1_discovery(state))
    assert state["stage_status"]["stage_1_discovery"] == "completed"

    state.update(stage_2_overview(state))
    assert state["stage_status"]["stage_2_overview"] == "completed"

    state.update(stage_3_graph(state))
    assert state["stage_status"]["stage_3_graph"] == "completed"

    state.update(stage_4_profiles(state))
    assert state["stage_status"]["stage_4_profiles"] == "completed"
    discovered = state.get("discovered_agents", [])
    assert len(discovered) == 10, f"Expected 10 agents, got {len(discovered)}"

    state.update(stage_5_prompts(state))
    assert state["stage_status"]["stage_5_prompts"] == "completed"

    state.update(stage_6_latency(state))
    assert state["stage_status"]["stage_6_latency"] == "completed"

    state.update(stage_7_tokens(state))
    assert state["stage_status"]["stage_7_tokens"] == "completed"

    state.update(stage_8_cost(state))
    assert state["stage_status"]["stage_8_cost"] == "completed"

    # Stage 9: MUST query EVERY discovered agent
    state.update(stage_9_comparison(state))
    assert state["stage_status"]["stage_9_comparison"] == "completed"
    comparisons = state.get("agent_run_comparisons", {})
    for agent_id in discovered:
        assert agent_id in comparisons, f"Stage 9 missed querying agent {agent_id}"

    # Stage 10: MUST query EVERY discovered agent across sample runs
    state.update(stage_10_evidence(state))
    assert state["stage_status"]["stage_10_evidence"] == "completed"
    evidence = state.get("execution_evidence", {})
    for agent_id in discovered:
        assert agent_id in evidence, f"Stage 10 missed querying evidence for agent {agent_id}"


# =============================================================================
# Checkpoint 3: Optimization Discovery Across All Mandatory Dimensions
# =============================================================================

def test_checkpoint_3_discovery_dimensions_and_constraints():
    """Verify discovery engine covers all 8 dimensions, verifies dependencies, and flags validation."""
    # Setup rich state for contract_review
    state: OptimizationAgentState = {
        "workflow_id": "contract_review",
        "stage_status": {},
        "retrieval_errors": [],
        "analysis_limitations": [],
    }
    for stage_fn in [
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
    ]:
        state.update(stage_fn(state))

    engine = OptimizationDiscoveryEngine()
    result = engine.discover_all_dimensions(state)
    candidates = result["optimization_candidates"]
    analyses = result["dimension_analyses"]

    # Verify all 8 dimensions are analyzed
    required_dims = [
        "agent_pruning",
        "agent_consolidation",
        "workflow_parallelization",
        "model_selection",
        "prompt_optimization",
        "tool_optimization",
        "communication_optimization",
        "additional_optimization",
    ]
    for dim in required_dims:
        assert dim in analyses, f"Missing analysis for dimension {dim}"

    # Verify candidates cite concrete MCP evidence
    assert len(candidates) > 0
    for c in candidates:
        assert "candidate_id" in c
        assert "observed_problem" in c
        assert "supporting_evidence" in c
        assert "mcp_stage_source" in c["supporting_evidence"]
        assert "expected_benefit" in c

    # Verify Parallelization Dependency Independence Constraint
    parallel_cands = [c for c in candidates if c["dimension"] == "workflow_parallelization"]
    assert len(parallel_cands) > 0
    for pc in parallel_cands:
        assert pc.get("dependency_verified") is True
        assert pc["supporting_evidence"].get("is_dependency_independent") is True

    # Verify Model Selection 'requires_validation' Constraint
    model_cands = [c for c in candidates if c["dimension"] == "model_selection"]
    assert len(model_cands) > 0
    for mc in model_cands:
        assert mc.get("requires_validation") is True, f"Model candidate {mc['candidate_id']} missing validation requirement"


def test_checkpoint_3_dependency_independence_helper():
    """Directly test reachability and dependency verification logic."""
    graph = {
        "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "edges": [{"source": "A", "target": "B"}],
    }
    reachability = _build_graph_reachability(graph)
    profiles = {
        "profiles": [
            {"agent_id": "A", "declared_inputs": [], "declared_outputs": ["outA"]},
            {"agent_id": "B", "declared_inputs": ["outA"], "declared_outputs": []},
            {"agent_id": "C", "declared_inputs": [], "declared_outputs": []},
        ]
    }

    # B depends on A -> Not independent
    is_indep_ab, msg_ab = verify_dependency_independence("A", "B", reachability, profiles)
    assert is_indep_ab is False
    assert "dependency detected" in msg_ab or "Data dependency" in msg_ab

    # C is independent of A and B
    is_indep_ac, msg_ac = verify_dependency_independence("A", "C", reachability, profiles)
    assert is_indep_ac is True


# =============================================================================
# Checkpoint 4: Feasibility and Trade-Off Assessment
# =============================================================================

def test_checkpoint_4_tradeoff_assessment_and_tiers():
    """Verify tradeoff evaluator classifies candidates into Tiers 1-4 with benefits and risks."""
    dummy_candidates = [
        {
            "candidate_id": "C1",
            "dimension": "prompt_optimization",
            "title": "Compress Prompt",
            "target_agents": ["a1"],
            "supporting_evidence": {"stage": "stage_5"},
            "expected_benefit": {"token_savings_pct": 30.0},
            "tradeoffs_and_risks": {
                "output_quality_risk": "Low",
                "implementation_complexity": "Low",
            },
            "requires_validation": False,
        },
        {
            "candidate_id": "C2",
            "dimension": "model_selection",
            "title": "Use Smaller Model",
            "target_agents": ["a2"],
            "supporting_evidence": {"stage": "stage_2"},
            "expected_benefit": {"cost_savings_pct": 60.0},
            "tradeoffs_and_risks": {
                "output_quality_risk": "Moderate",
                "implementation_complexity": "Low",
            },
            "requires_validation": True,
        },
        {
            "candidate_id": "C3",
            "dimension": "agent_pruning",
            "title": "Prune Deep Logic",
            "target_agents": ["a3"],
            "supporting_evidence": {"stage": "stage_4"},
            "expected_benefit": {"latency_savings_pct": 50.0},
            "tradeoffs_and_risks": {
                "output_quality_risk": "High",
                "implementation_complexity": "High",
            },
            "requires_validation": False,
        },
        {
            "candidate_id": "C4",
            "dimension": "tool_optimization",
            "title": "Speculative Tool Idea",
            "target_agents": ["a4"],
            "supporting_evidence": {},  # Insufficient
            "expected_benefit": {},
            "tradeoffs_and_risks": {},
            "requires_validation": False,
        },
    ]

    evaluator = TradeoffEvaluator()
    res = evaluator.evaluate_candidates(dummy_candidates)
    assessed = res["tradeoff_assessments"]
    tiers = res["tier_classifications"]

    assert len(assessed) == 4
    assert assessed[0]["feasibility_tier"] == TIER_1_LABEL
    assert assessed[1]["feasibility_tier"] == TIER_2_LABEL
    assert assessed[2]["feasibility_tier"] == "Tier 3: Potential benefit, high trade-off"
    assert assessed[3]["feasibility_tier"] == "Tier 4: Insufficient evidence for recommendation"

    assert len(tiers["tier_1"]) == 1
    assert len(tiers["tier_2"]) == 1
    assert len(tiers["tier_3"]) == 1
    assert len(tiers["tier_4"]) == 1


# =============================================================================
# Checkpoint 5: Groq Key Distribution Verification
# =============================================================================

def test_checkpoint_5_key_distribution_and_masking():
    """Verify distinct key distribution per dimension and strict secret masking."""
    km = GroqKeyManager()
    assert km.total_loaded_keys >= 20

    # 1. Distinct key allocation across dimensions
    keys_used = set()
    dimensions_to_test = [
        "agent_pruning",
        "agent_consolidation",
        "workflow_parallelization",
        "model_selection",
        "prompt_optimization",
        "tool_optimization",
        "communication_optimization",
        "additional_optimization",
        "tradeoff_assessment",
        "report_generation",
    ]

    for dim in dimensions_to_test:
        raw, ev, masked = km.get_key_for_task(dim)
        assert raw, f"Raw key missing for {dim}"
        assert ev in TASK_KEY_MAPPING[dim] or "GROQ_API_KEY_" in ev
        assert "masked" in masked and "gsk_" in masked
        # Verify masking format: gsk_***...PW
        assert mask_key(raw) in masked
        keys_used.add(ev)

    # Must NOT be a simple round-robin or single key
    assert len(keys_used) >= 8, f"Expected distinct key allocations across dimensions, got {len(keys_used)}"

    # 2. Secret Protection: Verify no secret leakage
    sample_text = f"Allocated key: {km.get_masked_identifier('GROQ_API_KEY_1')} successfully."
    assert km.contains_raw_secrets(sample_text) is False

    # Check sanitizer
    raw_k1 = km._keys["GROQ_API_KEY_1"]
    leaked_str = f"Critical error in {raw_k1} during call"
    assert km.contains_raw_secrets(leaked_str) is True
    sanitized = km.sanitize_text(leaked_str)
    assert km.contains_raw_secrets(sanitized) is False
    assert "masked" in sanitized


# =============================================================================
# Checkpoint 6: End-to-End Execution on Target MAS (contract_review)
# =============================================================================

def test_checkpoint_6_end_to_end_contract_review():
    """Execute complete 13-node LangGraph pipeline on contract_review and verify generated reports."""
    agent = OptimizationDiscoveryAgent()
    state = agent.run("contract_review")

    # Verify all 13 stages completed
    stage_status = state.get("stage_status", {})
    required_stages = [
        "stage_1_discovery",
        "stage_2_overview",
        "stage_3_graph",
        "stage_4_profiles",
        "stage_5_prompts",
        "stage_6_latency",
        "stage_7_tokens",
        "stage_8_cost",
        "stage_9_comparison",
        "stage_10_evidence",
        "discovery_analysis",
        "tradeoff_assessment",
        "generate_report",
    ]
    for stage_name in required_stages:
        assert stage_name in stage_status, f"Missing stage {stage_name}"
        assert stage_status[stage_name] == "completed", f"Stage {stage_name} did not complete: {stage_status[stage_name]}"

    # Verify reports generated on disk
    json_path = Path("Target-MAS-Dataset-main/reports/contract_review_optimization_report.json")
    md_path = Path("Target-MAS-Dataset-main/reports/contract_review_optimization_report.md")
    if not json_path.exists():
        json_path = Path("reports/contract_review_optimization_report.json")
        md_path = Path("reports/contract_review_optimization_report.md")

    assert json_path.exists(), f"JSON report file not found at {json_path}"
    assert md_path.exists(), f"Markdown report file not found at {md_path}"

    # Verify all 12 sections in JSON report
    report_data = json.loads(json_path.read_text(encoding="utf-8"))
    required_12_sections = [
        "report_metadata",
        "executive_summary",
        "workflow_baseline",
        "agent_pruning_opportunities",
        "agent_consolidation_opportunities",
        "workflow_parallelization_opportunities",
        "model_selection_recommendations",
        "prompt_optimization_opportunities",
        "tool_optimization_opportunities",
        "communication_optimization_opportunities",
        "additional_optimization_opportunities",
        "prioritized_action_plan",
    ]
    for sec in required_12_sections:
        assert sec in report_data, f"Section '{sec}' missing in JSON report"

    # Verify Markdown report content
    md_content = md_path.read_text(encoding="utf-8")
    assert "contract_review" in md_content
    assert "Executive Summary" in md_content
    assert "Workflow Baseline Metrics" in md_content
    assert "Prioritized Action Plan" in md_content
    assert "Tier 1" in md_content

