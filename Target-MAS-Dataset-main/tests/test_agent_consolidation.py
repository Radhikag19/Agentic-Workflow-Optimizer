"""
Focused Unit Tests for Redesigned Dimension B: Agent Consolidation.

Covers all 18 mandatory scenarios using strictly mocked Groq responses
(zero live API calls).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
import pytest

from mcp_server import tools as evidence_tools
from optimization_agent.consolidation import (
    AgentConsolidationEngine,
    GroqConsolidationClient,
    evaluate_context_budget,
    extract_sequential_chains,
    segment_sequential_chain,
    validate_classification_output,
    validate_combined_prompt_output,
)
from optimization_agent.discovery import OptimizationDiscoveryEngine
from optimization_agent.key_manager import GroqKeyManager
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
from optimization_agent.tradeoff import TIER_2_LABEL, tradeoff_assessment


def _dummy_key_manager() -> GroqKeyManager:
    pool = {f"GROQ_API_KEY_{i}": f"gsk_testkey_{i:02d}_abcdefghijklmnop" for i in range(1, 21)}
    return GroqKeyManager(key_pool=pool)


def _make_profile(aid: str, role: str, ins: List[str], outs: List[str], prompt: str) -> Dict[str, Any]:
    return {
        "agent_id": aid,
        "role": role,
        "goal": f"Execute {role} accurately.",
        "backstory": f"Specialist performing {role}.",
        "declared_responsibilities": f"Handle {role} and emit {', '.join(outs)}.",
        "declared_inputs": ins,
        "declared_outputs": outs,
        "tasks": [{"task_name": f"{aid}_task", "description": prompt}],
    }


def _make_token_entry(mean_in: float, mean_out: float, obs: int = 30) -> Dict[str, Any]:
    return {
        "total_tokens": int((mean_in + mean_out) * obs),
        "mean_tokens": round(mean_in + mean_out, 2),
        "total_prompt_tokens": int(mean_in * obs),
        "mean_prompt_tokens": mean_in,
        "total_completion_tokens": int(mean_out * obs),
        "mean_completion_tokens": mean_out,
        "observation_count": obs,
    }


def _build_synthetic_state(
    agents_spec: List[Dict[str, Any]],
    declared_edges: List[tuple],
    observed_edges: List[tuple] | None = None,
    observed_orders: Dict[str, int] | None = None,
    model_id: str = "openai/gpt-oss-120b",
    model_context_window: int | None = None,
) -> OptimizationAgentState:
    edges = []
    for u, v in declared_edges:
        edges.append({"source": f"agent:{u}", "target": f"agent:{v}", "relationship": "declared_dependency"})
    for u, v in (observed_edges or []):
        edges.append({"source": f"agent:{u}", "target": f"agent:{v}", "relationship": "observed_execution_order"})

    nodes = [{"id": f"agent:{a['agent_id']}", "label": a["agent_id"]} for a in agents_spec]
    profiles = [_make_profile(a["agent_id"], a["role"], a["ins"], a["outs"], a["prompt"]) for a in agents_spec]
    prompts = [
        {
            "agent_id": a["agent_id"],
            "prompt_count": 1,
            "static_prompt_char_length": len(a["prompt"]),
            "static_prompt_word_count": len(a["prompt"].split()),
            "sample_prompt_text": a["prompt"],
            "prompt_text": a["prompt"],
            "is_truncated": False,
        }
        for a in agents_spec
    ]
    token_metrics = {a["agent_id"]: a["tokens"] for a in agents_spec}
    latency_metrics = {a["agent_id"]: {"mean_seconds": a.get("latency", 1.5), "observation_count": 30} for a in agents_spec}
    exec_ev = {
        a["agent_id"]: {
            "observed_execution_orders": [
                observed_orders[a["agent_id"]] if observed_orders and a["agent_id"] in observed_orders else (idx + 1)
            ],
            "sample_inputs": {"input": a["ins"]},
            "sample_outputs": {"output": a["outs"]},
        }
        for idx, a in enumerate(agents_spec)
    }

    state: OptimizationAgentState = {
        "workflow_id": "synthetic_wf",
        "target_model": model_id,
        "workflow_metadata": {
            "workflow_id": "synthetic_wf",
            "workflow_name": "synthetic_wf",
            "model_reconciliation": {"runtime_observed_model": model_id, "declared_model": model_id},
        },
        "workflow_overview": {"total_runs_recorded": 30, "successful_runs": 30, "failed_runs": 0},
        "workflow_graph": {"nodes": nodes, "edges": edges},
        "agent_profiles": {"profiles": profiles},
        "discovered_agents": [a["agent_id"] for a in agents_spec],
        "prompt_metrics": {"prompt_metrics": prompts},
        "latency_metrics": {"agent_latency_metrics": latency_metrics, "workflow_latency": {"mean_seconds": 10.0}},
        "token_metrics": {"agent_token_metrics": token_metrics, "workflow_tokens": {"mean_tokens": 2000}},
        "cost_metrics": {"workflow_cost": {"mean_cost_usd": 0.01}},
        "execution_evidence": exec_ev,
        "agent_run_comparisons": {a["agent_id"]: {"runs_compared_count": 5} for a in agents_spec},
        "stage_status": {},
        "retrieval_errors": [],
        "analysis_limitations": [],
    }
    if model_context_window is not None:
        state["model_context_window"] = model_context_window
    return state


def _mock_groq_factory(
    validator_agents: set[str] | None = None,
    uncertain_agents: set[str] | None = None,
    custom_classifier_fn=None,
    custom_combiner_fn=None,
) -> GroqConsolidationClient:
    validators = validator_agents or set()
    uncertain = uncertain_agents or set()

    def _chat(system_prompt: str, user_prompt: str, model: str) -> str:
        # Stage 3: classification request
        if "operational responsibility classifier" in system_prompt.lower() or "responsibility classifier" in system_prompt.lower():
            payload_str = user_prompt.split("\n", 1)[1]
            payload = json.loads(payload_str)
            aid = payload["agent_id"]
            if custom_classifier_fn:
                return json.dumps(custom_classifier_fn(payload))
            if aid in uncertain:
                return json.dumps({
                    "agent_id": aid,
                    "responsibility_type": "mixed_or_unclear",
                    "is_independent_validator": False,
                    "classification_confidence": 0.45,
                    "evidence": ["Unclear dual scope"],
                    "reasoning": "Mixed validation and transformation hints.",
                    "consolidation_restrictions": ["uncertain_scope"],
                })
            if aid in validators:
                return json.dumps({
                    "agent_id": aid,
                    "responsibility_type": "independent_validation_verification",
                    "is_independent_validator": True,
                    "classification_confidence": 0.94,
                    "evidence": [f"Agent audits upstream payload: {payload.get('declared_responsibilities')}"],
                    "reasoning": "Performs independent verification gatekeeping on upstream outputs.",
                    "consolidation_restrictions": ["never_consolidate_with_execution_agent"],
                })
            return json.dumps({
                "agent_id": aid,
                "responsibility_type": "execution_extraction_transformation_analysis_synthesis",
                "is_independent_validator": False,
                "classification_confidence": 0.92,
                "evidence": [f"Primary task transformation: {payload.get('declared_responsibilities')}"],
                "reasoning": "Executes primary domain extraction/transformation work without independent auditing.",
                "consolidation_restrictions": [],
            })

        # Stage 4: prompt combination request
        payload_str = user_prompt.split("\n", 1)[1]
        req = json.loads(payload_str)
        if custom_combiner_fn:
            return json.dumps(custom_combiner_fn(req))

        agents = req["agents"]
        exec_order = req["execution_order"]
        req_ins = req["required_preserved_inputs"]
        req_outs = req["required_preserved_outputs"]
        steps_txt = "\n".join(
            f"Step {i+1} ({a['agent_id']}): {a['declared_responsibilities']} Instructions: {a['prompt_text']}"
            for i, a in enumerate(agents)
        )
        unified_prompt = (
            f"Execute steps sequentially in strict order ({' -> '.join(exec_order)}).\n"
            f"Inputs: {req_ins}. Required Outputs: {req_outs}.\n{steps_txt}"
        )
        return json.dumps({
            "group_id": req["group_id"],
            "source_agent_ids": exec_order,
            "combined_agent_name": f"unified_{'_'.join(exec_order)}",
            "combined_role": " + ".join(a["role"] or a["agent_id"] for a in agents),
            "combined_goal": "Complete sequential extraction and structuring cleanly in one pass.",
            "unified_prompt": unified_prompt,
            "preserved_responsibilities": [a["declared_responsibilities"] for a in agents],
            "merged_responsibilities": [f"Sequential merge of {', '.join(exec_order)}"],
            "removed_redundancies": ["Removed intermediate JSON serialization/deserialization preamble."],
            "preserved_input_schema": req_ins,
            "preserved_output_schema": req_outs,
            "compatibility_assessment": {
                "is_compatible": True,
                "incompatible_responsibilities": [],
                "notes": "Compatible sequential execution tasks.",
            },
            "combination_rationale": "Adjacent sequential agents share context with non-conflicting responsibilities.",
            "estimated_combined_prompt_tokens": max(1, len(unified_prompt) // 4),
            "requires_validation": True,
        })

    return GroqConsolidationClient(
        key_manager=_dummy_key_manager(),
        model="openai/gpt-oss-120b",
        chat_completion_fn=_chat,
    )


# =============================================================================
# 1 & 2. Dynamic Sequential Chains & Execution-Order Preservation
# =============================================================================

def test_01_dynamic_sequential_chain_identification():
    """Verify DAG extraction ignores parallel branches and handles transitive shortcuts and order-only edges."""
    discovered = ["start", "branch_a", "branch_b", "join", "s1", "s2", "s3", "s4"]
    graph = {
        "nodes": [{"id": f"agent:{a}"} for a in discovered],
        "edges": [
            # Parallel diamond (not strictly sequential)
            {"source": "agent:start", "target": "agent:branch_a", "relationship": "declared_dependency"},
            {"source": "agent:start", "target": "agent:branch_b", "relationship": "declared_dependency"},
            {"source": "agent:branch_a", "target": "agent:join", "relationship": "declared_dependency"},
            {"source": "agent:branch_b", "target": "agent:join", "relationship": "declared_dependency"},
            # Join feeds s1, plus sequential backbone s1 -> s2 -> s3 -> s4 with transitive shortcut s1 -> s3
            {"source": "agent:join", "target": "agent:s1", "relationship": "declared_dependency"},
            {"source": "agent:s1", "target": "agent:s2", "relationship": "declared_dependency"},
            {"source": "agent:s2", "target": "agent:s3", "relationship": "declared_dependency"},
            {"source": "agent:s3", "target": "agent:s4", "relationship": "declared_dependency"},
            {"source": "agent:s1", "target": "agent:s3", "relationship": "declared_dependency"},
            # Consecutive observed execution order between parallel peers MUST NOT create chain
            {"source": "agent:branch_a", "target": "agent:branch_b", "relationship": "observed_execution_order"},
        ],
    }
    res = extract_sequential_chains(graph, discovered)
    assert res["dependency_evidence_status"] == "verified"
    assert res["chains"] == [["join", "s1", "s2", "s3", "s4"]]
    assert ["branch_a", "branch_b"] not in res["chains"]

    # Graph with malformed empty declared targets + only observed_execution_order must NOT fabricate chains
    order_only_graph = {
        "nodes": [{"id": "agent:a1"}, {"id": "agent:a2"}],
        "edges": [
            {"source": "agent:a1", "target": "agent:", "relationship": "declared_dependency"},
            {"source": "agent:a1", "target": "agent:a2", "relationship": "observed_execution_order"},
        ],
    }
    res2 = extract_sequential_chains(order_only_graph, ["a1", "a2"])
    assert res2["chains"] == []
    assert res2["dependency_evidence_status"] == "insufficient_dependency_evidence"


def test_02_execution_order_preservation():
    """Shuffled discovered_agents must still yield execution-ordered chains, groups, and targets."""
    agents_shuffled = [
        {"agent_id": "step_4", "role": "Publisher", "ins": ["clean"], "outs": ["final"], "prompt": "Publish output", "tokens": _make_token_entry(200, 100)},
        {"agent_id": "step_2", "role": "Normalizer", "ins": ["raw"], "outs": ["norm"], "prompt": "Normalize fields", "tokens": _make_token_entry(120, 80)},
        {"agent_id": "step_1", "role": "Extractor", "ins": ["doc"], "outs": ["raw"], "prompt": "Extract fields", "tokens": _make_token_entry(120, 80)},
        {"agent_id": "step_3", "role": "Cleaner", "ins": ["norm"], "outs": ["clean"], "prompt": "Clean fields", "tokens": _make_token_entry(200, 100)},
    ]
    state = _build_synthetic_state(
        agents_shuffled,
        declared_edges=[("step_1", "step_2"), ("step_2", "step_3"), ("step_3", "step_4")],
        observed_edges=[("step_1", "step_2"), ("step_2", "step_3"), ("step_3", "step_4")],
        observed_orders={"step_1": 1, "step_2": 2, "step_3": 3, "step_4": 4},
        model_context_window=20000,
    )
    engine = AgentConsolidationEngine(key_manager=_dummy_key_manager(), groq_client=_mock_groq_factory())
    cands, analysis = engine.analyze(state)
    expected_order = ["step_1", "step_2", "step_3", "step_4"]
    assert analysis["sequential_chains"] == [expected_order]
    assert len(cands) == 1
    assert cands[0]["target_agents"] == ["step_1", "step_2"]
    for c in cands:
        t = c["target_agents"]
        indices = [expected_order.index(x) for x in t]
        assert indices == sorted(indices) and indices == list(range(indices[0], indices[-1] + 1))


# =============================================================================
# 3, 4, 5, 6. 50% Token Segmentation, Recursion, Singletons & Token Separation
# =============================================================================

def test_03_50pct_cumulative_token_segmentation():
    """8 equal-token agents A1..A8 segment cleanly into [A1..A4] and [A5..A8]."""
    chain = [f"A{i}" for i in range(1, 9)]
    token_map = {aid: _make_token_entry(75, 25) for aid in chain}  # 100 each => 12.5%
    seg = segment_sequential_chain(chain, token_map, chain_id="c8")
    assert seg["evidence_status"] == "sufficient"
    assert [g["agent_ids"] for g in seg["groups"]] == [
        ["A1", "A2", "A3", "A4"],
        ["A5", "A6", "A7", "A8"],
    ]
    assert seg["groups"][0]["cumulative_chain_pct_end"] == 50.0
    assert seg["groups"][1]["cumulative_chain_pct_end"] == 100.0


def test_04_recursive_segmentation_oversized_group():
    """Groups still exceeding 50% of the original chain are recursively segmented without empty groups."""
    chain = ["A1", "A2", "A3", "A4"]
    token_map = {
        "A1": _make_token_entry(200, 100),  # 30%
        "A2": _make_token_entry(200, 100),  # 30%
        "A3": _make_token_entry(150, 50),   # 20%
        "A4": _make_token_entry(150, 50),   # 20%
    }
    seg = segment_sequential_chain(chain, token_map)
    groups = [g["agent_ids"] for g in seg["groups"]]
    assert groups == [["A1"], ["A2"], ["A3", "A4"]]
    assert all(len(g) > 0 for g in groups)

    # Also test boundary falling on [A1, A2] = 60% via [25%, 35%, 20%, 20%]
    token_map_2 = {
        "A1": _make_token_entry(150, 100),  # 25%
        "A2": _make_token_entry(250, 100),  # 35%
        "A3": _make_token_entry(100, 100),  # 20%
        "A4": _make_token_entry(100, 100),  # 20%
    }
    seg2 = segment_sequential_chain(chain, token_map_2)
    assert [g["agent_ids"] for g in seg2["groups"]] == [["A1"], ["A2"], ["A3", "A4"]]


def test_05_high_contribution_singleton_isolation():
    """Single agent contributing >50% is isolated as singleton and excluded from merges."""
    specs = [
        {"agent_id": "A1", "role": "Intake", "ins": ["doc"], "outs": ["d1"], "prompt": "p1", "tokens": _make_token_entry(70, 30)},
        {"agent_id": "A2", "role": "HeavyAnalyzer", "ins": ["d1"], "outs": ["d2"], "prompt": "p2", "tokens": _make_token_entry(450, 150)},
        {"agent_id": "A3", "role": "Normalizer", "ins": ["d2"], "outs": ["d3"], "prompt": "p3", "tokens": _make_token_entry(100, 50)},
        {"agent_id": "A4", "role": "Formatter", "ins": ["d3"], "outs": ["d4"], "prompt": "p4", "tokens": _make_token_entry(100, 50)},
    ]
    seg = segment_sequential_chain(["A1", "A2", "A3", "A4"], {s["agent_id"]: s["tokens"] for s in specs})
    groups = seg["groups"]
    assert [g["agent_ids"] for g in groups] == [["A1"], ["A2"], ["A3", "A4"]]
    assert groups[1]["is_isolated_singleton"] is True
    assert groups[1]["segmentation_reason"] == "single_agent_exceeds_50pct_threshold"

    state = _build_synthetic_state(specs, [("A1", "A2"), ("A2", "A3"), ("A3", "A4")], model_context_window=10000)
    engine = AgentConsolidationEngine(key_manager=_dummy_key_manager(), groq_client=_mock_groq_factory())
    cands, _ = engine.analyze(state)
    assert [c["target_agents"] for c in cands] == [["A3", "A4"]]


def test_06_separate_input_output_tokens_and_missing_fallback():
    """Input/output tokens remain separate; mean_tokens is never substituted; missing triggers conservative fallback."""
    chain = ["X1", "X2", "X3"]
    # Only mean_tokens present without separate mean_prompt_tokens / mean_completion_tokens
    bad_tokens = {
        "X1": {"mean_tokens": 500, "observation_count": 30},
        "X2": _make_token_entry(100, 50),
        "X3": _make_token_entry(100, 50),
    }
    seg = segment_sequential_chain(chain, bad_tokens)
    assert seg["evidence_status"] == "insufficient_evidence"
    assert seg["fallback_applied"] is True
    assert [g["agent_ids"] for g in seg["groups"]] == [["X1"], ["X2"], ["X3"]]

    budget = evaluate_context_budget(
        agent_ids=["X1", "X2"],
        model_id="openai/gpt-oss-120b",
        prompt_metrics_map={},
        token_metrics_map=bad_tokens,
        profile_map={},
    )
    assert budget["passed"] is False
    assert budget["evidence_status"] == "insufficient_evidence"


# =============================================================================
# 7 & 8. Context-Window 70% Filter & Double-Counting Prevention
# =============================================================================

def test_07_context_window_rejection_above_70pct_and_no_double_counting():
    """Verify >70% rejection, missing C rejection, and subtraction of static prompt tokens from R."""
    prompt_a = "Extract contract payment terms verbatim. " * 20
    prompt_b = "Normalize payment terms into ISO days. " * 20
    pm_map = {
        "A": {"prompt_text": prompt_a},
        "B": {"prompt_text": prompt_b},
    }
    # Compute static tokens via helper to construct an exact boundary scenario
    from optimization_agent.consolidation.token_budget import count_tokens
    p_a_tok = count_tokens(prompt_a, "openai/gpt-oss-120b")["tokens"]
    p_b_tok = count_tokens(prompt_b, "openai/gpt-oss-120b")["tokens"]
    P = p_a_tok + p_b_tok

    # Dynamic input is 50 each; output is 25 each => R=50, O=50 => total = P + 100
    tm_map = {
        "A": _make_token_entry(p_a_tok + 50, 25),
        "B": _make_token_entry(p_b_tok + 50, 25),
    }

    # Case 1: Set capacity so P + 100 is > 70%
    small_c = int((P + 100) / 0.75)
    over = evaluate_context_budget(
        agent_ids=["A", "B"],
        model_id="openai/gpt-oss-120b",
        prompt_metrics_map=pm_map,
        token_metrics_map=tm_map,
        profile_map={},
        state={"model_context_window": small_c},
    )
    assert over["passed"] is False
    assert over["utilization_pct"] > 70.0
    assert over["runtime_input_tokens_R"] == 50  # P was NOT double-counted inside R

    # Unknown model without state override => insufficient_evidence
    unknown = evaluate_context_budget(
        agent_ids=["A", "B"],
        model_id="unknown/custom-model",
        prompt_metrics_map=pm_map,
        token_metrics_map=tm_map,
        profile_map={},
    )
    assert unknown["passed"] is False
    assert unknown["evidence_status"] == "insufficient_evidence"
    assert unknown["rejection_reason"] == "context_window_unavailable"


def test_08_context_window_acceptance_at_or_below_70pct():
    """Verify acceptance at exactly 70.0% and strictly below 70.0%."""
    # With empty prompts (Wait: P=0, dynamic_in=600, O=100 => total = 700 on C=1000 => exactly 70.0%)
    pm_map = {"A": {"prompt_text": ""}, "B": {"prompt_text": ""}}
    tm_map = {
        "A": _make_token_entry(600, 50),
        "B": _make_token_entry(400, 50),
    }
    at_70 = evaluate_context_budget(
        agent_ids=["A", "B"],
        model_id="openai/gpt-oss-120b",
        prompt_metrics_map=pm_map,
        token_metrics_map=tm_map,
        profile_map={},
        state={"model_context_window": 1000},
    )
    assert at_70["total_projected_tokens"] == 700
    assert at_70["utilization_pct"] == 70.0
    assert at_70["passed"] is True

    below_70 = evaluate_context_budget(
        agent_ids=["A", "B"],
        model_id="openai/gpt-oss-120b",
        prompt_metrics_map=pm_map,
        token_metrics_map={"A": _make_token_entry(599, 50), "B": _make_token_entry(400, 50)},
        profile_map={},
        state={"model_context_window": 1000},
    )
    assert below_70["utilization_pct"] < 70.0
    assert below_70["passed"] is True


# =============================================================================
# 9, 10, 11. Independent Validator Detection & Conservative Exclusions
# =============================================================================

def test_09_independent_validation_agent_identification():
    """Independent validator is identified by semantic role/evidence even with a neutral agent_id."""
    specs = [
        {"agent_id": "stage_one_worker", "role": "Clause Extractor", "ins": ["raw"], "outs": ["clauses"], "prompt": "Extract clauses", "tokens": _make_token_entry(60, 40)},
        {"agent_id": "stage_two_worker", "role": "Compliance Gatekeeper", "ins": ["clauses"], "outs": ["audit_verdict"], "prompt": "Independently audit policy breaches", "tokens": _make_token_entry(60, 40)},
        {"agent_id": "validation_string_normalizer", "role": "Data Formatter", "ins": ["clauses"], "outs": ["norm_clauses"], "prompt": "Format ISO dates", "tokens": _make_token_entry(60, 40)},
        {"agent_id": "stage_four_writer", "role": "Draft Composer", "ins": ["norm_clauses"], "outs": ["draft"], "prompt": "Draft summary", "tokens": _make_token_entry(60, 40)},
    ]
    # Note: 'stage_two_worker' has NO 'validation'/'audit' keyword in ID, yet is validator;
    # 'validation_string_normalizer' HAS 'validation' in ID, yet performs transformation!
    state = _build_synthetic_state(
        specs,
        declared_edges=[("stage_one_worker", "stage_two_worker"), ("stage_two_worker", "validation_string_normalizer"), ("validation_string_normalizer", "stage_four_writer")],
        model_context_window=20000,
    )
    # Give stage_one_worker+stage_two_worker <=50% and validation_string_normalizer+stage_four_writer <=50%
    mock_client = _mock_groq_factory(validator_agents={"stage_two_worker"})
    cands, analysis = AgentConsolidationEngine(_dummy_key_manager(), groq_client=mock_client).analyze(state)

    summary = analysis["summary"]
    assert "stage_two_worker" in summary["independent_validation_agent_ids"]
    assert "validation_string_normalizer" not in summary["independent_validation_agent_ids"]
    assert all("stage_two_worker" not in c["target_agents"] for c in cands)


def test_10_prevent_validation_execution_consolidation():
    """Execution + independent validator must NEVER consolidate; validator stays separately retained."""
    specs = [
        {"agent_id": "extract_a", "role": "Extractor", "ins": ["doc"], "outs": ["fields"], "prompt": "Extract data", "tokens": _make_token_entry(100, 50)},
        {"agent_id": "transform_b", "role": "Transformer", "ins": ["fields"], "outs": ["norm"], "prompt": "Transform data", "tokens": _make_token_entry(100, 50)},
        {"agent_id": "audit_c", "role": "Independent Auditor", "ins": ["norm"], "outs": ["audit_status"], "prompt": "Verify compliance", "tokens": _make_token_entry(150, 50)},
        {"agent_id": "report_d", "role": "Reporter", "ins": ["audit_status"], "outs": ["final"], "prompt": "Compile report", "tokens": _make_token_entry(150, 50)},
    ]
    # Segment split: [extract_a, transform_b] (300 = 37.5%), [audit_c] (200), [report_d] (200) -> Wait:
    # Cumulative: 150 (18.75%), +150=300 (37.5%), +200=500 (>=50%) => Group 1 = [extract_a, transform_b, audit_c]?
    # Wait: 500 > 400 (50% of 800), so Group 1 recursively splits! [150, 150] (37.5%) and [200] (25%).
    # Instead make extract_a + transform_b = 45%, audit_c = 25%, report_d = 30%? No, put all in a 2-agent + 2-agent chain or test 3 agents in group 1:
    specs_group = [
        {"agent_id": "extract_a", "role": "Extractor", "ins": ["doc"], "outs": ["fields"], "prompt": "Extract data", "tokens": _make_token_entry(100, 50)},
        {"agent_id": "audit_c", "role": "Independent Auditor", "ins": ["fields"], "outs": ["audit_status"], "prompt": "Audit independently", "tokens": _make_token_entry(100, 50)},
        {"agent_id": "transform_b", "role": "Transformer", "ins": ["audit_status"], "outs": ["norm"], "prompt": "Normalize data", "tokens": _make_token_entry(250, 150)},
        {"agent_id": "report_d", "role": "Reporter", "ins": ["norm"], "outs": ["final"], "prompt": "Format report", "tokens": _make_token_entry(150, 50)},
    ]
    # Group 1 = [extract_a (15%), audit_c (15%)], Group 2 = [transform_b (40%)], Group 3 = [report_d (20%)] -> wait let's put transform_b + report_d in same group by adjusting tokens:
    specs_group[2]["tokens"] = _make_token_entry(200, 100)  # 300 (30%), + report_d 400? No: total=1000: extract_a=150, audit_c=150, report_d=350 (wait singletons <=50%):
    # [100, 100, 200, 600]? Wait 600 > 50% is singleton. Try [150, 150, 150, 250, 300] (1000): Group1=[150,150,150]=45%, wait cum at index2=450 <500, at index3=700>=500!
    specs_5 = [
        {"agent_id": "e1", "role": "Extractor 1", "ins": ["doc"], "outs": ["o1"], "prompt": "p1", "tokens": _make_token_entry(100, 50)},
        {"agent_id": "e2", "role": "Extractor 2", "ins": ["o1"], "outs": ["o2"], "prompt": "p2", "tokens": _make_token_entry(100, 50)},
        {"agent_id": "v1", "role": "Independent QA", "ins": ["o2"], "outs": ["qa"], "prompt": "p3", "tokens": _make_token_entry(100, 50)},
        {"agent_id": "e3", "role": "Writer", "ins": ["qa"], "outs": ["o3"], "prompt": "p4", "tokens": _make_token_entry(300, 200)},
        {"agent_id": "e4", "role": "Formatter", "ins": ["o3"], "outs": ["final"], "prompt": "p5", "tokens": _make_token_entry(400, 150)},
    ]
    # Cumulative of 150, 150, 150, 500, 550 (total=1500, 50%=750):
    # Cum: 150, 300, 450, 950 >= 750 => Group 1 = [e1, e2, v1, e3] (950 > 750 -> recursive: 475 -> [e1, e2, v1] = 450 <= 750, [e3] = 500 <= 750)
    state = _build_synthetic_state(
        specs_5,
        declared_edges=[("e1", "e2"), ("e2", "v1"), ("v1", "e3"), ("e3", "e4")],
        model_context_window=20000,
    )
    cands, analysis = AgentConsolidationEngine(
        _dummy_key_manager(), groq_client=_mock_groq_factory(validator_agents={"v1"})
    ).analyze(state)

    # Group 1 = [e1, e2, v1] yields subchains [e1,e2], [e2,v1], [e1,e2,v1].
    # Only [e1, e2] survives Stage 3; [e2, v1] and [e1, e2, v1] are eliminated while v1 stays separate!
    assert [c["target_agents"] for c in cands] == [["e1", "e2"]]
    assert analysis["summary"]["independent_validation_agent_ids"] == ["v1"]
    assert analysis["summary"]["eliminated_by_validation_exclusion"] == 2
    assert cands[0]["supporting_evidence"]["validation_exclusion_evidence"][0]["agent_id"] == "v1"


def test_11_conservative_exclusion_uncertain_contradictory_or_error():
    """Uncertain, contradictory, unsupported, mismatched, or errored LLM classifications exclude safely."""
    # Direct post-validator unit checks across all failure modes
    low_conf = validate_classification_output("a1", {
        "agent_id": "a1", "responsibility_type": "execution_extraction_transformation_analysis_synthesis",
        "is_independent_validator": False, "classification_confidence": 0.45,
        "evidence": ["some"], "reasoning": "low conf", "consolidation_restrictions": [],
    })
    contradict = validate_classification_output("a1", {
        "agent_id": "a1", "responsibility_type": "execution_extraction_transformation_analysis_synthesis",
        "is_independent_validator": True, "classification_confidence": 0.95,
        "evidence": ["some"], "reasoning": "contradictory", "consolidation_restrictions": [],
    })
    no_ev = validate_classification_output("a1", {
        "agent_id": "a1", "responsibility_type": "execution_extraction_transformation_analysis_synthesis",
        "is_independent_validator": False, "classification_confidence": 0.95,
        "evidence": [], "reasoning": "no evidence", "consolidation_restrictions": [],
    })
    mismatch = validate_classification_output("a1", {
        "agent_id": "wrong_agent", "responsibility_type": "execution_extraction_transformation_analysis_synthesis",
        "is_independent_validator": False, "classification_confidence": 0.95,
        "evidence": ["some"], "reasoning": "ok", "consolidation_restrictions": [],
    })
    malformed = validate_classification_output("a1", None, error="malformed_json_response")

    for res in (low_conf, contradict, no_ev, mismatch, malformed):
        assert res["excluded_from_consolidation"] is True
        assert res["exclusion_category"] == "conservatively_excluded_uncertain"

    # Also test end-to-end exclusion when an agent returns mixed_or_unclear in Stage 3
    specs = [
        {"agent_id": "u1", "role": "Worker 1", "ins": ["doc"], "outs": ["x"], "prompt": "p1", "tokens": _make_token_entry(120, 80)},
        {"agent_id": "u2", "role": "Worker 2", "ins": ["x"], "outs": ["y"], "prompt": "p2", "tokens": _make_token_entry(120, 80)},
        {"agent_id": "u3", "role": "Worker 3", "ins": ["y"], "outs": ["z"], "prompt": "p3", "tokens": _make_token_entry(200, 100)},
        {"agent_id": "u4", "role": "Worker 4", "ins": ["z"], "outs": ["w"], "prompt": "p4", "tokens": _make_token_entry(200, 100)},
    ]
    state = _build_synthetic_state(specs, [("u1", "u2"), ("u2", "u3"), ("u3", "u4")], model_context_window=20000)
    cands, analysis = AgentConsolidationEngine(
        _dummy_key_manager(), groq_client=_mock_groq_factory(uncertain_agents={"u2"})
    ).analyze(state)
    assert cands == []
    assert "u2" in analysis["summary"]["conservatively_excluded_agent_ids"]


# =============================================================================
# 12. Groq-Only Client, Key Manager & Failover
# =============================================================================

def test_12_groq_only_client_key_manager_and_failover(monkeypatch):
    """Verify Groq-exclusive client uses GROQ_API_KEY_4, handles 429 failover, and masks secrets."""
    km = _dummy_key_manager()
    client = GroqConsolidationClient(key_manager=km, model="openai/gpt-oss-120b")
    assert client.provider_name == "groq"
    assert client.allocated_env_var == "GROQ_API_KEY_4"
    assert "GROQ_API_KEY_4" in client.allocated_key_masked
    assert km.contains_raw_secrets(client.allocated_key_masked) is False

    # Simulate HTTP 429 on first key, succeeding on failover key
    used_keys: List[str] = []

    class _FakeMsg:
        def __init__(self, content: str):
            self.content = content

    class _FakeChoice:
        def __init__(self, content: str):
            self.message = _FakeMsg(content)

    class _FakeResp:
        def __init__(self, content: str):
            self.choices = [_FakeChoice(content)]

    class _FakeCompletions:
        def __init__(self, api_key: str):
            self._api_key = api_key

        def create(self, **kwargs):
            used_keys.append(self._api_key)
            if len(used_keys) == 1:
                raise RuntimeError(f"Error 429 rate limit exceeded using {self._api_key}")
            return _FakeResp('{"ok": true}')

    class _FakeChat:
        def __init__(self, api_key: str):
            self.completions = _FakeCompletions(api_key)

    class _FakeGroq:
        def __init__(self, api_key: str, **kwargs):
            self.chat = _FakeChat(api_key)

    import groq
    monkeypatch.setattr(groq, "Groq", _FakeGroq)

    parsed, err = client.call_json("sys", "usr")
    assert err is None
    assert parsed == {"ok": True}
    assert len(used_keys) == 2
    assert used_keys[0] != used_keys[1]
    assert km.contains_raw_secrets(json.dumps(km.get_usage_log())) is False


# =============================================================================
# 13, 14, 15. Prompt Combination Validation, Post-Gen Budget & Deduplication
# =============================================================================

def test_13_unified_prompt_generation_and_schema_preservation():
    """Verify all 15 Stage 4 keys + rejection of dropped fields, wrong IDs, and incompatible roles."""
    valid_raw = {
        "group_id": "G1",
        "source_agent_ids": ["a1", "a2"],
        "combined_agent_name": "unified_a1_a2",
        "combined_role": "Unified",
        "combined_goal": "Extract and format",
        "unified_prompt": "Step 1 read doc_id. Step 2 emit final_json.",
        "preserved_responsibilities": ["extract", "format"],
        "merged_responsibilities": ["single pass"],
        "removed_redundancies": ["handoff"],
        "preserved_input_schema": ["doc_id"],
        "preserved_output_schema": ["final_json"],
        "compatibility_assessment": {"is_compatible": True, "incompatible_responsibilities": [], "notes": "ok"},
        "combination_rationale": "aligned",
        "estimated_combined_prompt_tokens": 20,
        "requires_validation": False,  # post-validator must enforce True
    }
    ok, val, reason, cat = validate_combined_prompt_output(
        "G1", ["a1", "a2"], valid_raw, required_inputs=["doc_id"], required_outputs=["final_json"]
    )
    assert ok is True and val["requires_validation"] is True

    # Reject dropped output field
    dropped_out = dict(valid_raw, preserved_output_schema=[], unified_prompt="Only extracts stuff.")
    ok2, _, reason2, cat2 = validate_combined_prompt_output(
        "G1", ["a1", "a2"], dropped_out, required_inputs=["doc_id"], required_outputs=["final_json"]
    )
    assert ok2 is False and "dropped_required_output_fields" in reason2

    # Reject unsupported / reordered agent IDs
    wrong_ids = dict(valid_raw, source_agent_ids=["a1", "phantom_agent"])
    ok3, _, reason3, _ = validate_combined_prompt_output("G1", ["a1", "a2"], wrong_ids, [], [])
    assert ok3 is False and "unsupported_or_reordered_agent_ids" in reason3

    # Reject incompatible responsibilities
    incompat = dict(valid_raw, compatibility_assessment={
        "is_compatible": False, "incompatible_responsibilities": ["adversarial audit conflict"], "notes": "conflict"
    })
    ok4, _, _, cat4 = validate_combined_prompt_output("G1", ["a1", "a2"], incompat, [], [])
    assert ok4 is False and cat4 == "incompatible_responsibilities"

    # Reject malformed JSON error
    ok5, _, _, cat5 = validate_combined_prompt_output("G1", ["a1", "a2"], None, [], [], error="malformed_json_response")
    assert ok5 is False and cat5 == "llm_or_schema_error"


def test_14_post_generation_context_window_validation():
    """Candidate passing Stage 1 pre-filter must be rejected if Groq emits an oversized unified prompt."""
    specs = [
        {"agent_id": "s1", "role": "Extractor", "ins": ["doc"], "outs": ["mid"], "prompt": "Short step 1", "tokens": _make_token_entry(120, 80)},
        {"agent_id": "s2", "role": "Normalizer", "ins": ["mid"], "outs": ["mid2"], "prompt": "Short step 2", "tokens": _make_token_entry(120, 80)},
        {"agent_id": "s3", "role": "Classifier", "ins": ["mid2"], "outs": ["mid3"], "prompt": "Short step 3", "tokens": _make_token_entry(200, 100)},
        {"agent_id": "s4", "role": "Publisher", "ins": ["mid3"], "outs": ["out"], "prompt": "Short step 4", "tokens": _make_token_entry(200, 100)},
    ]
    # [s1, s2] = 40% of chain, pre-filter tokens P+R+O well below 70% of C=2000 (1400)
    state = _build_synthetic_state(
        specs,
        [("s1", "s2"), ("s2", "s3"), ("s3", "s4")],
        model_context_window=2000,
    )

    def bloated_combiner(req: Dict[str, Any]) -> Dict[str, Any]:
        bloated_text = ("Preserve doc and mid2. " + ("verbose instruction token bloat ") * 260)
        return {
            "group_id": req["group_id"],
            "source_agent_ids": req["execution_order"],
            "combined_agent_name": "bloated_unified",
            "combined_role": "Bloated",
            "combined_goal": "Over-explain everything",
            "unified_prompt": bloated_text,
            "preserved_responsibilities": ["extract", "normalize"],
            "merged_responsibilities": ["both"],
            "removed_redundancies": [],
            "preserved_input_schema": ["doc"],
            "preserved_output_schema": ["mid2"],
            "compatibility_assessment": {"is_compatible": True, "incompatible_responsibilities": [], "notes": "ok"},
            "combination_rationale": "bloated test",
            "estimated_combined_prompt_tokens": 1300,
            "requires_validation": True,
        }

    cands, analysis = AgentConsolidationEngine(
        _dummy_key_manager(), groq_client=_mock_groq_factory(custom_combiner_fn=bloated_combiner)
    ).analyze(state)

    assert cands == []
    assert analysis["summary"]["prompt_combinations_generated"] == 1
    assert analysis["summary"]["rejected_after_final_prompt_tokenization"] == 1
    assert any(e["stage"] == "stage_4_rejected_post_tokenization" for e in analysis["eliminations_audit"])


def test_15_duplicate_candidate_elimination():
    """Duplicate declared edges / identical contiguous subchains must not produce duplicate candidates."""
    specs = [
        {"agent_id": "d1", "role": "Step 1", "ins": ["doc"], "outs": ["mid"], "prompt": "Do step 1", "tokens": _make_token_entry(120, 80)},
        {"agent_id": "d2", "role": "Step 2", "ins": ["mid"], "outs": ["mid2"], "prompt": "Do step 2", "tokens": _make_token_entry(120, 80)},
        {"agent_id": "d3", "role": "Step 3", "ins": ["mid2"], "outs": ["mid3"], "prompt": "Do step 3", "tokens": _make_token_entry(200, 100)},
        {"agent_id": "d4", "role": "Step 4", "ins": ["mid3"], "outs": ["res"], "prompt": "Do step 4", "tokens": _make_token_entry(200, 100)},
    ]
    # Duplicate declared edge ("d1", "d2") in graph
    state = _build_synthetic_state(
        specs,
        [("d1", "d2"), ("d1", "d2"), ("d2", "d3"), ("d3", "d4")],
        model_context_window=20000,
    )
    cands, _ = AgentConsolidationEngine(
        _dummy_key_manager(), groq_client=_mock_groq_factory()
    ).analyze(state)
    assert len(cands) == 1
    assert cands[0]["target_agents"] == ["d1", "d2"]
    assert len({tuple(c["target_agents"]) for c in cands}) == 1


# =============================================================================
# 16, 17, 18. Schema, Tier/Report Consistency & Full Dimension Compatibility
# =============================================================================

def test_16_candidate_schema_and_requires_validation_true():
    """Verify every required top-level field, all supporting_evidence fields, and requires_validation=True."""
    specs = [
        {"agent_id": "p1", "role": "Parser", "ins": ["raw_text"], "outs": ["parsed"], "prompt": "Parse raw_text", "tokens": _make_token_entry(120, 80)},
        {"agent_id": "p2", "role": "Mapper", "ins": ["parsed"], "outs": ["mapped"], "prompt": "Map parsed items", "tokens": _make_token_entry(120, 80)},
        {"agent_id": "p3", "role": "Ranker", "ins": ["mapped"], "outs": ["ranked"], "prompt": "Rank mapped items", "tokens": _make_token_entry(200, 100)},
        {"agent_id": "p4", "role": "Writer", "ins": ["ranked"], "outs": ["report"], "prompt": "Write report", "tokens": _make_token_entry(200, 100)},
    ]
    state = _build_synthetic_state(
        specs,
        [("p1", "p2"), ("p2", "p3"), ("p3", "p4")],
        model_context_window=20000,
    )
    cands, _ = AgentConsolidationEngine(_dummy_key_manager(), groq_client=_mock_groq_factory()).analyze(state)
    assert len(cands) == 1
    c = cands[0]

    top_keys = {
        "candidate_id", "dimension", "title", "target_agents", "observed_problem",
        "supporting_evidence", "proposed_solution", "expected_benefit",
        "tradeoffs_and_risks", "tier", "requires_validation", "allocated_key", "unified_prompt",
    }
    assert top_keys.issubset(c.keys())
    assert c["dimension"] == "agent_consolidation"
    assert c["requires_validation"] is True
    assert c["expected_benefit"]["is_estimate"] is True
    assert "GROQ_API_KEY_4" in c["allocated_key"]

    ev = c["supporting_evidence"]
    required_ev_keys = {
        "original_sequential_chain", "candidate_group", "agent_execution_order",
        "per_agent_input_tokens", "per_agent_output_tokens", "per_agent_contribution_index",
        "group_contribution", "segmentation_boundary_and_reason",
        "independent_validation_classification", "validation_exclusion_evidence",
        "original_prompt_token_count", "unified_prompt_token_count",
        "context_window_capacity", "context_utilization_percentage", "context_budget_pass_fail",
        "groq_prompt_combination_result", "original_agent_responsibilities",
        "consolidated_responsibilities", "evidence_provenance",
    }
    assert required_ev_keys.issubset(ev.keys())


def test_17_stage12_tier_consistency_and_stage13_reports():
    """Verify Stage 12 Tier 2 consistency and all 10 Stage 13 audit summary metrics in JSON & Markdown."""
    # [c1(15%), c2(15%), c3(20%), c4(50%)] => Group 1 = [c1, c2, c3] (50%), Group 2 = [c4] (50%)
    # c3 is independent validator => [c1, c2] consolidates while c3 is kept separate!
    specs = [
        {"agent_id": "c1", "role": "Intake", "ins": ["doc"], "outs": ["clean"], "prompt": "Ingest doc", "tokens": _make_token_entry(100, 50)},
        {"agent_id": "c2", "role": "Extractor", "ins": ["clean"], "outs": ["items"], "prompt": "Extract items", "tokens": _make_token_entry(100, 50)},
        {"agent_id": "c3", "role": "Independent Reviewer", "ins": ["items"], "outs": ["approved"], "prompt": "Audit compliance", "tokens": _make_token_entry(130, 70)},
        {"agent_id": "c4", "role": "Summarizer", "ins": ["approved"], "outs": ["summary"], "prompt": "Summarize items", "tokens": _make_token_entry(350, 150)},
    ]
    state = _build_synthetic_state(
        specs,
        [("c1", "c2"), ("c2", "c3"), ("c3", "c4")],
        model_context_window=20000,
    )
    mock_client = _mock_groq_factory(validator_agents={"c3"})

    disc_res = OptimizationDiscoveryEngine(
        key_manager=_dummy_key_manager(), consolidation_client=mock_client
    ).discover_all_dimensions(state)
    state.update(disc_res)

    trade_res = tradeoff_assessment(state)
    state.update(trade_res)

    cons_cands = [c for c in state["optimization_candidates"] if c["dimension"] == "agent_consolidation"]
    assert len(cons_cands) == 1
    assert cons_cands[0]["tier"] == "Tier 2"
    assert cons_cands[0]["feasibility_tier"] == TIER_2_LABEL
    assert cons_cands[0]["requires_validation"] is True

    rep_json = generate_json_report(state, "GROQ_API_KEY_13 (masked: gsk_***...test)")
    cons_section = rep_json["agent_consolidation_opportunities"]
    assert isinstance(cons_section, dict)

    required_10_metrics = {
        "original_agent_count",
        "initial_theoretical_search_space_size",
        "sequential_chains_identified",
        "eliminated_by_context_window_filtering",
        "eliminated_by_50pct_segmentation_rule",
        "independent_validation_agents_identified",
        "rejected_for_incompatible_responsibilities",
        "prompt_combinations_generated",
        "rejected_after_final_prompt_tokenization",
        "final_consolidation_opportunities",
    }
    assert required_10_metrics.issubset(cons_section["summary"].keys())
    assert "search_space_reduction_summary" in cons_section["summary"]
    assert cons_section["candidates"][0]["tier"] == "Tier 2"
    assert cons_section["candidates"][0]["unified_prompt"]

    # Confirm NOT listed in Tier 1 action plan or Tier 1 roadmap
    t1_ids = [x["candidate_id"] for x in rep_json["prioritized_action_plan"]["tier_1_immediate_implementation"]]
    t2_ids = [x["candidate_id"] for x in rep_json["prioritized_action_plan"]["tier_2_implementation_with_validation"]]
    assert cons_cands[0]["candidate_id"] not in t1_ids
    assert cons_cands[0]["candidate_id"] in t2_ids

    md = generate_markdown_report(rep_json)
    assert "#### [Tier 2]" in md
    assert "Generated Unified Prompt" in md
    assert "Search-Space Reduction:" in md


def test_18_compatibility_with_all_other_dimensions():
    """Run all 8 dimensions together on real MCP evidence for customer_support with mocked Groq."""
    state: OptimizationAgentState = {
        "workflow_id": "customer_support",
        "stage_status": {},
        "retrieval_errors": [],
        "analysis_limitations": [],
    }
    for fn in [
        stage_1_discovery, stage_2_overview, stage_3_graph, stage_4_profiles,
        stage_5_prompts, stage_6_latency, stage_7_tokens, stage_8_cost,
        stage_9_comparison, stage_10_evidence,
    ]:
        state.update(fn(state))

    # customer_support graph has sequential chain: issue_investigation_agent -> resolution_agent -> response_drafting_agent -> resolution_verification_agent
    mock_client = _mock_groq_factory(validator_agents={"resolution_verification_agent"})
    engine = OptimizationDiscoveryEngine(key_manager=_dummy_key_manager(), consolidation_client=mock_client)
    res = engine.discover_all_dimensions(state)
    state.update(res)
    state.update(tradeoff_assessment(state))
    rep_json = generate_json_report(state, "GROQ_API_KEY_13 (masked: gsk_***...test)")

    for dim in [
        "agent_pruning", "agent_consolidation", "workflow_parallelization",
        "model_selection", "prompt_optimization", "tool_optimization",
        "communication_optimization", "additional_optimization",
    ]:
        assert dim in state["dimension_analyses"]

    assert isinstance(rep_json["agent_consolidation_opportunities"], dict)
    assert rep_json["agent_consolidation_opportunities"]["summary"]["original_agent_count"] == 6
    assert rep_json["agent_consolidation_opportunities"]["summary"]["sequential_chains_identified"] >= 1
    assert "agent_pruning_opportunities" in rep_json
    assert isinstance(rep_json["prompt_optimization_opportunities"], dict)
    assert isinstance(rep_json["tool_optimization_opportunities"], list)
