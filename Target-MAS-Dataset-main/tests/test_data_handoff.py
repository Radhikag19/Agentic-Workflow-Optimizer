"""
Focused Unit Tests for Dimension G: Communication & Data Handoff Optimization.

Covers all 23 mandatory scenarios using mocked Groq responses (zero live API calls).
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List
import pytest

from optimization_agent.consolidation.groq_client import GroqConsolidationClient
from optimization_agent.consolidation.token_budget import count_tokens
from optimization_agent.data_handoff import (
    DataHandoffOptimizationEngine,
    analyze_payloads,
    extract_verified_handoffs,
    map_consumer_requirements,
    validate_groq_semantic_output,
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


def _dummy_km() -> GroqKeyManager:
    pool = {f"GROQ_API_KEY_{i}": f"gsk_testhandoff_{i:02d}_abcdefghijklmnop" for i in range(1, 21)}
    return GroqKeyManager(key_pool=pool)


def _mock_handoff_groq(
    response_fn=None,
    call_counter: List[int] | None = None,
) -> GroqConsolidationClient:
    counter = call_counter if call_counter is not None else []

    def _chat(system_prompt: str, user_prompt: str, model: str) -> str:
        counter.append(1)
        req = json.loads(user_prompt.split("\n", 1)[1])
        if response_fn:
            out = response_fn(req)
            return out if isinstance(out, str) else json.dumps(out)

        prod = req["producer_agent"]
        cons = req["consumer_agent"]
        py_cls = req["python_field_classification"]
        uncertain = list(req["suspected_redundant_duplicate_or_uncertain_fields"]["uncertain"])
        required = [k for k, v in py_cls.items() if v["classification"] in ("required", "used")]
        return json.dumps({
            "producer_agent": prod,
            "consumer_agent": cons,
            "field_assessments": {k: "classified" for k in req["original_handoff_fields"]},
            "redundant_fields": uncertain,
            "duplicate_fields": [],
            "required_fields": required,
            "uncertain_fields": [],
            "proposed_compact_payload_schema": {"retained": required},
            "semantic_rationale": "Verified uncertain fields are unreferenced upstream baggage.",
            "confidence": 0.91,
            "requires_human_or_empirical_validation": True,
        })

    return GroqConsolidationClient(
        key_manager=_dummy_km(),
        model="openai/gpt-oss-120b",
        chat_completion_fn=_chat,
        task_type="communication_data_handoff",
    )


def _make_handoff_state(
    agents: List[Dict[str, Any]],
    declared_edges: List[tuple],
    observed_edges: List[tuple] | None = None,
) -> OptimizationAgentState:
    edges = []
    for u, v in declared_edges:
        edges.append({"source": f"agent:{u}", "target": f"agent:{v}", "relationship": "declared_dependency"})
    for u, v in (observed_edges or []):
        edges.append({"source": f"agent:{u}", "target": f"agent:{v}", "relationship": "observed_execution_order"})

    return {
        "workflow_id": "synthetic_handoff_wf",
        "target_model": "openai/gpt-oss-120b",
        "workflow_metadata": {
            "workflow_id": "synthetic_handoff_wf",
            "workflow_name": "synthetic_handoff_wf",
            "model_reconciliation": {"runtime_observed_model": "openai/gpt-oss-120b"},
        },
        "workflow_overview": {"total_runs_recorded": 30, "successful_runs": 30, "failed_runs": 0},
        "discovered_agents": [a["agent_id"] for a in agents],
        "workflow_graph": {
            "nodes": [{"id": f"agent:{a['agent_id']}", "label": a["agent_id"]} for a in agents],
            "edges": edges,
        },
        "agent_profiles": {
            "profiles": [
                {
                    "agent_id": a["agent_id"],
                    "role": a.get("role", "Specialist"),
                    "goal": a.get("goal", ""),
                    "declared_responsibilities": a.get("responsibilities", ""),
                    "declared_inputs": a.get("declared_inputs", []),
                    "declared_outputs": a.get("declared_outputs", []),
                    "tasks": a.get("tasks", []),
                }
                for a in agents
            ]
        },
        "prompt_metrics": {
            "prompt_metrics": [
                {
                    "agent_id": a["agent_id"],
                    "prompt_text": a.get("prompt_text", ""),
                    "sample_prompt_text": a.get("prompt_text", ""),
                    "is_truncated": False,
                }
                for a in agents
            ]
        },
        "token_metrics": {
            "agent_token_metrics": {
                a["agent_id"]: {"mean_prompt_tokens": 500, "mean_completion_tokens": 200, "observation_count": a.get("obs", 30)}
                for a in agents
            }
        },
        "latency_metrics": {"workflow_latency": {"mean_seconds": 12.0}, "agent_latency_metrics": {}},
        "cost_metrics": {"workflow_cost": {"mean_cost_usd": 0.01}},
        "execution_evidence": {
            a["agent_id"]: {
                "input": a.get("runtime_input"),
                "output": a.get("runtime_output"),
            }
            for a in agents
        },
        "agent_run_comparisons": {a["agent_id"]: {"runs_compared_count": a.get("obs", 30)} for a in agents},
        "stage_status": {},
        "retrieval_errors": [],
        "analysis_limitations": [],
    }


def _standard_two_agent_agents(extra_payload: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    full_doc = "Master Services Agreement legal clause body text with full obligations and terms. " * 25
    out_payload = {
        "full_document": full_doc,
        "contract_id": "C101",
        "diff_clauses": ["C12", "C18"],
        "flagged_items": ["liability", "termination"],
        "risk_evidence": ["Clause C12 uncapped indemnity"],
        "source_metadata": {"ingestion_ts": "2026-11-20T00:00:00Z", "sha256": "abc123def456"},
    }
    if extra_payload:
        out_payload.update(extra_payload)

    return [
        {
            "agent_id": "document_analyzer",
            "role": "Document Analyzer",
            "declared_outputs": list(out_payload.keys()),
            "runtime_output": out_payload,
        },
        {
            "agent_id": "risk_analyzer",
            "role": "Risk Evaluator",
            "goal": "Evaluate legal risk from contract_id and diff_clauses.",
            "responsibilities": "Score deviations on flagged_items using risk_evidence.",
            "declared_inputs": ["contract_id", "diff_clauses"],
            "declared_outputs": ["risk_score", "flagged_items"],
            "prompt_text": "Inspect {contract_id} and {diff_clauses}. Incorporate flagged_items and risk_evidence into the score.",
        },
    ]


# =============================================================================
# 1, 2, 3. Verified Handoff Extraction & Evidence Distinction
# =============================================================================

def test_01_verified_producer_consumer_extraction():
    """Verify extraction uses declared_dependency only and ignores observed_execution_order-only pairs."""
    agents = _standard_two_agent_agents() + [
        {"agent_id": "peer_agent", "role": "Independent Peer", "declared_inputs": ["x"]}
    ]
    state = _make_handoff_state(
        agents,
        declared_edges=[("document_analyzer", "risk_analyzer")],
        observed_edges=[("risk_analyzer", "peer_agent")],
    )
    res = extract_verified_handoffs(state)
    assert res["dependency_evidence_status"] == "verified"
    assert [(h["producer_agent"], h["consumer_agent"]) for h in res["handoffs"]] == [
        ("document_analyzer", "risk_analyzer")
    ]

    # Order-only graph without declared dependencies must not infer handoffs
    order_only = _make_handoff_state(agents, declared_edges=[], observed_edges=[("document_analyzer", "risk_analyzer")])
    res2 = extract_verified_handoffs(order_only)
    assert res2["handoffs"] == []
    assert res2["dependency_evidence_status"] == "insufficient_dependency_evidence"


def test_02_runtime_observed_vs_statically_inferred_handoffs():
    """Verify runtime_observed vs statically_inferred labeling without fabricating payload bytes/tokens."""
    agents_runtime = _standard_two_agent_agents()
    ext_rt = extract_verified_handoffs(_make_handoff_state(agents_runtime, [("document_analyzer", "risk_analyzer")]))
    h_rt = ext_rt["handoffs"][0]
    assert h_rt["evidence_quality"] == "runtime_observed"
    assert h_rt["is_statically_inferred"] is False
    assert h_rt["payload_tokens"] > 0 and h_rt["payload_size_bytes"] > 0

    agents_static = [
        {"agent_id": "p", "declared_outputs": ["full_document", "contract_id"], "runtime_output": None},
        {"agent_id": "c", "declared_inputs": ["contract_id"], "prompt_text": "Use {contract_id}"},
    ]
    ext_st = extract_verified_handoffs(_make_handoff_state(agents_static, [("p", "c")]))
    h_st = ext_st["handoffs"][0]
    assert h_st["evidence_quality"] == "statically_inferred"
    assert h_st["is_statically_inferred"] is True
    assert h_st["payload_tokens"] is None and h_st["payload_size_bytes"] is None


def test_03_transmitted_payload_field_extraction():
    """Verify transmitted_fields reflects actual runtime keys including undeclared extra fields."""
    agents = _standard_two_agent_agents(extra_payload={"undeclared_debug_trace": "trace data " * 5})
    agents[0]["declared_outputs"] = ["contract_id", "diff_clauses"]
    ext = extract_verified_handoffs(_make_handoff_state(agents, [("document_analyzer", "risk_analyzer")]))
    h = ext["handoffs"][0]
    assert "undeclared_debug_trace" in h["transmitted_fields"]
    assert "undeclared_debug_trace" not in h["producer_declared_outputs"]


# =============================================================================
# 4, 5, 6. Payload Size, Repetition & Duplicate Value Analysis
# =============================================================================

def test_04_oversized_payload_detection():
    """Detect oversized full_document text and large JSON objects."""
    agents = _standard_two_agent_agents(extra_payload={"huge_nested": {"items": [f"entry_{i}" for i in range(120)]}})
    ext = extract_verified_handoffs(_make_handoff_state(agents, [("document_analyzer", "risk_analyzer")]))
    pa = analyze_payloads(ext["handoffs"])[0]["payload_analysis"]
    assert pa["is_large_payload"] is True
    assert "full_document" in pa["large_text_or_document_fields"]
    assert "huge_nested" in pa["large_json_objects"]


def test_05_repeated_fields_across_handoffs_detection():
    """Detect fields transmitted repeatedly across >= 2 handoffs."""
    shared_doc = "Contract body text repeated everywhere. " * 20
    agents = [
        {"agent_id": "a1", "declared_outputs": ["full_document", "contract_id"], "runtime_output": {"full_document": shared_doc, "contract_id": "C101"}},
        {"agent_id": "a2", "declared_inputs": ["contract_id"], "declared_outputs": ["full_document", "contract_id", "clauses"], "runtime_output": {"full_document": shared_doc, "contract_id": "C101", "clauses": ["C1"]}},
        {"agent_id": "a3", "declared_inputs": ["contract_id", "clauses"], "declared_outputs": ["summary"]},
    ]
    ext = extract_verified_handoffs(_make_handoff_state(agents, [("a1", "a2"), ("a2", "a3")]))
    enriched = analyze_payloads(ext["handoffs"])
    pa0 = enriched[0]["payload_analysis"]
    assert "full_document" in pa0["repeated_across_handoffs"]
    assert "full_document" in pa0["repeated_static_fields"]


def test_06_duplicate_or_overlapping_values_detection():
    """Detect identical values transmitted under two distinct field names."""
    agents = _standard_two_agent_agents(extra_payload={"risk_evidence_copy": ["Clause C12 uncapped indemnity"]})
    ext = extract_verified_handoffs(_make_handoff_state(agents, [("document_analyzer", "risk_analyzer")]))
    pa = analyze_payloads(ext["handoffs"])[0]["payload_analysis"]
    dup_pairs = pa["duplicate_value_pairs"]
    assert any(p["canonical_field"] == "risk_evidence" and p["duplicate_field"] == "risk_evidence_copy" for p in dup_pairs)


# =============================================================================
# 7, 8, 9, 10. Consumer Mapping, Preservation & Conservative Uncertainty
# =============================================================================

def test_07_producer_output_and_consumer_input_schema_comparison():
    """Compare producer outputs against consumer declared_inputs cleanly."""
    state = _make_handoff_state(_standard_two_agent_agents(), [("document_analyzer", "risk_analyzer")])
    cands, ana = DataHandoffOptimizationEngine(_dummy_km(), groq_client=_mock_handoff_groq()).analyze(state)
    assert len(cands) == 1
    ev = cands[0]["supporting_evidence"]
    assert set(ev["consumer_required_fields"]) >= {"contract_id", "diff_clauses"}
    assert "full_document" in ev["fields_excluded"]
    assert "source_metadata" in ev["fields_excluded"]


def test_08_preservation_of_explicitly_required_consumer_fields():
    """Even a very large field explicitly in declared_inputs MUST be preserved, never removed."""
    agents = _standard_two_agent_agents()
    agents[1]["declared_inputs"] = ["contract_id", "diff_clauses", "full_document"]
    state = _make_handoff_state(agents, [("document_analyzer", "risk_analyzer")])
    cands, ana = DataHandoffOptimizationEngine(_dummy_km(), groq_client=_mock_handoff_groq()).analyze(state)
    assert len(cands) == 1
    assert "full_document" in cands[0]["supporting_evidence"]["fields_retained"]
    assert "full_document" not in cands[0]["supporting_evidence"]["fields_excluded"]


def test_09_conditional_field_preservation():
    """Fields mentioned in conditional/fallback/exception instructions must be preserved as required."""
    agents = _standard_two_agent_agents(extra_payload={"fallback_clauses": ["Section 9 fallback"]})
    agents[1]["prompt_text"] += "\nIf diff_clauses is empty, use fallback_clauses as exception fallback."
    state = _make_handoff_state(agents, [("document_analyzer", "risk_analyzer")])
    cands, _ = DataHandoffOptimizationEngine(_dummy_km(), groq_client=_mock_handoff_groq()).analyze(state)
    ev = cands[0]["supporting_evidence"]
    assert "fallback_clauses" in ev["conditional_fields_preserved"]
    assert "fallback_clauses" in ev["fields_retained"]


def test_10_conservative_treatment_of_uncertain_fields():
    """Missing consumer schema & prompt makes all fields uncertain and retains all without pruning."""
    agents = [
        {"agent_id": "p", "declared_outputs": ["full_document", "contract_id"], "runtime_output": {"full_document": "Long doc " * 40, "contract_id": "C1"}},
        {"agent_id": "c", "declared_inputs": [], "prompt_text": "", "goal": "", "responsibilities": ""},
    ]
    state = _make_handoff_state(agents, [("p", "c")])
    calls: List[int] = []
    cands, ana = DataHandoffOptimizationEngine(_dummy_km(), groq_client=_mock_handoff_groq(call_counter=calls)).analyze(state)
    assert calls == []
    assert cands == []
    assert set(ana["handoffs_analyzed"][0]["uncertain_fields"]) == {"full_document", "contract_id"}


# =============================================================================
# 11, 12, 13, 14. Optional Groq Semantic Analysis & Guardrails
# =============================================================================

def test_11_groq_called_only_for_semantic_ambiguity_and_uses_key9():
    """When uncertain ambiguous fields exist alongside clear consumer role, Groq is called once using GROQ_API_KEY_9."""
    km = _dummy_km()
    real_cfg = GroqConsolidationClient(key_manager=km, model="openai/gpt-oss-120b", task_type="communication_data_handoff")
    assert real_cfg.allocated_env_var == "GROQ_API_KEY_9"
    assert "GROQ_API_KEY_9" in real_cfg.allocated_key_masked
    assert km.contains_raw_secrets(real_cfg.allocated_key_masked) is False

    agents = _standard_two_agent_agents(extra_payload={"legal_background_summary": "Long background overview " * 15})
    # Consumer mentions "legal" and "summary" in goal without explicitly naming field `legal_background_summary`
    agents[1]["goal"] = "Provide concise summary of legal risk deviations."
    state = _make_handoff_state(agents, [("document_analyzer", "risk_analyzer")])

    calls: List[int] = []
    cands, ana = DataHandoffOptimizationEngine(km, groq_client=_mock_handoff_groq(call_counter=calls)).analyze(state)
    assert len(calls) == 1
    assert ana["summary"]["groq_calls_made"] == 1
    assert "legal_background_summary" in cands[0]["supporting_evidence"]["fields_excluded"]


def test_12_no_groq_call_for_straightforward_deterministic_cases():
    """Deterministic Python resolves clear document/metadata exclusions with zero Groq calls."""
    state = _make_handoff_state(_standard_two_agent_agents(), [("document_analyzer", "risk_analyzer")])
    calls: List[int] = []
    cands, ana = DataHandoffOptimizationEngine(_dummy_km(), groq_client=_mock_handoff_groq(call_counter=calls)).analyze(state)
    assert calls == []
    assert ana["summary"]["groq_calls_made"] == 0
    assert ana["summary"]["handoffs_analyzed_without_groq"] == 1
    assert len(cands) == 1


def test_13_groq_malformed_response_and_hard_constraint_override_rejection():
    """Malformed JSON or Groq trying to drop a Python-protected required field must be rejected safely."""
    agents = _standard_two_agent_agents(extra_payload={"legal_background_summary": "Long background overview " * 15})
    agents[1]["goal"] = "Provide concise summary of legal risk deviations."
    state = _make_handoff_state(agents, [("document_analyzer", "risk_analyzer")])

    # Case A: Malformed JSON -> fallback keeps uncertain field safely
    cands_bad, _ = DataHandoffOptimizationEngine(
        _dummy_km(), groq_client=_mock_handoff_groq(response_fn=lambda _: "{broken json")
    ).analyze(state)
    assert "legal_background_summary" in cands_bad[0]["supporting_evidence"]["fields_retained"]
    assert cands_bad[0]["supporting_evidence"]["groq_semantic_analysis"]["accepted"] is False

    # Case B: Groq tries to put required `contract_id` into redundant_fields -> rejected by hard constraint check
    ok, _, reason = validate_groq_semantic_output(
        expected_producer="document_analyzer",
        expected_consumer="risk_analyzer",
        transmitted_fields=["contract_id", "legal_background_summary"],
        python_mapping={"required_fields": ["contract_id"], "used_fields": [], "conditional_fields_preserved": []},
        raw_result={
            "producer_agent": "document_analyzer",
            "consumer_agent": "risk_analyzer",
            "field_assessments": {},
            "redundant_fields": ["contract_id"],
            "duplicate_fields": [],
            "required_fields": [],
            "uncertain_fields": [],
            "proposed_compact_payload_schema": {},
            "semantic_rationale": "drop contract_id",
            "confidence": 0.95,
            "requires_human_or_empirical_validation": True,
        },
    )
    assert ok is False and "hard_constraint_violation_protected_fields" in reason


def test_14_groq_low_confidence_response_handling():
    """Low-confidence Groq recommendations (<0.75) are rejected and uncertain fields retained."""
    ok, _, reason = validate_groq_semantic_output(
        expected_producer="p",
        expected_consumer="c",
        transmitted_fields=["a", "b"],
        python_mapping={"required_fields": ["a"], "used_fields": [], "conditional_fields_preserved": []},
        raw_result={
            "producer_agent": "p",
            "consumer_agent": "c",
            "field_assessments": {},
            "redundant_fields": ["b"],
            "duplicate_fields": [],
            "required_fields": ["a"],
            "uncertain_fields": [],
            "proposed_compact_payload_schema": {},
            "semantic_rationale": "uncertain",
            "confidence": 0.55,
            "requires_human_or_empirical_validation": True,
        },
    )
    assert ok is False and "low_confidence" in reason


# =============================================================================
# 15, 16, 17, 18, 19. Compact Proposals, Token Math & No Double-Counting
# =============================================================================

def test_15_compact_payload_recommendation_generation():
    """Original payload and global state remain unmodified while compact proposal omits full_document."""
    agents = _standard_two_agent_agents()
    orig_snapshot = copy.deepcopy(agents[0]["runtime_output"])
    state = _make_handoff_state(agents, [("document_analyzer", "risk_analyzer")])
    cands, _ = DataHandoffOptimizationEngine(_dummy_km(), groq_client=_mock_handoff_groq()).analyze(state)

    assert state["execution_evidence"]["document_analyzer"]["output"] == orig_snapshot  # Original untouched
    c = cands[0]
    prop_sample = c["supporting_evidence"]["proposed_compact_payload_sample"]
    assert set(prop_sample.keys()) == {"contract_id", "diff_clauses", "flagged_items", "risk_evidence"}
    assert len(c["supporting_evidence"]["fields_replaced_with_reference"]) == 1


def test_16_token_savings_calculation():
    """Verify single-tokenizer token_count difference matches candidate savings."""
    state = _make_handoff_state(_standard_two_agent_agents(), [("document_analyzer", "risk_analyzer")])
    cands, _ = DataHandoffOptimizationEngine(_dummy_km(), groq_client=_mock_handoff_groq()).analyze(state)
    ev = cands[0]["supporting_evidence"]

    orig_tok = count_tokens(json.dumps(ev["original_payload_sample"], sort_keys=True, ensure_ascii=False), "openai/gpt-oss-120b")["tokens"]
    prop_tok = count_tokens(json.dumps(ev["proposed_compact_payload_sample"], sort_keys=True, ensure_ascii=False), "openai/gpt-oss-120b")["tokens"]
    assert ev["original_payload_tokens"] == orig_tok
    assert ev["proposed_payload_tokens"] == prop_tok
    assert ev["tokens_saved"] == orig_tok - prop_tok
    assert ev["tokens_saved"] > 0


def test_17_percentage_reduction_calculation():
    """Verify percentage reduction equals round(tokens_saved / original_payload_tokens * 100, 2)."""
    state = _make_handoff_state(_standard_two_agent_agents(), [("document_analyzer", "risk_analyzer")])
    cands, ana = DataHandoffOptimizationEngine(_dummy_km(), groq_client=_mock_handoff_groq()).analyze(state)
    ev = cands[0]["supporting_evidence"]
    expected_pct = round((ev["tokens_saved"] / ev["original_payload_tokens"]) * 100.0, 2)
    assert ev["savings_percentage"] == expected_pct
    assert ana["summary"]["percentage_reduction"] == expected_pct


def test_18_measured_vs_estimated_and_static_savings_distinction():
    """Static-only handoffs must NOT fabricate runtime token counts or savings percentages."""
    agents_static = [
        {"agent_id": "p", "declared_outputs": ["full_document", "contract_id"], "runtime_output": None},
        {"agent_id": "c", "declared_inputs": ["contract_id"], "prompt_text": "Inspect {contract_id} clauses"},
    ]
    cands, ana = DataHandoffOptimizationEngine(
        _dummy_km(), groq_client=_mock_handoff_groq()
    ).analyze(_make_handoff_state(agents_static, [("p", "c")]))

    assert len(cands) == 1
    ev = cands[0]["supporting_evidence"]
    assert ev["evidence_basis"] == "statically_inferred_schema_only"
    assert ev["original_payload_tokens"] is None
    assert ev["tokens_saved"] is None
    assert ev["savings_percentage"] is None
    assert ana["summary"]["statically_inferred_handoffs"] == 1
    assert ana["summary"]["tokens_saved"] is None


def test_19_no_double_counting_of_repeated_trace_measurements():
    """Observation count (30 traces) must stay separate and never multiply per-handoff aggregate token sums."""
    state = _make_handoff_state(_standard_two_agent_agents(), [("document_analyzer", "risk_analyzer")])
    cands, ana = DataHandoffOptimizationEngine(_dummy_km(), groq_client=_mock_handoff_groq()).analyze(state)
    assert cands[0]["supporting_evidence"]["observation_count"] == 30
    assert ana["summary"]["tokens_saved"] == cands[0]["supporting_evidence"]["tokens_saved"]


# =============================================================================
# 20, 21, 22, 23. Candidate Schema, Tradeoff, Reports & All Dimensions
# =============================================================================

def test_20_candidate_schema_and_requires_validation_true():
    """Validate all required OptimizationCandidate and supporting_evidence keys."""
    state = _make_handoff_state(_standard_two_agent_agents(), [("document_analyzer", "risk_analyzer")])
    cands, _ = DataHandoffOptimizationEngine(_dummy_km(), groq_client=_mock_handoff_groq()).analyze(state)
    c = cands[0]

    assert c["dimension"] == "communication_data_handoff"
    assert c["requires_validation"] is True
    assert "proposed_compact_payload_schema" in c
    assert "GROQ_API_KEY_9" in c["allocated_key"]
    required_ev = {
        "producer_agent", "consumer_agent", "verified_dependency_relationship",
        "original_transmitted_fields", "consumer_required_fields", "consumer_used_fields",
        "potentially_redundant_fields", "duplicate_fields", "uncertain_fields",
        "original_payload_tokens", "proposed_payload_tokens", "tokens_saved", "savings_percentage",
        "original_payload_bytes", "proposed_payload_bytes", "observation_count",
        "tokenization_method", "python_analysis_results", "groq_semantic_analysis", "evidence_provenance",
    }
    assert required_ev.issubset(c["supporting_evidence"].keys())


def test_21_stage12_tradeoff_integration_tier2():
    """Stage 12 enriches handoff tradeoff details and assigns Tier 2 with requires_validation=True."""
    state = _make_handoff_state(_standard_two_agent_agents(), [("document_analyzer", "risk_analyzer")])
    cands, ana = DataHandoffOptimizationEngine(_dummy_km(), groq_client=_mock_handoff_groq()).analyze(state)
    state["optimization_candidates"] = cands
    state["dimension_analyses"] = {"communication_optimization": ana}

    t_res = tradeoff_assessment(state)
    synced = t_res["optimization_candidates"][0]
    assert synced["tier"] == "Tier 2"
    assert synced["feasibility_tier"] == TIER_2_LABEL
    assert synced["requires_validation"] is True
    assert t_res["tradeoff_assessments"][0]["tradeoffs"]["schema_compatibility_check_required"] is True


def test_22_report_json_markdown_and_dashboard_compatibility():
    """Verify all 15 Section 7 metrics, 9-row candidate table in Markdown, and dict/list UI shape."""
    state = _make_handoff_state(_standard_two_agent_agents(), [("document_analyzer", "risk_analyzer")])
    cands, ana = DataHandoffOptimizationEngine(_dummy_km(), groq_client=_mock_handoff_groq()).analyze(state)
    state["optimization_candidates"] = cands
    state["dimension_analyses"] = {"communication_optimization": ana}
    state.update(tradeoff_assessment(state))

    rep = generate_json_report(state, "GROQ_API_KEY_13 (masked: gsk_***...test)")
    sec = rep["communication_optimization_opportunities"]
    assert isinstance(sec, dict)
    expected_15 = {
        "total_verified_handoffs_identified", "handoffs_with_runtime_payload_evidence",
        "statically_inferred_handoffs", "large_payloads_detected",
        "potential_redundant_fields_identified", "duplicate_fields_identified",
        "uncertain_fields_identified", "handoffs_analyzed_without_groq",
        "handoffs_requiring_groq_semantic_analysis", "groq_calls_made",
        "compact_payload_recommendations_generated", "original_payload_tokens",
        "proposed_payload_tokens", "tokens_saved", "percentage_reduction",
    }
    assert expected_15.issubset(sec["summary"].keys())
    assert len(sec["candidates"]) == 1

    md = generate_markdown_report(rep)
    assert "### Communication & Data-Handoff Optimization" in md
    assert "| Producer | `document_analyzer` |" in md
    assert "| Consumer | `risk_analyzer` |" in md
    assert "REQUIRES VALIDATION" in md


def test_23_compatibility_with_all_other_dimensions_real_mcp():
    """Run all 8 dimensions on real contract_review MCP evidence with mocked Groq clients."""
    state: OptimizationAgentState = {
        "workflow_id": "contract_review",
        "stage_status": {},
        "retrieval_errors": [],
        "analysis_limitations": [],
    }
    for fn in (
        stage_1_discovery, stage_2_overview, stage_3_graph, stage_4_profiles,
        stage_5_prompts, stage_6_latency, stage_7_tokens, stage_8_cost,
        stage_9_comparison, stage_10_evidence,
    ):
        state.update(fn(state))

    def _no_op_chat(sys_p: str, usr_p: str, model: str) -> str:
        return json.dumps({"ok": True})

    mock_cons = GroqConsolidationClient(_dummy_km(), chat_completion_fn=_no_op_chat, task_type="agent_consolidation")
    mock_opt = GroqConsolidationClient(_dummy_km(), chat_completion_fn=_no_op_chat, task_type="prompt_optimization")
    mock_handoff = _mock_handoff_groq()

    res = OptimizationDiscoveryEngine(
        key_manager=_dummy_km(),
        consolidation_client=mock_cons,
        prompt_opt_client=mock_opt,
        handoff_client=mock_handoff,
    ).discover_all_dimensions(state)
    state.update(res)
    state.update(tradeoff_assessment(state))
    rep = generate_json_report(state, "GROQ_API_KEY_13 (masked: gsk_***...test)")

    assert set(res["dimension_analyses"].keys()) == {
        "agent_pruning", "agent_consolidation", "workflow_parallelization",
        "model_selection", "prompt_optimization", "tool_optimization",
        "communication_optimization", "additional_optimization",
    }
    comm = rep["communication_optimization_opportunities"]
    assert comm["summary"]["total_verified_handoffs_identified"] == 12
    assert comm["summary"]["handoffs_with_runtime_payload_evidence"] == 12
    assert len(comm["candidates"]) >= 1
    assert all(c["requires_validation"] is True and c["tier"] == "Tier 2" for c in comm["candidates"])
