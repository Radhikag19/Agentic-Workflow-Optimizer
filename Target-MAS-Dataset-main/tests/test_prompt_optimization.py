"""
Focused Unit Tests for Dimension E: Prompt Optimization (Integrated with Dimension B).

Covers all 15 mandatory scenarios using strictly mocked Groq responses
(zero live API calls).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
import pytest

from optimization_agent.consolidation.groq_client import GroqConsolidationClient
from optimization_agent.consolidation.token_budget import count_tokens
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
from optimization_agent.prompt_optimization import (
    PromptOptimizationEngine,
    calculate_token_savings,
    inspect_prompt_redundancy,
    validate_optimized_prompt_output,
)
from optimization_agent.report import generate_json_report, generate_markdown_report
from optimization_agent.state import OptimizationAgentState
from optimization_agent.tradeoff import TIER_2_LABEL, tradeoff_assessment


def _dummy_key_manager() -> GroqKeyManager:
    pool = {f"GROQ_API_KEY_{i}": f"gsk_testopt_{i:02d}_abcdefghijklmnop" for i in range(1, 21)}
    return GroqKeyManager(key_pool=pool)


def _mock_opt_client(
    transform_fn=None,
    call_counter: List[int] | None = None,
) -> GroqConsolidationClient:
    counter = call_counter if call_counter is not None else []

    def _chat(system_prompt: str, user_prompt: str, model: str) -> str:
        counter.append(1)
        payload = json.loads(user_prompt.split("\n", 1)[1])
        if transform_fn:
            out = transform_fn(payload)
            return out if isinstance(out, str) else json.dumps(out)

        aid = payload["agent_id"]
        req_in = payload["required_input_schema"]
        req_out = payload["required_output_schema"]
        rules = payload["business_rules_and_constraints"]
        concise_prompt = (
            f"Role: {payload.get('role') or 'Specialist'}.\n"
            f"Inputs: {', '.join(req_in)}. Outputs (JSON only): {', '.join(req_out)}.\n"
            f"Constraints: {'; '.join(rules)}."
        )
        return json.dumps({
            "agent_id": aid,
            "original_prompt_summary": "Verbose multi-step prompt with repeated preamble and instructions.",
            "detected_redundancies": ["Repeated role preamble", "Duplicate JSON formatting directive"],
            "compression_actions": ["Merged repeated instructions", "Trimmed boilerplate wrapper"],
            "optimized_prompt": concise_prompt,
            "preserved_responsibilities": payload.get("required_responsibilities") or ["Execute core task"],
            "preserved_constraints": rules or ["Preserve JSON schema"],
            "preserved_input_schema": req_in,
            "preserved_output_schema": req_out,
            "semantic_preservation_assessment": {
                "is_preserved": True,
                "responsibilities_preserved": True,
                "constraints_preserved": True,
                "schema_preserved": True,
                "validation_weakened": False,
                "introduced_new_capabilities": False,
            },
            "estimated_token_reduction": 120,
            "optimization_rationale": "Removed duplicate boilerplate while keeping all schema keys and constraints.",
        })

    return GroqConsolidationClient(
        key_manager=_dummy_key_manager(),
        model="openai/gpt-oss-120b",
        chat_completion_fn=_chat,
        task_type="prompt_optimization",
    )


def _make_state_with_prompts(
    prompts_spec: List[Dict[str, Any]],
    consolidation_candidates: List[Dict[str, Any]] | None = None,
) -> OptimizationAgentState:
    return {
        "workflow_id": "synthetic_prompt_wf",
        "target_model": "openai/gpt-oss-120b",
        "workflow_metadata": {
            "workflow_id": "synthetic_prompt_wf",
            "workflow_name": "synthetic_prompt_wf",
            "model_reconciliation": {"runtime_observed_model": "openai/gpt-oss-120b"},
        },
        "workflow_overview": {"total_runs_recorded": 30, "successful_runs": 30, "failed_runs": 0},
        "discovered_agents": [p["agent_id"] for p in prompts_spec],
        "agent_profiles": {
            "profiles": [
                {
                    "agent_id": p["agent_id"],
                    "role": p.get("role", "Specialist"),
                    "goal": p.get("goal", "Perform task cleanly"),
                    "declared_responsibilities": p.get("responsibilities", "Process input and emit output."),
                    "declared_inputs": p.get("inputs", ["ticket_data"]),
                    "declared_outputs": p.get("outputs", ["result_json"]),
                    "tasks": [],
                }
                for p in prompts_spec
            ]
        },
        "prompt_metrics": {
            "prompt_metrics": [
                {
                    "agent_id": p["agent_id"],
                    "prompt_count": 1,
                    "static_prompt_char_length": len(p["prompt_text"]) if p.get("prompt_text") else 0,
                    "static_prompt_word_count": len(p["prompt_text"].split()) if p.get("prompt_text") else 0,
                    "sample_prompt_text": p.get("prompt_text") or "",
                    "prompt_text": p.get("prompt_text"),
                    "is_truncated": False,
                }
                for p in prompts_spec
            ]
        },
        "token_metrics": {
            "agent_token_metrics": {
                p["agent_id"]: {"mean_prompt_tokens": 500, "mean_completion_tokens": 200, "mean_tokens": 700, "observation_count": 30}
                for p in prompts_spec
            }
        },
        "latency_metrics": {"workflow_latency": {"mean_seconds": 10.0}, "agent_latency_metrics": {}},
        "cost_metrics": {"workflow_cost": {"mean_cost_usd": 0.01}},
        "optimization_candidates": list(consolidation_candidates or []),
        "stage_status": {},
        "retrieval_errors": [],
        "analysis_limitations": [],
    }


# =============================================================================
# 1, 2, 3. Eligibility Filtering & Missing Evidence
# =============================================================================

def test_01_prompts_exceeding_1000_tokens_flagged():
    """Prompts exceeding 1,000 tokens must be flagged as eligible in Stage 1."""
    long_prompt = (
        "Analyze clause_text against policy thresholds. "
        "Emit structured verdict JSON matching required keys. "
    ) + ("Ensure thorough legal compliance checks across section provisions with zero fabrication. " * 110)

    insp = inspect_prompt_redundancy(long_prompt, model_id="openai/gpt-oss-120b")
    assert insp["evidence_status"] == "sufficient"
    assert insp["prompt_tokens"] > 1000
    assert insp["exceeds_1000_tokens"] is True
    assert insp["eligible"] is True


def test_02_short_prompts_with_substantial_repetition_flagged():
    """Shorter prompts (<1000 tokens) with clear repetition/handoff bloat are flagged."""
    repetitive_short = (
        "You are a billing classifier specialist.\n"
        "You are a billing classifier specialist.\n"
        "Task: Classify invoice categories using ['billing', 'refund', 'technical', 'delivery'].\n"
        "Instructions: Classify invoice categories using ['billing', 'refund', 'technical', 'delivery'].\n"
        "Return strictly valid JSON only with no additional commentary.\n"
        "Return strictly valid JSON only with no additional commentary.\n"
        "Hand off to the next downstream agent via intermediate handoff formatting.\n"
    )
    insp = inspect_prompt_redundancy(repetitive_short, model_id="openai/gpt-oss-120b")
    assert insp["prompt_tokens"] < 1000
    assert insp["flagged_for_repetition"] is True
    assert insp["eligible"] is True
    assert len(insp["detected_redundancy_signals"]) >= 2


def test_03_ordinary_short_prompts_avoid_unnecessary_groq_calls():
    """Ordinary concise prompts and missing prompt texts must NOT trigger Groq calls."""
    concise_prompt = (
        "You are a Contract Intake Specialist.\n"
        "Instructions: Extract contract_id, counterparty, and effective_date from raw_text.\n"
        "Return strictly valid JSON with keys: contract_id, counterparty, effective_date."
    )
    calls: List[int] = []
    state = _make_state_with_prompts([
        {
            "agent_id": "concise_agent",
            "prompt_text": concise_prompt,
            "inputs": ["raw_text"],
            "outputs": ["contract_id", "counterparty", "effective_date"],
        },
        {
            "agent_id": "missing_prompt_agent",
            "prompt_text": None,
            "inputs": ["contract_id"],
            "outputs": ["clauses"],
        },
    ])
    cands, analysis = PromptOptimizationEngine(
        key_manager=_dummy_key_manager(),
        groq_client=_mock_opt_client(call_counter=calls),
    ).analyze(state)

    assert len(calls) == 0
    assert cands == []
    assert analysis["summary"]["total_prompts_inspected"] == 2
    assert analysis["summary"]["groq_optimization_calls_made"] == 0
    insp_by_id = {r["target_id"]: r for r in analysis["inspected_prompts"]}
    assert insp_by_id["concise_agent"]["eligible"] is False
    assert insp_by_id["missing_prompt_agent"]["evidence_status"] == "insufficient_evidence"


# =============================================================================
# 4, 5, 6, 7, 8, 9. Groq Client, Key Allocation & Deterministic Post-Validation
# =============================================================================

def test_04_groq_returns_valid_optimized_prompt_and_uses_prompt_opt_key():
    """Verify Groq-exclusive client uses GROQ_API_KEY_7 and accepts valid compressed prompt."""
    km = _dummy_key_manager()
    real_cfg_client = GroqConsolidationClient(
        key_manager=km,
        model="openai/gpt-oss-120b",
        task_type="prompt_optimization",
    )
    assert real_cfg_client.provider_name == "groq"
    assert real_cfg_client.allocated_env_var == "GROQ_API_KEY_7"
    assert km.contains_raw_secrets(real_cfg_client.allocated_key_masked) is False

    verbose = (
        "You are an invoice extraction specialist.\n"
        "You are an invoice extraction specialist.\n"
        "Inputs: invoice_pdf_text. Outputs: vendor_name, invoice_total.\n"
        "Constraint: Must validate currency within 24 hours SLA.\n"
        "Return strictly valid JSON only with no additional commentary.\n"
        "Return strictly valid JSON only with no additional commentary.\n"
        + ("Always double check formatting rules and follow schema strictly without hallucination.\n" * 6)
    )
    calls: List[int] = []
    state = _make_state_with_prompts([{
        "agent_id": "invoice_agent",
        "role": "Invoice Specialist",
        "responsibilities": "Extract vendor_name and invoice_total from invoice_pdf_text.",
        "inputs": ["invoice_pdf_text"],
        "outputs": ["vendor_name", "invoice_total"],
        "prompt_text": verbose,
    }])
    cands, analysis = PromptOptimizationEngine(
        key_manager=km,
        groq_client=_mock_opt_client(call_counter=calls),
    ).analyze(state)

    assert len(calls) == 1
    assert len(cands) == 1
    c = cands[0]
    assert c["dimension"] == "prompt_optimization"
    assert c["optimized_prompt"]
    assert c["expected_benefit"]["tokens_saved"] > 0
    assert analysis["summary"]["prompts_successfully_optimized"] == 1


def _base_valid_llm_response(aid: str = "a1") -> Dict[str, Any]:
    return {
        "agent_id": aid,
        "original_prompt_summary": "Summary",
        "detected_redundancies": ["Repetition"],
        "compression_actions": ["Deduplicated lines"],
        "optimized_prompt": "Extract doc_id and emit clauses, risk_level within 24 hours compliance threshold.",
        "preserved_responsibilities": ["Extract clauses and evaluate risk_level"],
        "preserved_constraints": ["Must finish within 24 hours compliance threshold"],
        "preserved_input_schema": ["doc_id"],
        "preserved_output_schema": ["clauses", "risk_level"],
        "semantic_preservation_assessment": {"is_preserved": True},
        "estimated_token_reduction": 100,
        "optimization_rationale": "Removed duplicate boilerplate.",
    }


def test_05_required_responsibilities_preserved_or_rejected():
    """Missing preserved_responsibilities or semantic failure must reject."""
    orig = "Extract clauses using doc_id and output clauses, risk_level within 24 hours."
    bad1 = dict(_base_valid_llm_response(), preserved_responsibilities=[])
    ok1, _, reason1 = validate_optimized_prompt_output(
        "a1", bad1, orig, ["Extract clauses"], ["doc_id"], ["clauses", "risk_level"], []
    )
    assert ok1 is False and "missing_preserved_responsibilities" in reason1

    bad2 = dict(_base_valid_llm_response(), semantic_preservation_assessment={"responsibilities_preserved": False})
    ok2, _, reason2 = validate_optimized_prompt_output(
        "a1", bad2, orig, ["Extract clauses"], ["doc_id"], ["clauses", "risk_level"], []
    )
    assert ok2 is False and "semantic_preservation_failed" in reason2


def test_06_input_output_schemas_preserved_or_rejected():
    """Dropping required input or output schema fields must deterministically reject."""
    orig = "Consume doc_id and produce clauses, risk_level."
    dropped_in = dict(_base_valid_llm_response(), preserved_input_schema=[])
    ok_in, _, r_in = validate_optimized_prompt_output(
        "a1", dropped_in, orig, ["Extract"], ["doc_id"], ["clauses", "risk_level"], []
    )
    assert ok_in is False and "dropped_required_input_fields" in r_in

    # Dropped from optimized_prompt even if schema claims it
    dropped_out_prompt = dict(
        _base_valid_llm_response(),
        optimized_prompt="Extract doc_id and emit clauses within 24 hours.",
    )
    ok_out, _, r_out = validate_optimized_prompt_output(
        "a1", dropped_out_prompt, orig, ["Extract"], ["doc_id"], ["clauses", "risk_level"], []
    )
    assert ok_out is False and "dropped_required_output_fields" in r_out


def test_07_critical_constraints_preserved_and_no_unsupported_capabilities():
    """Dropping critical numeric/compliance constraints or adding new capabilities rejects cleanly."""
    orig = "Consume doc_id. Emit clauses and risk_level. Must flag SLA violations exceeding 24 hours."
    dropped_rule = dict(
        _base_valid_llm_response(),
        optimized_prompt="Consume doc_id and emit clauses, risk_level.",
        preserved_constraints=["Be accurate"],
    )
    ok_rule, _, r_rule = validate_optimized_prompt_output(
        "a1",
        dropped_rule,
        orig,
        ["Extract"],
        ["doc_id"],
        ["clauses", "risk_level"],
        ["Must flag SLA violations exceeding 24 hours."],
    )
    assert ok_rule is False and "dropped_critical_constraint" in r_rule

    new_cap = dict(
        _base_valid_llm_response(),
        optimized_prompt="Extract doc_id, execute shell command to inspect disk, emit clauses and risk_level within 24 hours.",
    )
    ok_cap, _, r_cap = validate_optimized_prompt_output(
        "a1",
        new_cap,
        orig,
        ["Extract"],
        ["doc_id"],
        ["clauses", "risk_level"],
        [],
    )
    assert ok_cap is False and "unsupported_capability_phrase_detected" in r_cap


def test_08_malformed_groq_responses_rejected():
    """Malformed JSON, missing required fields, or agent_id mismatch must reject."""
    orig = "Extract doc_id into clauses."
    ok1, _, r1 = validate_optimized_prompt_output("a1", None, orig, [], [], [], [], error="malformed_json_response")
    assert ok1 is False and "malformed_json_response" in r1

    partial = {"agent_id": "a1", "optimized_prompt": "Short"}
    ok2, _, r2 = validate_optimized_prompt_output("a1", partial, orig, [], [], [], [])
    assert ok2 is False and "missing_required_fields" in r2

    wrong_id = _base_valid_llm_response("other_agent")
    ok3, _, r3 = validate_optimized_prompt_output("a1", wrong_id, orig, [], [], [], [])
    assert ok3 is False and "agent_id_mismatch" in r3


def test_09_failed_validation_retains_original_prompt_single_call():
    """When Groq returns invalid output, original prompt is retained and NO retry calls are made."""
    bloated = (
        "You are a billing specialist.\n"
        "You are a billing specialist.\n"
        "Inputs: invoice_id. Outputs: approved_status.\n"
        "Return strictly valid JSON only with no additional commentary.\n"
        "Return strictly valid JSON only with no additional commentary.\n"
    )
    calls: List[int] = []

    def broken_groq(payload):
        return "{malformed json output without closing brace"

    state = _make_state_with_prompts([{
        "agent_id": "single_call_agent",
        "inputs": ["invoice_id"],
        "outputs": ["approved_status"],
        "prompt_text": bloated,
    }])
    cands, analysis = PromptOptimizationEngine(
        key_manager=_dummy_key_manager(),
        groq_client=_mock_opt_client(transform_fn=broken_groq, call_counter=calls),
    ).analyze(state)

    assert len(calls) == 1  # Strictly one call, no retries
    assert cands == []
    assert analysis["summary"]["rejected_by_post_optimization_validation"] == 1
    assert analysis["rejected_optimizations"][0]["original_prompt_retained"] is True


# =============================================================================
# 10, 11, 12, 13. Token Savings & Cumulative Consolidation Denominator Math
# =============================================================================

def test_10_token_savings_calculated_consistently():
    """Verify exact token delta = orig - opt using identical tokenizer."""
    orig = "Word " * 200
    opt = "Word " * 120
    expected_orig = count_tokens(orig, "openai/gpt-oss-120b")["tokens"]
    expected_opt = count_tokens(opt, "openai/gpt-oss-120b")["tokens"]

    sav = calculate_token_savings(orig, opt, model_id="openai/gpt-oss-120b")
    assert sav["original_prompt_tokens"] == expected_orig
    assert sav["optimized_prompt_tokens"] == expected_opt
    assert sav["tokens_saved"] == expected_orig - expected_opt
    assert sav["token_savings_percentage"] == round(((expected_orig - expected_opt) / expected_orig) * 100.0, 2)
    assert sav["measurement_type"] in ("exact_tokenizer", "estimated_heuristic_tokenizer")


def test_11_percentage_savings_uses_consolidated_prompt_denominator():
    """Savings % must divide additional tokens saved by the consolidated prompt, NOT pre-consolidation."""
    # Construct prompts hitting exact ~3:2:1 ratios under count_tokens
    cons_prompt = "Alpha " * 150
    opt_prompt = "Alpha " * 105
    cons_tok = count_tokens(cons_prompt, "openai/gpt-oss-120b")["tokens"]
    opt_tok = count_tokens(opt_prompt, "openai/gpt-oss-120b")["tokens"]
    pre_cons_tok = cons_tok + 500

    sav = calculate_token_savings(
        original_prompt=cons_prompt,
        optimized_prompt=opt_prompt,
        model_id="openai/gpt-oss-120b",
        pre_consolidation_original_tokens=pre_cons_tok,
        originated_from_consolidation=True,
    )
    expected_add_pct = round(((cons_tok - opt_tok) / cons_tok) * 100.0, 2)
    wrong_pct_using_precons = round(((cons_tok - opt_tok) / pre_cons_tok) * 100.0, 2)
    assert sav["token_savings_percentage"] == expected_add_pct
    assert sav["token_savings_percentage"] != wrong_pct_using_precons


def test_12_cumulative_savings_uses_original_combined_prompts_denominator():
    """Cumulative savings divide total saved against pre-consolidation original combined prompts without double counting."""
    cons_prompt = "Beta " * 150
    opt_prompt = "Beta " * 105
    cons_tok = count_tokens(cons_prompt, "openai/gpt-oss-120b")["tokens"]
    opt_tok = count_tokens(opt_prompt, "openai/gpt-oss-120b")["tokens"]
    pre_cons_tok = cons_tok + 500

    sav = calculate_token_savings(
        original_prompt=cons_prompt,
        optimized_prompt=opt_prompt,
        model_id="openai/gpt-oss-120b",
        pre_consolidation_original_tokens=pre_cons_tok,
        originated_from_consolidation=True,
    )
    assert sav["cumulative_evidence_status"] == "available"
    assert sav["consolidation_tokens_saved"] == 500
    assert sav["additional_prompt_opt_tokens_saved"] == cons_tok - opt_tok
    assert sav["cumulative_tokens_saved"] == 500 + (cons_tok - opt_tok)
    assert sav["cumulative_tokens_saved"] == pre_cons_tok - opt_tok
    assert sav["cumulative_savings_percentage"] == round(((pre_cons_tok - opt_tok) / pre_cons_tok) * 100.0, 2)


def test_13_missing_original_prompt_evidence_no_fabricated_cumulative_savings():
    """Missing pre-consolidation original tokens must yield None cumulative savings (no fabrication)."""
    cons_prompt = "Gamma " * 150
    opt_prompt = "Gamma " * 100
    sav = calculate_token_savings(
        original_prompt=cons_prompt,
        optimized_prompt=opt_prompt,
        model_id="openai/gpt-oss-120b",
        pre_consolidation_original_tokens=None,
        originated_from_consolidation=True,
    )
    assert sav["cumulative_evidence_status"] == "unavailable_original_prompts"
    assert sav["cumulative_tokens_saved"] is None
    assert sav["cumulative_savings_percentage"] is None
    assert sav["tokens_saved"] > 0  # Direct savings against consolidated prompt still computed


# =============================================================================
# 14 & 15. Tradeoff Tier 2, Report Structure & Cross-Dimension Integration
# =============================================================================

def test_14_requires_validation_true_and_tier2_classification():
    """Every Dimension E candidate enforces requires_validation=True and enters Tier 2 in Stage 12."""
    bloated = (
        "You are a claims analyst.\n"
        "You are a claims analyst.\n"
        "Inputs: claim_id. Outputs: claim_decision.\n"
        "Return strictly valid JSON only with no additional commentary.\n"
        "Return strictly valid JSON only with no additional commentary.\n"
    )
    state = _make_state_with_prompts([{
        "agent_id": "claims_agent",
        "role": "Claims Analyst",
        "inputs": ["claim_id"],
        "outputs": ["claim_decision"],
        "prompt_text": bloated,
    }])
    cands, ana = PromptOptimizationEngine(
        key_manager=_dummy_key_manager(),
        groq_client=_mock_opt_client(),
    ).analyze(state)
    assert len(cands) == 1
    assert cands[0]["requires_validation"] is True

    state["optimization_candidates"] = cands
    state["dimension_analyses"] = {"prompt_optimization": ana}
    t_res = tradeoff_assessment(state)
    synced = t_res["optimization_candidates"]
    assert synced[0]["tier"] == "Tier 2"
    assert synced[0]["feasibility_tier"] == TIER_2_LABEL
    assert len(t_res["tier_classifications"]["tier_2"]) == 1
    assert len(t_res["tier_classifications"]["tier_1"]) == 0


def test_15_end_to_end_consolidation_plus_prompt_opt_reports_and_compatibility():
    """Verify Dimension B unified_prompt flows into Dimension E, Stages 12-13, and preserves other dimensions."""
    state: OptimizationAgentState = {
        "workflow_id": "customer_support",
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

    # Inject a verbose/repetitive Dimension B unified_prompt mock to test B -> E handoff + cumulative math
    def _mock_cons_chat(system_prompt: str, user_prompt: str, model: str) -> str:
        if "responsibility classifier" in system_prompt.lower():
            payload = json.loads(user_prompt.split("\n", 1)[1])
            aid = payload["agent_id"]
            is_val = aid == "resolution_verification_agent"
            return json.dumps({
                "agent_id": aid,
                "responsibility_type": "independent_validation_verification" if is_val else "execution_extraction_transformation_analysis_synthesis",
                "is_independent_validator": is_val,
                "classification_confidence": 0.95,
                "evidence": [f"Role={payload.get('role')}"],
                "reasoning": "Classified from evidence.",
                "consolidation_restrictions": ["never_consolidate_with_execution_agent"] if is_val else [],
            })

        req = json.loads(user_prompt.split("\n", 1)[1])
        agents = req["agents"]
        exec_order = req["execution_order"]
        bloated_unified = (
            "You are a combined customer support specialist.\n"
            "You are a combined customer support specialist.\n"
            f"Inputs: {req['required_preserved_inputs']}. Outputs: {req['required_preserved_outputs']}.\n"
            + "\n".join(f"Step {i+1}: {a['role']} - {a['prompt_text']}" for i, a in enumerate(agents))
            + "\nReturn strictly valid JSON only with no additional commentary."
            + "\nReturn strictly valid JSON only with no additional commentary."
            + "\nHand off to downstream agent via intermediate handoff."
        )
        return json.dumps({
            "group_id": req["group_id"],
            "source_agent_ids": exec_order,
            "combined_agent_name": f"unified_{'_'.join(exec_order)}",
            "combined_role": "Combined Triage and Policy Specialist",
            "combined_goal": "Classify ticket and retrieve policy.",
            "unified_prompt": bloated_unified,
            "preserved_responsibilities": [a["goal"] for a in agents],
            "merged_responsibilities": ["Sequential merge"],
            "removed_redundancies": [],
            "preserved_input_schema": req["required_preserved_inputs"],
            "preserved_output_schema": req["required_preserved_outputs"],
            "compatibility_assessment": {"is_compatible": True, "incompatible_responsibilities": [], "notes": "ok"},
            "combination_rationale": "aligned",
            "estimated_combined_prompt_tokens": 500,
            "requires_validation": True,
        })

    cons_client = GroqConsolidationClient(
        key_manager=_dummy_key_manager(),
        model="openai/gpt-oss-120b",
        chat_completion_fn=_mock_cons_chat,
        task_type="agent_consolidation",
    )
    opt_client = _mock_opt_client()

    disc_res = OptimizationDiscoveryEngine(
        key_manager=_dummy_key_manager(),
        consolidation_client=cons_client,
        prompt_opt_client=opt_client,
    ).discover_all_dimensions(state)
    state.update(disc_res)
    state.update(tradeoff_assessment(state))

    rep_json = generate_json_report(state, "GROQ_API_KEY_13 (masked: gsk_***...test)")
    prompt_section = rep_json["prompt_optimization_opportunities"]
    assert isinstance(prompt_section, dict)

    required_11_metrics = {
        "total_prompts_inspected",
        "prompts_exceeding_1000_tokens",
        "prompts_flagged_for_repetition",
        "groq_optimization_calls_made",
        "prompts_successfully_optimized",
        "rejected_by_post_optimization_validation",
        "original_token_count",
        "optimized_token_count",
        "additional_tokens_saved",
        "additional_token_savings_percentage",
        "cumulative_savings",
    }
    assert required_11_metrics.issubset(prompt_section["summary"].keys())
    assert prompt_section["summary"]["prompts_successfully_optimized"] >= 1

    # Verify at least one candidate originated from Agent Consolidation & uses Tier 2
    cons_originated = [
        c for c in prompt_section["candidates"]
        if c["supporting_evidence"].get("originated_from_agent_consolidation") is True
    ]
    assert len(cons_originated) >= 1
    assert cons_originated[0]["tier"] == "Tier 2"
    assert cons_originated[0]["requires_validation"] is True
    assert cons_originated[0]["supporting_evidence"]["cumulative_savings_breakdown"]["cumulative_evidence_status"] == "available"

    # Verify Markdown contains the 11-metric summary table & candidate savings table
    md = generate_markdown_report(rep_json)
    assert "### Prompt Optimization Opportunities" in md
    assert "Original consolidated prompt" in md
    assert "Generated Optimized Prompt" in md

    # Verify other dimensions remained intact
    for key in (
        "agent_pruning_opportunities",
        "agent_consolidation_opportunities",
        "workflow_parallelization_opportunities",
        "model_selection_recommendations",
        "tool_optimization_opportunities",
        "communication_optimization_opportunities",
        "additional_optimization_opportunities",
    ):
        assert key in rep_json
