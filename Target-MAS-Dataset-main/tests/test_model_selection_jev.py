"""
Unit and Integration Test Suite for JEV-Based Agentic Model Selection.

Tests all 9 mandatory validation test cases specified in the directive:
  1. Clause Mapping Agent baseline and eligibility
  2. Specialized Model Exclusion (Safeguard & Prompt Guard)
  3. No Eligible Models handling
  4. Unknown JEV Selection rejection
  5. Tool-Calling Constraint enforcement
  6. Context Window & Max Input Constraint enforcement
  7. Missing Evidence handling (no invented metrics)
  8. LangGraph sequential execution, state passing, and single-invocation guarantee
  9. Multi-agent compatibility and regression testing across workflows
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

from optimization_agent.model_selection.groq_catalog import (
    MODEL_CATALOG,
    resolve_available_llms,
)
from optimization_agent.model_selection.eligibility_filter import (
    filter_eligible_models,
    agent_requires_tools,
    estimate_context_requirement,
)
from optimization_agent.model_selection.agent1_requirement_analyzer import (
    build_agent1_questions,
    extract_score_answer,
    run_agent1_requirement_analyzer,
)
from optimization_agent.model_selection.agent2_model_selector import (
    build_agent2_state,
    run_agent2_model_selector,
    validate_model_selection,
)
from optimization_agent.model_selection.mcp_context_builder import (
    build_agent_context_from_mcp,
    calculate_workflow_token_reference,
)
from optimization_agent.model_selection.orchestrator import (
    build_model_selection_graph,
    run_model_selection,
)


# =============================================================================
# Mock Helpers
# =============================================================================

def create_mock_jev_caller(agent1_answers=None, agent2_choice="openai/gpt-oss-20b"):
    """Creates a mock call_jev function that records invocations and returns controlled outputs."""
    call_log = []

    def _mock_call(state, questions, **kwargs):
        call_log.append({"state": state, "questions": questions})
        # Determine whether this is Agent 1 or Agent 2 by inspecting question keys
        if "task_complexity" in questions:
            answers = agent1_answers or {
                "task_complexity": {
                    "type": "score",
                    "score": 1.0,
                    "confidence": 0.88,
                    "probabilities": {"0": 0.05, "1": 0.88, "2": 0.07},
                },
                "prompt_complexity": {
                    "type": "score",
                    "score": 2.0,
                    "confidence": 0.92,
                    "probabilities": {"0": 0.02, "1": 0.06, "2": 0.92},
                },
                "token_consumption": {
                    "type": "score",
                    "score": 1.0,
                    "confidence": 0.85,
                    "probabilities": {"0": 0.10, "1": 0.85, "2": 0.05},
                },
            }
            return {
                "model": "jev-mock-1.0",
                "answers": answers,
                "usage": {"input_tokens": 1500, "output_tokens": 60},
            }
        elif "recommended_model" in questions:
            return {
                "model": "jev-mock-1.0",
                "answers": {
                    "recommended_model": {
                        "type": "choice",
                        "choice": agent2_choice,
                        "value": agent2_choice,
                        "confidence": 0.92,
                        "probabilities": {agent2_choice: 0.94, "no_suitable_model": 0.06},
                    }
                },
                "usage": {"input_tokens": 2000, "output_tokens": 40},
            }
        else:
            raise ValueError(f"Unrecognized questions in mock: {list(questions.keys())}")

    _mock_call.call_log = call_log
    return _mock_call


# =============================================================================
# Test Case 1: Clause Mapping Agent Baseline & Eligibility
# =============================================================================

def test_test_case_1_clause_mapping_agent_mcp_and_eligibility():
    """Verify MCP evidence retrieval, context building, and eligibility for clause_mapping_agent."""
    workflow_id = "contract_review"
    agent_id = "clause_mapping_agent"

    context = build_agent_context_from_mcp(workflow_id, agent_id)

    # 1. Verify context fields
    assert context["workflow"]["workflow_id"] == workflow_id
    assert context["workflow"]["baseline_model"] == "openai/gpt-oss-120b"
    assert context["agent_definition"]["agent_id"] == agent_id
    assert "Clause-to-Policy Mapping" in context["agent_definition"]["role"]
    assert len(context["prompt_information"]) > 0
    assert context["execution_metrics"]["mean_total_tokens"] is not None

    # 2. Verify token reference quantiles
    token_ref = context["token_consumption_reference"]
    assert token_ref["low_upper_boundary_approx"] > 0
    assert token_ref["medium_upper_boundary_approx"] > token_ref["low_upper_boundary_approx"]

    # 3. Verify Groq catalog intersection and eligibility
    available_models, _ = resolve_available_llms(live_models={m: {"id": m, "active": True} for m in MODEL_CATALOG})
    eligible, excluded = filter_eligible_models(available_models, context)

    eligible_ids = {m["model_id"] for m in eligible}
    excluded_ids = {m["model_id"] for m in excluded}

    # Expected General-Purpose Eligible Models
    assert "openai/gpt-oss-120b" in eligible_ids
    assert "openai/gpt-oss-20b" in eligible_ids
    assert "qwen/qwen3.8-27b" in eligible_ids

    # Expected Excluded Specialized Models
    assert "openai/gpt-oss-safeguard-20b" in excluded_ids
    assert "meta-llama/llama-prompt-guard-2-22m" in excluded_ids
    assert "meta-llama/llama-prompt-guard-2-86m" in excluded_ids


# =============================================================================
# Test Case 2: Specialized Model Exclusion
# =============================================================================

def test_test_case_2_specialized_models_never_eligible_and_rejected_in_post_validation():
    """Verify Safeguard and Prompt Guard models are excluded and rejected if JEV attempts to choose them."""
    context = build_agent_context_from_mcp("contract_review", "clause_mapping_agent")
    all_models = [{"model_id": k, **v} for k, v in MODEL_CATALOG.items()]

    eligible, excluded = filter_eligible_models(all_models, context)

    for m in eligible:
        assert m["suitable_for_general_agent_tasks"] is True, f"Specialized model {m['model_id']} passed filter!"

    # Simulate post-validation rejecting a specialized model if JEV hallucinated it
    selected, chosen_meta, is_valid, reason = validate_model_selection(
        selected="openai/gpt-oss-safeguard-20b",
        candidates=eligible,  # Not in eligible candidates
    )
    assert is_valid is False
    assert selected is None
    assert "unknown or ineligible model ID" in reason.lower()


# =============================================================================
# Test Case 3: No Eligible Models
# =============================================================================

def test_test_case_3_no_eligible_models():
    """Verify that when no models are eligible, Agent 2 is not forced and returns clean status."""
    agent1_requirements = {
        "task_complexity": {"class": "High"},
        "prompt_complexity": {"class": "High"},
        "token_consumption": {"class": "High"},
    }
    context = {"agent_definition": {"agent_id": "test_agent"}}

    # Call with empty candidate list
    result = run_agent2_model_selector(
        agent1_requirements=agent1_requirements,
        agent_context=context,
        candidates=[],
    )

    assert result["selected_model"] is None
    assert result["selection_validated"] is False
    assert "No general-purpose Groq models passed deterministic eligibility checks" in result["reason"]


# =============================================================================
# Test Case 4: Unknown JEV Selection
# =============================================================================

def test_test_case_4_unknown_jev_selection_rejection():
    """Verify that unknown/hallucinated model IDs returned by JEV are rejected by post-validation."""
    candidates = [
        {"model_id": "openai/gpt-oss-120b", "suitable_for_general_agent_tasks": True},
        {"model_id": "openai/gpt-oss-20b", "suitable_for_general_agent_tasks": True},
    ]

    selected, meta, is_valid, reason = validate_model_selection(
        selected="anthropic/claude-3-opus",
        candidates=candidates,
    )

    assert is_valid is False
    assert selected is None
    assert meta is None
    assert "unknown or ineligible model ID" in reason


# =============================================================================
# Test Case 5: Tool-Calling Constraint
# =============================================================================

def test_test_case_5_tool_calling_constraint():
    """Verify models without tool-calling support are excluded when agent requires tools."""
    context = {
        "agent_definition": {
            "agent_id": "tool_heavy_agent",
            "registered_tools": ["custom_search_tool", "calculator"],
        },
        "task_information": [],
    }

    assert agent_requires_tools(context) is True

    models = [
        {
            "model_id": "tool_capable_model",
            "suitable_for_general_agent_tasks": True,
            "tool_calling_support": True,
            "context_window": 128000,
            "reasoning_capability": "High",
            "instruction_following": "High",
            "max_output_tokens": 4096,
        },
        {
            "model_id": "no_tool_model",
            "suitable_for_general_agent_tasks": True,
            "tool_calling_support": False,
            "context_window": 128000,
            "reasoning_capability": "High",
            "instruction_following": "High",
            "max_output_tokens": 4096,
        },
    ]

    eligible, excluded = filter_eligible_models(models, context)
    eligible_ids = [m["model_id"] for m in eligible]
    excluded_ids = [m["model_id"] for m in excluded]

    assert "tool_capable_model" in eligible_ids
    assert "no_tool_model" in excluded_ids
    assert any("tool calling" in r for r in excluded[0]["reasons"])


# =============================================================================
# Test Case 6: Context Window Constraint
# =============================================================================

def test_test_case_6_context_window_constraint():
    """Verify models with insufficient context window are excluded."""
    # Build large context
    context = {
        "agent_definition": {"agent_id": "large_context_agent", "registered_tools": []},
        "prompt_information": ["A" * 8000],  # ~2000 tokens
        "task_information": [],
    }

    est_tokens = estimate_context_requirement(context)
    assert est_tokens >= 2000

    models = [
        {
            "model_id": "tiny_model",
            "suitable_for_general_agent_tasks": True,
            "tool_calling_support": True,
            "context_window": 1000,  # Below estimated tokens
            "reasoning_capability": "High",
            "instruction_following": "High",
            "max_output_tokens": 512,
        },
        {
            "model_id": "large_model",
            "suitable_for_general_agent_tasks": True,
            "tool_calling_support": True,
            "context_window": 131072,
            "reasoning_capability": "High",
            "instruction_following": "High",
            "max_output_tokens": 4096,
        },
    ]

    eligible, excluded = filter_eligible_models(models, context)
    assert "large_model" in [m["model_id"] for m in eligible]
    assert "tiny_model" in [m["model_id"] for m in excluded]


# =============================================================================
# Test Case 7: Missing Evidence Handling
# =============================================================================

def test_test_case_7_missing_evidence_does_not_fabricate():
    """Verify that if an unknown workflow or invalid agent is requested, it fails cleanly."""
    with pytest.raises(Exception):
        # Invalid workflow ID
        build_agent_context_from_mcp("non_existent_workflow", "some_agent")

    with pytest.raises(Exception):
        # Invalid agent ID
        build_agent_context_from_mcp("contract_review", "phantom_agent_id_999")


# =============================================================================
# Test Case 8: LangGraph Sequential Execution
# =============================================================================

def test_test_case_8_langgraph_orchestration_and_call_counts():
    """Verify LangGraph executes Agent 1 then Agent 2 sequentially, each called exactly once."""
    mock_caller = create_mock_jev_caller(agent2_choice="openai/gpt-oss-20b")

    result = run_model_selection(
        workflow_id="contract_review",
        agent_id="clause_mapping_agent",
        custom_call_jev=mock_caller,
        live_groq_models={m: {"id": m, "active": True} for m in MODEL_CATALOG},
        export_report=False,
    )

    # Exactly 2 JEV API calls made in total (1 for Agent 1, 1 for Agent 2)
    assert len(mock_caller.call_log) == 2, f"Expected 2 JEV calls, got {len(mock_caller.call_log)}"

    # First call must be Agent 1 (questions has 'task_complexity')
    assert "task_complexity" in mock_caller.call_log[0]["questions"]

    # Second call must be Agent 2 (questions has 'recommended_model')
    assert "recommended_model" in mock_caller.call_log[1]["questions"]

    # Agent 2 state must contain Agent 1 classifications
    a2_state = mock_caller.call_log[1]["state"]
    assert a2_state["agent_requirements"]["task_complexity"] == "Medium"
    assert a2_state["agent_requirements"]["prompt_complexity"] == "High"
    assert a2_state["agent_requirements"]["token_consumption"] == "Medium"

    # Verify structured result
    assert result["workflow_id"] == "contract_review"
    assert result["selected_agent_id"] == "clause_mapping_agent"
    assert result["baseline_model"] == "openai/gpt-oss-120b"
    assert result["model_recommendation"]["selected_model"] == "openai/gpt-oss-20b"
    assert result["model_recommendation"]["selection_validated"] is True
    assert result["selection_validation"]["is_downgrade_candidate"] is True
    assert result["selection_validation"]["requires_empirical_quality_validation"] is True


# =============================================================================
# Test Case 9: Multi-Agent Compatibility (Not Hardcoded)
# =============================================================================

def test_test_case_9_multi_agent_compatibility():
    """Verify that the implementation works dynamically for other agents and workflows."""
    mock_caller = create_mock_jev_caller(agent2_choice="openai/gpt-oss-20b")

    # Test on contract_intake_agent
    res1 = run_model_selection(
        workflow_id="contract_review",
        agent_id="contract_intake_agent",
        custom_call_jev=mock_caller,
        live_groq_models={m: {"id": m, "active": True} for m in MODEL_CATALOG},
        export_report=False,
    )
    assert res1["selected_agent_id"] == "contract_intake_agent"
    assert res1["model_recommendation"]["selected_model"] == "openai/gpt-oss-20b"

    # Test on invoice_processing / po_matching_agent
    res2 = run_model_selection(
        workflow_id="invoice_processing",
        agent_id="po_matching_agent",
        custom_call_jev=mock_caller,
        live_groq_models={m: {"id": m, "active": True} for m in MODEL_CATALOG},
        export_report=False,
    )
    assert res2["workflow_id"] == "invoice_processing"
    assert res2["selected_agent_id"] == "po_matching_agent"
    assert res2["baseline_model"] == "openai/gpt-oss-120b"

