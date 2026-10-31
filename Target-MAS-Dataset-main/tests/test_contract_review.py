"""
Unit Tests for Workflow 3: Contract Review and Compliance (10 Agents).

Tests execute with mocks to avoid consuming Groq API credits.
Validates 10 agents, clause-to-policy mapping, deviation detection,
cross-clause contradiction detection, and evidence citation integrity.
Smoke test is marked with @pytest.mark.smoke for live testing.
"""

import json
import os
from unittest.mock import MagicMock, patch
import pytest

from common.validation import (
    ContractIntakeOutput,
    ClauseExtractionOutput,
    PolicyInterpretationOutput,
    ClauseMappingOutput,
    ComplianceDeviationOutput,
    RiskAmbiguityOutput,
    CrossClauseConsistencyOutput,
    EvidenceValidationOutput,
    IndependentReviewOutput,
    ContractFinalReportOutput,
)
from Agents.contract_review.workflow import (
    ContractReviewWorkflow,
    load_sample_contract_input,
    EXPECTED_AGENT_COUNT,
)


def test_contract_review_agent_count():
    """Verify workflow implements exactly 10 agents."""
    workflow = ContractReviewWorkflow(enable_tracing=False)
    assert workflow.get_agent_count() == 10
    assert EXPECTED_AGENT_COUNT == 10


def test_contract_datasets():
    """Verify local synthetic contract datasets exist and adhere to schema."""
    contracts_path = "data/contract_review/contracts.json"
    policies_path = "data/contract_review/policies.json"

    assert os.path.exists(contracts_path)
    assert os.path.exists(policies_path)

    with open(contracts_path, "r", encoding="utf-8") as f:
        contracts = json.load(f)
    with open(policies_path, "r", encoding="utf-8") as f:
        policies = json.load(f)

    assert len(contracts) >= 100
    assert len(policies) >= 3

    sample = contracts[0]
    assert "contract_id" in sample
    assert "clauses" in sample
    assert len(sample["clauses"]) >= 3
    for cl in sample["clauses"]:
        assert "clause_id" in cl
        assert "clause_type" in cl
        assert "clause_text" in cl


def test_contract_workflow_mock_execution(tmp_path):
    """
    Test end-to-end execution of the 10-agent contract review workflow using mocks.
    """
    sample_input = load_sample_contract_input()

    mock_intake = {
        "document_id": "DOC-CTR-2026-0001",
        "contract_id": "CTR-2026-0001",
        "contract_title": "SaaS Subscription Agreement with Vendor 001",
        "contract_type": "SaaS Subscription Agreement",
        "contract_version": "1.0",
        "policy_id": "POL-CORP-SEC-01",
        "policy_version": "2.4",
        "document_complete": True,
        "section_count": 4,
        "source_metadata": {"party_a": "AWO Enterprise Corp", "party_b": "Vendor 001"},
        "intake_warnings": [],
    }

    mock_extract = {
        "contract_id": "CTR-2026-0001",
        "clauses": [
            {
                "clause_id": "CLS-0001-PAY",
                "clause_type": "payment_terms",
                "clause_title": "Section 3: Invoicing",
                "clause_text": "Invoices payable within 90 days of receipt.",
                "section_reference": "Section 3.1",
            },
            {
                "clause_id": "CLS-0001-DATA",
                "clause_type": "data_protection",
                "clause_title": "Section 8: Security",
                "clause_text": "Vendor shall notify within 24 hours of data breach.",
                "section_reference": "Section 8.2",
            },
            {
                "clause_id": "CLS-0001-LIAB",
                "clause_type": "liability_cap",
                "clause_title": "Section 10: Liability",
                "clause_text": "Liability capped at 12 months fees.",
                "section_reference": "Section 10.1",
            },
            {
                "clause_id": "CLS-0001-TERM",
                "clause_type": "termination",
                "clause_title": "Section 12: Term",
                "clause_text": "Terminate for convenience on 30 days notice.",
                "section_reference": "Section 12.1",
            },
        ],
        "absent_clause_types": [],
        "extraction_notes": [],
    }

    mock_policy = {
        "policy_id": "POL-CORP-SEC-01",
        "policy_version": "2.4",
        "policy_name": "Corporate Data Protection Standard",
        "mandatory_requirements": [
            {
                "requirement_id": "REQ-SEC-01",
                "clause_type": "data_protection",
                "is_mandatory": True,
                "requirement_text": "Breach notification within 48 hours",
                "threshold_or_condition": "48 hours",
                "exception_rules": [],
            },
            {
                "requirement_id": "REQ-FIN-01",
                "clause_type": "payment_terms",
                "is_mandatory": True,
                "requirement_text": "Payment terms Net 45 maximum",
                "threshold_or_condition": "Net 45",
                "exception_rules": [],
            },
        ],
        "recommended_requirements": [],
        "ambiguities_in_policy": [],
    }

    mock_mapping = {
        "contract_id": "CTR-2026-0001",
        "policy_id": "POL-CORP-SEC-01",
        "mappings": [
            {
                "mapping_id": "MAP-01",
                "clause_id": "CLS-0001-DATA",
                "requirement_id": "REQ-SEC-01",
                "mapping_status": "satisfied",
                "supporting_clause_text": "Vendor notifies in 24 hours.",
                "notes": "24 hours is stricter than 48 hours requirement.",
            },
            {
                "mapping_id": "MAP-02",
                "clause_id": "CLS-0001-PAY",
                "requirement_id": "REQ-FIN-01",
                "mapping_status": "conflicting",
                "supporting_clause_text": "Invoices payable within 90 days.",
                "notes": "Net 90 conflicts with Net 45 policy limit.",
            },
        ],
        "coverage_percentage": 100.0,
    }

    mock_dev = {
        "contract_id": "CTR-2026-0001",
        "deviations": [
            {
                "deviation_id": "DEV-01",
                "requirement_id": "REQ-FIN-01",
                "clause_id": "CLS-0001-PAY",
                "severity": "high",
                "deviation_summary": "Net 90 payment exceeds policy maximum of Net 45.",
                "supporting_evidence": "Invoices payable within 90 days of receipt.",
                "recommended_remediation": "Amend Section 3.1 to Net 45 days.",
            }
        ],
        "has_critical_deviations": False,
    }

    mock_risk = {
        "contract_id": "CTR-2026-0001",
        "risk_findings": [],
    }

    mock_consist = {
        "contract_id": "CTR-2026-0001",
        "inconsistencies": [],
    }

    mock_ev_val = {
        "contract_id": "CTR-2026-0001",
        "all_citations_valid": True,
        "verified_finding_ids": ["DEV-01"],
        "invalid_or_unsupported_findings": [],
        "evidence_notes": ["All clause references verified against extracted inventory."],
    }

    mock_rev = {
        "contract_id": "CTR-2026-0001",
        "review_status": "approved_for_report",
        "duplicate_findings_merged": 0,
        "severity_adjustments": [],
        "unresolved_matters": ["Confirm with finance if Net 90 payment is acceptable under exception."],
    }

    mock_final = {
        "run_id": "ctr_test_001",
        "contract_id": "CTR-2026-0001",
        "contract_version": "1.0",
        "policy_id": "POL-CORP-SEC-01",
        "policy_version": "2.4",
        "contract_type": "SaaS Subscription Agreement",
        "executive_summary": "Contract generally compliant except for high-severity payment terms deviation.",
        "policy_coverage": {"coverage_percentage": 100.0, "total_requirements": 2},
        "clause_inventory": mock_extract["clauses"],
        "compliance_findings": mock_dev["deviations"],
        "risk_findings": [],
        "cross_clause_inconsistencies": [],
        "evidence_validation": mock_ev_val,
        "unresolved_questions": ["Is commercial exception permissible for Net 90?"],
        "recommended_actions": ["Request vendor amendment to Net 45."],
        "human_review_required": True,
        "validation_warnings": [],
        "legal_disclaimer": "This AI-assisted review is for preliminary compliance screening only and is NOT a substitute for qualified legal or compliance counsel. No detected issues does NOT imply complete legal validity.",
    }

    returns = [
        mock_intake,
        mock_extract,
        mock_policy,
        mock_mapping,
        mock_dev,
        mock_risk,
        mock_consist,
        mock_ev_val,
        mock_rev,
        mock_final,
    ]

    from common.agent_runner import set_mock_task_handler
    iterator = iter(returns)
    set_mock_task_handler(lambda agent_id, desc: next(iterator))
    try:
        workflow = ContractReviewWorkflow(enable_tracing=False)
        result = workflow.run(input_data=sample_input, run_id="ctr_test_001", runs_base_dir=str(tmp_path))
    finally:
        set_mock_task_handler(None)

    assert result["run_id"] == "ctr_test_001"
    assert result["contract_id"] == "CTR-2026-0001"
    assert len(result["compliance_findings"]) == 1
    assert result["compliance_findings"][0]["severity"] == "high"
    assert result["human_review_required"] is True
    assert "NOT a substitute for qualified legal" in result["legal_disclaimer"]


@pytest.mark.smoke
def test_contract_live_smoke():
    """Live smoke test calling Groq API (requires valid GROQ_API_KEY)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not configured; skipping live smoke test")

    workflow = ContractReviewWorkflow(enable_tracing=True)
    result = workflow.run()
    assert result is not None
    assert "compliance_findings" in result
    assert "legal_disclaimer" in result
