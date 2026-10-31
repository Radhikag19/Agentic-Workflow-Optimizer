"""
Tests verifying end-to-end execution of all 4 workflows with mocked LLM calls.
Zero API credits consumed during this test suite.
"""

from unittest.mock import patch, MagicMock
import pytest
from common.settings import RUNS_DIR

from Agents.invoice_processing.workflow import run_invoice_processing_workflow
from Agents.customer_support.workflow import run_customer_support_workflow
from Agents.procurement_supplier_selection.workflow import run_procurement_workflow
from Agents.sales_lead_qualification.workflow import run_sales_lead_workflow


@pytest.fixture(autouse=True)
def mock_llm_execution():
    """Mock Crew.kickoff at instance level or mock litellm so instrumentation remains intact."""
    with patch("crewai.Crew.kickoff", autospec=True) as mock_kickoff:
        mock_kickoff.return_value = '{"status": "MOCKED_SUCCESS", "summary": "Audit completed under test."}'
        yield mock_kickoff


def test_invoice_processing_workflow_mocked():
    sample_invoice = {
        "invoice_id": "INV-TEST-001",
        "invoice_number": "INV-TEST-001",
        "supplier_id": "SUP-101",
        "po_reference": "PO-2026-0001",
        "total_amount": 1000.0,
        "line_items": []
    }
    result = run_invoice_processing_workflow(sample_invoice, run_id="test_inv_run_01")
    assert result["status"] == "SUCCESS"
    assert len(result["agent_executions"]) == 8
    assert (RUNS_DIR / "invoice_processing" / "test_inv_run_01.json").exists()


def test_customer_support_workflow_mocked():
    sample_ticket = {
        "ticket_id": "TICK-TEST-001",
        "customer_id": "CUST-0001",
        "category": "account_access",
        "subject": "Password reset needed"
    }
    result = run_customer_support_workflow(sample_ticket, run_id="test_cs_run_01")
    assert result["status"] == "SUCCESS"
    assert len(result["agent_executions"]) == 6
    assert (RUNS_DIR / "customer_support" / "test_cs_run_01.json").exists()


def test_procurement_workflow_mocked():
    sample_case = {
        "procurement_request": {"request_id": "RFQ-TEST-01", "item_description": "Machined Part", "target_delivery_date": "2026-11-01"},
        "quotations": [{"supplier_id": "PRO-SUP-01", "unit_price": 50.0, "item_total": 5000.0, "shipping_cost": 200.0, "tax_amount": 350.0}],
        "delivery_schedules": [{"supplier_id": "PRO-SUP-01", "lead_time_days": 20, "promised_delivery_date": "2026-10-25"}]
    }
    result = run_procurement_workflow(sample_case, run_id="test_pro_run_01")
    assert result["status"] == "SUCCESS"
    assert len(result["agent_executions"]) == 9
    assert (RUNS_DIR / "procurement_supplier_selection" / "test_pro_run_01.json").exists()


def test_sales_lead_workflow_mocked():
    sample_lead = {
        "lead_id": "LEAD-TEST-001",
        "company_id": "COMP-101",
        "contact_name": "Test Contact",
        "job_title": "CTO",
        "stated_budget": 50000.0,
        "purchase_timeline": "Within 3 months"
    }
    result = run_sales_lead_workflow(sample_lead, run_id="test_sales_run_01")
    assert result["status"] == "SUCCESS"
    assert len(result["agent_executions"]) == 7
    assert (RUNS_DIR / "sales_lead_qualification" / "test_sales_run_01.json").exists()
