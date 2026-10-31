"""
Tests verifying deterministic operations across all 4 workflows.
No LLM calls are made in these unit tests.
"""

import pytest

# Invoice processing deterministic functions
from Agents.invoice_processing.invoice_validation_agent import deterministic_arithmetic_check
from Agents.invoice_processing.po_matching_agent import deterministic_po_match
from Agents.invoice_processing.policy_compliance_agent import deterministic_policy_check
from Agents.invoice_processing.duplicate_detection_agent import deterministic_duplicate_check

# Customer support deterministic functions
from Agents.customer_support.ticket_classification_agent import deterministic_classification_hints
from Agents.customer_support.policy_retrieval_agent import deterministic_policy_retrieval

# Procurement deterministic functions
from Agents.procurement_supplier_selection.cost_comparison_agent import deterministic_cost_ranking
from Agents.procurement_supplier_selection.delivery_analysis_agent import deterministic_delivery_check
from Agents.procurement_supplier_selection.weighted_evaluation_agent import deterministic_weighted_scoring

# Sales lead qualification deterministic functions
from Agents.sales_lead_qualification.lead_fit_agent import deterministic_icp_check
from Agents.sales_lead_qualification.lead_scoring_agent import deterministic_lead_score
from Agents.sales_lead_qualification.next_action_agent import deterministic_action_rule


# --- 1. Invoice Processing Tests ---
def test_invoice_arithmetic_check():
    valid_inv = {
        "line_items": [
            {"description": "Item A", "quantity": 2, "unit_price": 50.0, "line_total": 100.0},
            {"description": "Item B", "quantity": 1, "unit_price": 40.0, "line_total": 40.0}
        ],
        "subtotal": 140.0,
        "tax": 11.20,
        "discount": 0.0,
        "total_amount": 151.20
    }
    res = deterministic_arithmetic_check(valid_inv)
    assert res["is_arithmetic_valid"] is True
    assert res["calculated_total"] == 151.20

    # Arithmetic mismatch case
    invalid_inv = dict(valid_inv, total_amount=200.0)
    res_bad = deterministic_arithmetic_check(invalid_inv)
    assert res_bad["is_arithmetic_valid"] is False
    assert abs(res_bad["discrepancy_amount"]) > 0.05


def test_po_matching_logic():
    inv_no_po = {"po_reference": ""}
    assert deterministic_po_match(inv_no_po)["match_status"] == "MISSING_PO_REFERENCE"

    inv_fake_po = {"po_reference": "PO-9999-NONEXISTENT"}
    assert deterministic_po_match(inv_fake_po)["match_status"] == "PO_NOT_FOUND"


def test_policy_compliance_thresholds():
    # Invoice > $1,000 without PO must trigger POL-AP-001
    inv_violating = {"total_amount": 1500.0, "po_reference": "", "supplier_id": "SUP-101"}
    res = deterministic_policy_check(inv_violating)
    assert res["is_compliant"] is False
    assert any(v["policy_id"] == "POL-AP-001" for v in res["violations"])


# --- 2. Customer Support Tests ---
def test_customer_support_classification_and_retrieval():
    ticket = {"subject": "Need password reset", "description": "Cannot login, locked out"}
    hints = deterministic_classification_hints(ticket)
    assert hints["suggested_category"] == "account_access"
    assert hints["suggested_complexity"] == "simple"

    retrieval = deterministic_policy_retrieval("account_access")
    assert retrieval["has_coverage"] is True


# --- 3. Procurement Tests ---
def test_procurement_cost_and_weighted_scoring():
    quotes = [
        {"supplier_id": "S1", "supplier_name": "Sup1", "item_total": 1000.0, "shipping_cost": 50.0, "tax_amount": 70.0},
        {"supplier_id": "S2", "supplier_name": "Sup2", "item_total": 1200.0, "shipping_cost": 30.0, "tax_amount": 80.0}
    ]
    ranked = deterministic_cost_ranking(quotes)
    assert ranked[0]["supplier_id"] == "S1"
    assert ranked[0]["total_landed_cost"] == 1120.0
    assert ranked[1]["supplier_id"] == "S2"

    deliv_calc = [{"supplier_id": "S1", "buffer_days": 5}, {"supplier_id": "S2", "buffer_days": -2}]
    risk_data = {"S1": {"delivery_failure_rate": 0.01, "quality_defect_rate": 0.005}, "S2": {"delivery_failure_rate": 0.05, "quality_defect_rate": 0.02}}
    profiles = {"S1": {"iso_certified": True}, "S2": {"iso_certified": False}}

    scores = deterministic_weighted_scoring(ranked, deliv_calc, risk_data, profiles)
    assert len(scores) == 2
    assert scores[0]["supplier_id"] == "S1"
    assert scores[0]["total_weighted_score"] > scores[1]["total_weighted_score"]


# --- 4. Sales Lead Tests ---
def test_sales_lead_icp_and_scoring():
    # Enterprise fit
    lead = {"job_title": "VP of Engineering", "stated_budget": 80000.0, "purchase_timeline": "Within 3 months"}
    comp = {"industry": "Software & Technology", "employee_count": 800}
    events = [{"type": "demo_request"}]

    fit = deterministic_icp_check(lead, comp)
    assert fit["is_icp_match"] is True

    score = deterministic_lead_score(lead, comp, events)
    assert score["total_score"] >= 75
    assert score["qualification_tier"] == "SALES_QUALIFIED_LEAD"
    assert deterministic_action_rule(score["qualification_tier"])["action"] == "schedule_executive_demo"

    # Micro bodega - clearly unsuitable
    bad_lead = {"job_title": "Cashier", "stated_budget": 50.0, "purchase_timeline": "None"}
    bad_comp = {"industry": "Retail & Food Services", "employee_count": 2}
    bad_fit = deterministic_icp_check(bad_lead, bad_comp)
    assert bad_fit["fit_tier"] == "DISQUALIFIED"
    bad_score = deterministic_lead_score(bad_lead, bad_comp, [])
    assert bad_score["qualification_tier"] == "DISQUALIFIED"
    assert deterministic_action_rule(bad_score["qualification_tier"])["action"] == "disqualify_lead"


# --- 5. IT Incident Triage Tests ---
from Agents.incident_triage.severity_agent import calculate_deterministic_severity

def test_incident_severity_matrix():
    # Tier-1 and users >= 5000 -> SEV-1
    res1 = calculate_deterministic_severity(impact_users=6000, service_tier="Tier-1", outage_minutes=30)
    assert res1["recommended_severity"] == "SEV-1"

    # Tier-1 and users >= 500 -> SEV-2
    res2 = calculate_deterministic_severity(impact_users=800, service_tier="Tier-1", outage_minutes=15)
    assert res2["recommended_severity"] == "SEV-2"

    # Outage >= 120 minutes -> SEV-1 regardless of initial count
    res3 = calculate_deterministic_severity(impact_users=50, service_tier="Tier-3", outage_minutes=150)
    assert res3["recommended_severity"] == "SEV-1"

    # Minor users and short duration -> SEV-4
    res4 = calculate_deterministic_severity(impact_users=25, service_tier="Tier-3", outage_minutes=5)
    assert res4["recommended_severity"] == "SEV-4"


# --- 6. Expense Reimbursement Tests ---
from Agents.expense_reimbursement.amount_agent import calculate_deterministic_arithmetic
from Agents.expense_reimbursement.duplicate_agent import query_candidate_duplicates
from Agents.expense_reimbursement.policy_agent import query_policies_for_category

def test_expense_arithmetic_checks():
    # Valid sum
    v1 = calculate_deterministic_arithmetic(subtotal=100.00, tax=8.88, total=108.88)
    assert v1["is_arithmetically_valid"] is True
    assert v1["discrepancy"] == 0.0

    # Arithmetic mismatch
    v2 = calculate_deterministic_arithmetic(subtotal=100.00, tax=8.88, total=125.00)
    assert v2["is_arithmetically_valid"] is False
    assert v2["discrepancy"] == 16.12

    # Negative value
    v3 = calculate_deterministic_arithmetic(subtotal=-50.00, tax=5.00, total=-45.00)
    assert v3["has_negative_values"] is True


def test_expense_policy_and_duplicates():
    meals_policies = query_policies_for_category("Meals & Entertainment")
    assert len(meals_policies) > 0
    assert any("POL-EXP" in p["policy_id"] for p in meals_policies)

    dups = query_candidate_duplicates("EMP-NONEXISTENT", "Nowhere Cafe", 99999.99)
    assert isinstance(dups, list)

