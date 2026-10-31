"""
Agent 4: Policy Compliance Agent
Evaluates invoice against accounts payable policies and supplier eligibility.
"""

import json
from pathlib import Path
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent
from common.settings import DATA_DIR

AGENT_ID = "policy_compliance_agent"

def deterministic_policy_check(invoice_data: dict) -> dict:
    """Verifies invoice rules against local synthetic AP policies and supplier statuses."""
    policy_file = DATA_DIR / "invoice_processing" / "policies.json"
    supplier_file = DATA_DIR / "invoice_processing" / "suppliers.json"

    policies = []
    if policy_file.exists():
        with open(policy_file, "r", encoding="utf-8") as f:
            policies = json.load(f)

    suppliers = {}
    if supplier_file.exists():
        with open(supplier_file, "r", encoding="utf-8") as f:
            for s in json.load(f):
                suppliers[s.get("supplier_id")] = s

    violations = []
    applied_policy_ids = []
    total_amt = float(invoice_data.get("total_amount", 0))
    sup_id = invoice_data.get("supplier_id")
    sup_info = suppliers.get(sup_id, {})

    # Check POL-AP-001 (Three-way matching for > $1000)
    applied_policy_ids.append("POL-AP-001")
    if total_amt >= 1000.0 and not invoice_data.get("po_reference"):
        violations.append({
            "policy_id": "POL-AP-001",
            "rule": "Every invoice above $1,000 must strictly match a valid Purchase Order.",
            "severity": "high",
            "detail": f"Invoice amount is ${total_amt:.2f} but PO reference is missing."
        })

    # Check POL-AP-002 (Supplier active and valid tax ID)
    applied_policy_ids.append("POL-AP-002")
    if sup_info.get("status") == "suspended":
        violations.append({
            "policy_id": "POL-AP-002",
            "rule": "Invoices from suspended suppliers cannot be approved.",
            "severity": "critical",
            "detail": f"Supplier {sup_id} ({sup_info.get('name')}) is currently suspended."
        })

    # Check POL-AP-003 (High value approval threshold > $25k)
    applied_policy_ids.append("POL-AP-003")
    if total_amt > 25000.0:
        violations.append({
            "policy_id": "POL-AP-003",
            "rule": "Invoices totaling over $25,000 require executive financial review.",
            "severity": "medium",
            "detail": f"Total amount ${total_amt:.2f} exceeds high-value threshold ($25,000)."
        })

    return {
        "applied_policy_ids": applied_policy_ids,
        "violations": violations,
        "is_compliant": len(violations) == 0
    }

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Corporate Accounts Payable Compliance Officer",
        goal="Audit invoice against corporate purchasing policies, threshold limits, and vendor eligibility.",
        backstory="An uncompromising governance officer ensuring adherence to internal procurement policies and SOX controls.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, invoice_data: dict) -> Task:
    policy_audit = deterministic_policy_check(invoice_data)
    description = (
        f"Perform policy compliance review on this invoice.\n"
        f"Invoice Data:\n{json.dumps(invoice_data, indent=2)}\n\n"
        f"Deterministic Policy Audit:\n{json.dumps(policy_audit, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Evaluate policy violations and threshold conditions.\n"
        f"2. Explicitly cite each applied policy ID.\n"
        f"3. Classify whether infractions are fatal policy violations or flags requiring secondary sign-off.\n"
        f"Return strictly valid JSON with keys: 'is_compliant', 'applied_policy_ids', 'violations', and 'compliance_summary'."
    )
    expected_output = "Strictly valid JSON containing is_compliant, applied_policy_ids, violations, and compliance_summary."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
