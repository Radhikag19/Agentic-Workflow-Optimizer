"""
Agent 5: Duplicate Detection Agent
Compares current invoice against historical invoice records to prevent duplicate payments.
"""

import json
from pathlib import Path
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent
from common.settings import DATA_DIR

AGENT_ID = "duplicate_detection_agent"

def deterministic_duplicate_check(invoice_data: dict) -> dict:
    """Scans historical invoices for exact and probable duplicate submissions."""
    hist_file = DATA_DIR / "invoice_processing" / "historical_invoices.json"
    if not hist_file.exists():
        return {"is_duplicate": False, "exact_matches": [], "potential_matches": []}

    with open(hist_file, "r", encoding="utf-8") as f:
        hist_records = json.load(f)

    inv_num = invoice_data.get("invoice_number", "").strip()
    sup_id = invoice_data.get("supplier_id", "").strip()
    po_ref = invoice_data.get("po_reference", "").strip()
    total_amt = float(invoice_data.get("total_amount", 0))

    exact_matches = []
    potential_matches = []

    for h in hist_records:
        h_num = h.get("invoice_number", "").strip()
        h_sup = h.get("supplier_id", "").strip()
        h_po = h.get("po_reference", "").strip()
        h_amt = float(h.get("total_amount", 0))

        # Exact match rule: same invoice number OR same supplier + PO + amount
        if h_num == inv_num or (sup_id and h_sup == sup_id and po_ref and h_po == po_ref and abs(h_amt - total_amt) < 0.01):
            exact_matches.append({
                "historical_invoice_id": h.get("invoice_id"),
                "invoice_number": h_num,
                "supplier_id": h_sup,
                "amount": h_amt,
                "reason": "Exact invoice number or identical (supplier, PO, amount) duplicate."
            })
        elif sup_id and h_sup == sup_id and abs(h_amt - total_amt) < 0.01:
            potential_matches.append({
                "historical_invoice_id": h.get("invoice_id"),
                "invoice_number": h_num,
                "supplier_id": h_sup,
                "amount": h_amt,
                "reason": "Same supplier and exact same dollar amount."
            })

    return {
        "is_duplicate": len(exact_matches) > 0,
        "is_potential_duplicate": len(potential_matches) > 0,
        "exact_matches": exact_matches,
        "potential_matches": potential_matches
    }

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Fraud & Duplicate Payment Prevention Analyst",
        goal="Screen invoices against historical payments to stop duplicate billing and overpayments.",
        backstory="An AP forensic specialist focused on identifying redundant billing and matching invoice histories.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, invoice_data: dict) -> Task:
    dup_audit = deterministic_duplicate_check(invoice_data)
    description = (
        f"Perform duplicate screening on this invoice.\n"
        f"Current Invoice:\n{json.dumps(invoice_data, indent=2)}\n\n"
        f"Deterministic Historical Match Scan:\n{json.dumps(dup_audit, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Evaluate whether this invoice is an exact duplicate, potential duplicate, or original billing.\n"
        f"2. Cite matching historical invoice IDs and field comparisons.\n"
        f"Return strictly valid JSON with keys: 'duplicate_status' ('NONE', 'POTENTIAL_DUPLICATE', 'EXACT_DUPLICATE'), 'matching_records', and 'risk_assessment'."
    )
    expected_output = "Strictly valid JSON containing duplicate_status, matching_records, and risk_assessment."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
