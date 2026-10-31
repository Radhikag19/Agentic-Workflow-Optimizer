"""
Agent 3: Purchase Order Matching Agent
Performs 3-way matching between synthetic invoices and purchase orders.
"""

import json
from pathlib import Path
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent
from common.settings import DATA_DIR

AGENT_ID = "po_matching_agent"

def deterministic_po_match(invoice_data: dict) -> dict:
    """Matches invoice against local synthetic purchase orders."""
    po_ref = invoice_data.get("po_reference", "").strip()
    po_file = DATA_DIR / "invoice_processing" / "purchase_orders.json"
    
    if not po_ref:
        return {
            "match_status": "MISSING_PO_REFERENCE",
            "matched_po_id": None,
            "discrepancies": ["No purchase order reference specified in invoice."]
        }
        
    matched_po = None
    if po_file.exists():
        with open(po_file, "r", encoding="utf-8") as f:
            pos = json.load(f)
            for po in pos:
                if po.get("po_id") == po_ref:
                    matched_po = po
                    break

    if not matched_po:
        return {
            "match_status": "PO_NOT_FOUND",
            "matched_po_id": po_ref,
            "discrepancies": [f"Purchase Order '{po_ref}' not found in registry."]
        }

    # Compare fields deterministically
    discrepancies = []
    if matched_po.get("supplier_id") != invoice_data.get("supplier_id"):
        discrepancies.append(
            f"Supplier mismatch: Invoice supplier '{invoice_data.get('supplier_id')}' != PO supplier '{matched_po.get('supplier_id')}'"
        )
    
    # Check total amount variance
    po_tot = float(matched_po.get("total_amount", 0))
    inv_tot = float(invoice_data.get("total_amount", 0))
    if abs(po_tot - inv_tot) > 1.0:
        discrepancies.append(
            f"Total amount mismatch: Invoice total ${inv_tot:.2f} differs from PO total ${po_tot:.2f}"
        )

    # Check line item quantities
    po_lines = {l.get("description"): l for l in matched_po.get("lines", [])}
    for inv_line in invoice_data.get("line_items", []):
        desc = inv_line.get("description")
        if desc in po_lines:
            po_qty = po_lines[desc].get("quantity", 0)
            inv_qty = inv_line.get("quantity", 0)
            if inv_qty != po_qty:
                discrepancies.append(
                    f"Line quantity mismatch on '{desc}': Invoiced {inv_qty} units vs PO {po_qty} units"
                )

    match_status = "EXACT_MATCH" if not discrepancies else "DISCREPANCY_DETECTED"
    return {
        "match_status": match_status,
        "matched_po_id": po_ref,
        "matched_po_data": matched_po,
        "discrepancies": discrepancies
    }

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Purchase Order Reconciliation Specialist",
        goal="Match invoice line items, pricing, quantities, and terms against authorized purchase orders.",
        backstory="An accounts payable specialist ensuring invoices only get paid if backed by approved purchase orders.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, invoice_data: dict) -> Task:
    po_match_result = deterministic_po_match(invoice_data)
    description = (
        f"Reconcile this invoice against its corresponding purchase order.\n"
        f"Invoice Data:\n{json.dumps(invoice_data, indent=2)}\n\n"
        f"Deterministic PO Matching Results:\n{json.dumps(po_match_result, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Evaluate whether the invoice accurately matches the PO.\n"
        f"2. Detail all quantity, price, or supplier discrepancies.\n"
        f"Return strictly valid JSON with keys: 'po_match_status', 'matched_po_id', 'discrepancies', and 'matching_confidence'."
    )
    expected_output = "Strictly valid JSON containing po_match_status, matched_po_id, discrepancies, and matching_confidence."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
