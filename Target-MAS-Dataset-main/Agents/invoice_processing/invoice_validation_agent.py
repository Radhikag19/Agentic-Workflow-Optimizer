"""
Agent 2: Invoice Validation Agent
Validates invoice completeness and performs deterministic arithmetic calculations on invoice totals.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "invoice_validation_agent"

def deterministic_arithmetic_check(invoice_data: dict) -> dict:
    """Performs deterministic Python arithmetic verification on invoice numbers."""
    lines = invoice_data.get("line_items", [])
    calc_subtotal = 0.0
    line_discrepancies = []
    
    for idx, item in enumerate(lines, 1):
        qty = float(item.get("quantity", 0))
        price = float(item.get("unit_price", 0))
        stated_tot = float(item.get("line_total", 0))
        expected_line_tot = round(qty * price, 2)
        calc_subtotal += expected_line_tot
        if abs(expected_line_tot - stated_tot) > 0.02:
            line_discrepancies.append({
                "line": idx,
                "description": item.get("description"),
                "expected": expected_line_tot,
                "stated": stated_tot,
                "diff": round(stated_tot - expected_line_tot, 2)
            })
            
    calc_subtotal = round(calc_subtotal, 2)
    stated_tax = float(invoice_data.get("tax", 0))
    stated_discount = float(invoice_data.get("discount", 0))
    stated_total = float(invoice_data.get("total_amount", 0))
    calc_total = round(calc_subtotal + stated_tax - stated_discount, 2)
    
    total_diff = round(stated_total - calc_total, 2)
    is_arithmetic_valid = (abs(total_diff) <= 0.05) and len(line_discrepancies) == 0

    return {
        "is_arithmetic_valid": is_arithmetic_valid,
        "calculated_subtotal": calc_subtotal,
        "calculated_total": calc_total,
        "stated_total": stated_total,
        "discrepancy_amount": total_diff,
        "line_discrepancies": line_discrepancies
    }

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Invoice Validation Auditor",
        goal="Verify invoice field integrity, validate calculations against deterministic math checks, and document discrepancies.",
        backstory="A meticulous compliance auditor specializing in accounts payable fraud and mathematical discrepancy detection.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, invoice_data: dict) -> Task:
    math_audit = deterministic_arithmetic_check(invoice_data)
    description = (
        f"Perform structured validation of this invoice.\n"
        f"Invoice Data:\n{json.dumps(invoice_data, indent=2)}\n\n"
        f"Verified Deterministic Math Audit from Python:\n{json.dumps(math_audit, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Check if all required fields are present (supplier, dates, line items, totals).\n"
        f"2. Incorporate the deterministic arithmetic check results.\n"
        f"3. Note whether the invoice has valid arithmetic or critical math errors.\n"
        f"Return strictly valid JSON with keys: 'validation_status' ('PASSED' or 'FAILED'), 'math_audit', and 'validation_errors'."
    )
    expected_output = "Strictly valid JSON containing validation_status, math_audit, and validation_errors."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
