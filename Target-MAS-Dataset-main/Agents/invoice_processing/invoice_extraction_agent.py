"""
Agent 1: Invoice Extraction Agent
Extracts structured invoice fields from synthetic invoice input.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "invoice_extraction_agent"

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Senior Invoice Extraction Specialist",
        goal="Accurately extract all structured invoice fields (supplier, PO, line items, dates, totals) without hallucinating missing data.",
        backstory="An expert document processor specialized in parsing accounts payable documents into structured formats.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, invoice_data: dict) -> Task:
    description = (
        f"Analyze the following synthetic invoice and extract its structured fields.\n"
        f"Raw Invoice Input:\n{json.dumps(invoice_data, indent=2)}\n\n"
        f"Tasks:\n"
        f"1. Extract: invoice_number, supplier_id, supplier_name, po_reference, invoice_date, due_date, currency.\n"
        f"2. Extract line_items with description, quantity, unit_price, line_total.\n"
        f"3. Extract subtotal, tax, discount, and total_amount.\n"
        f"4. Identify any missing, ambiguous, or inconsistent fields.\n"
        f"Output strictly valid JSON with keys: 'extracted_invoice' and 'extraction_flags'."
    )
    expected_output = "Strictly valid JSON containing extracted_invoice and extraction_flags."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
