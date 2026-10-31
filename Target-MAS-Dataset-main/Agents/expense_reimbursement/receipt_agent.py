"""Receipt Agent for Employee Expense Reimbursement.

Extracts structured merchant, transaction date, currency, subtotal,
tax, and line items from synthetic receipt text.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_receipt_agent() -> Agent:
    return Agent(
        role="Receipt Extraction Specialist",
        goal=(
            "Extract merchant name, transaction date, currency, subtotal, tax amount, "
            "total amount, and itemized lines from raw receipt text. Flag any missing fields."
        ),
        backstory=(
            "You are a Forensic Accounts Payable Auditor specializing in reading receipt structures, "
            "itemized charges, and payment metadata accurately without guessing."
        ),
        llm=create_agent_llm("receipt_agent"),
        verbose=False,
        allow_delegation=False,
    )
