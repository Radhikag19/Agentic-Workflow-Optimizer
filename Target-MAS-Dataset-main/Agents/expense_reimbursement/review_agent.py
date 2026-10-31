"""Review Agent for Employee Expense Reimbursement.

Synthesizes findings from all 7 validation agents and issues a final
recommendation (ELIGIBLE, INELIGIBLE, NEEDS_REVIEW). Never executes actual payment.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_expense_review_agent() -> Agent:
    return Agent(
        role="Expense Review & Adjudication Specialist",
        goal=(
            "Synthesize outputs from receipt extraction, normalization, categorization, amount check, "
            "policy verification, duplicate check, and exception audit. Deliver final adjudication "
            "(ELIGIBLE, INELIGIBLE, NEEDS_REVIEW) with citations and justifications. Never disburse funds."
        ),
        backstory=(
            "You are the Head of Accounts Payable & Disbursements. You make the final compliance "
            "determination on expense claims, balancing fair employee reimbursement with strict governance."
        ),
        llm=create_agent_llm("expense_review_agent"),
        verbose=False,
        allow_delegation=False,
    )
