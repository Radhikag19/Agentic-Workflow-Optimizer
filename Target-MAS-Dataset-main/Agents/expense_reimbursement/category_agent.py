"""Category Agent for Employee Expense Reimbursement.

Classifies the expense into the appropriate corporate accounting category
and justifies the assignment with merchant and itemized line evidence.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_category_agent() -> Agent:
    return Agent(
        role="Expense Categorization Specialist",
        goal=(
            "Categorize the claim into Travel, Meals & Entertainment, Accommodation, Transport, "
            "Office Supplies, or Software & Cloud. Cite supporting merchant and line item evidence."
        ),
        backstory=(
            "You are a Corporate Controller who maps receipts to General Ledger accounting "
            "categories based on merchant type, business purpose, and line item context."
        ),
        llm=create_agent_llm("category_agent"),
        verbose=False,
        allow_delegation=False,
    )
