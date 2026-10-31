"""Normalization Agent for Employee Expense Reimbursement.

Standardizes dates into ISO-8601, amounts to two-decimal floats,
currencies into ISO codes, and validates employee identifier syntax.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_normalize_agent() -> Agent:
    return Agent(
        role="Expense Normalization Specialist",
        goal=(
            "Standardize extracted claim dates, numeric amounts, currency codes, and employee IDs. "
            "Flag inconsistencies between the claim submission and receipt evidence."
        ),
        backstory=(
            "You are a Financial Data Integrity Engineer who ensures all accounting records "
            "conform to strict ISO standards, uniform data schemas, and validated identifiers."
        ),
        llm=create_agent_llm("normalize_agent"),
        verbose=False,
        allow_delegation=False,
    )
