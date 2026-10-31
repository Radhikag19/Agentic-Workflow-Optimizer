"""Exception Agent for Employee Expense Reimbursement.

Independently inspects claims for anomalies, policy infractions, and red flags.
Intentionally overlaps with Amount and Policy agents to serve as an AWO pruning/consolidation candidate.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_exception_agent() -> Agent:
    return Agent(
        role="Expense Exception & Audit Specialist",
        goal=(
            "Conduct an independent red-team audit of the claim: check for policy violations, "
            "missing receipt documentation, excessive sums, or unapproved perks. Generate structured warnings."
        ),
        backstory=(
            "You are an Internal Affairs Financial Auditor. You operate with healthy skepticism, "
            "spotting irregular spending patterns, missing documentation, and policy circumvention."
        ),
        llm=create_agent_llm("exception_agent"),
        verbose=False,
        allow_delegation=False,
    )
