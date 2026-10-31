"""Fact-Check Agent for Market and Competitor Research.

Audits the final synthesis report against local source documents,
validating source IDs, flagging uncorroborated assertions, and ensuring synthetic data labeling.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_fact_check_agent() -> Agent:
    return Agent(
        role="Fact-Checking & Citation Verification Specialist",
        goal=(
            "Audit all factual statements and metrics in the synthesis report against source doc_ids. "
            "Flag unsupported claims, verify citations, and guarantee the report is marked as synthetic data."
        ),
        backstory=(
            "You are a Head of Research Integrity and Fact-Checking. You ensure no unverified assertion, "
            "hallucination, or fabricated citation survives into the published report."
        ),
        llm=create_agent_llm("fact_check_agent"),
        verbose=False,
        allow_delegation=False,
    )
