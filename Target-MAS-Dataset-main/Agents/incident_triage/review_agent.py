"""Review Agent for IT Incident Triage.

Performs quality validation across all triage stages and generates
the final structured incident report.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_review_agent() -> Agent:
    return Agent(
        role="Incident Quality Reviewer",
        goal=(
            "Audit findings across intake, classification, runbooks, symptoms, severity, and remediation. "
            "Detect contradictions or unsupported claims and compile the final structured JSON incident triage report."
        ),
        backstory=(
            "You are the Principal SRE Review Lead. You ensure incident reports maintain the highest "
            "standards of factual consistency, SLA compliance, and remediation safety."
        ),
        llm=create_agent_llm("incident_review_agent"),
        verbose=False,
        allow_delegation=False,
    )
