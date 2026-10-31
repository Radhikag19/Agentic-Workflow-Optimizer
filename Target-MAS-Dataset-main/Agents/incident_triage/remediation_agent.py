"""Remediation Agent for IT Incident Triage.

Recommends safe, documented, reversible remediation procedures
derived from retrieved runbooks and correlated incident evidence.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_remediation_agent() -> Agent:
    return Agent(
        role="Remediation Planning Engineer",
        goal=(
            "Recommend structured, reversible, and safe troubleshooting actions based on retrieved "
            "runbooks and incident evidence. Never execute commands or connect to real infrastructure."
        ),
        backstory=(
            "You are a Senior Infrastructure Automation Engineer focused on safe system recovery. "
            "You specify explicit rollback procedures and safety prerequisites for every action."
        ),
        llm=create_agent_llm("remediation_agent"),
        verbose=False,
        allow_delegation=False,
    )
