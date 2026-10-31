"""Intake Agent for IT Incident Triage.

Normalizes raw incident tickets, extracts structured system telemetry,
affected services, user counts, and flags missing information.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_intake_agent() -> Agent:
    return Agent(
        role="Incident Intake Specialist",
        goal=(
            "Normalize raw incident ticket text, extract affected services, symptoms, "
            "timestamps, and impacted users, and flag any missing critical fields without inventing facts."
        ),
        backstory=(
            "You are a Tier-1 SRE Intake Specialist. You parse messy, emergency incident tickets into "
            "crisp, unambiguous facts and telemetry without speculating or adding unsupported assumptions."
        ),
        llm=create_agent_llm("intake_agent"),
        verbose=False,
        allow_delegation=False,
    )
