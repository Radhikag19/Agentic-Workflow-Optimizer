"""Classification Agent for IT Incident Triage.

Categorizes the technical failure domain and determines incident complexity
to support future dynamic optimizer routing.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_classification_agent() -> Agent:
    return Agent(
        role="Incident Classification Engineer",
        goal=(
            "Categorize the incident into a precise failure domain (network_failure, "
            "application_error, authentication_issue, database_problem, storage_exhaustion) "
            "and assess complexity level (Simple, Moderate, Complex) for dynamic routing."
        ),
        backstory=(
            "You are a Senior Systems Architect who excels at recognizing system failure topologies, "
            "root failure domains, and whether an incident is standard or multi-service complex."
        ),
        llm=create_agent_llm("classification_agent"),
        verbose=False,
        allow_delegation=False,
    )
