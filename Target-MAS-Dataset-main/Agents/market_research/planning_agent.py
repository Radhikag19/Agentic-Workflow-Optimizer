"""Planning Agent for Market and Competitor Research.

Deconstructs the research question into structured inquiry dimensions,
identifies target competitors, and defines required evidence types.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_planning_agent() -> Agent:
    return Agent(
        role="Research Planning Specialist",
        goal=(
            "Deconstruct the user's market inquiry into key comparison dimensions "
            "(architecture, latency, pricing, SLA, customer sentiment). Identify target "
            "competitors and define the evidence matrix."
        ),
        backstory=(
            "You are a Strategic Market Intelligence Lead who designs structured, rigorous "
            "competitive research methodologies and investigative plans."
        ),
        llm=create_agent_llm("planning_agent"),
        verbose=False,
        allow_delegation=False,
    )
