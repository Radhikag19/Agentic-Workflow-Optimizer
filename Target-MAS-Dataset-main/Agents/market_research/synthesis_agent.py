"""Synthesis Agent for Market and Competitor Research.

Produces the comprehensive competitive research narrative, executive summary,
strategic trade-offs, and market positioning synthesis.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_synthesis_agent() -> Agent:
    return Agent(
        role="Research Synthesis & Narrative Specialist",
        goal=(
            "Synthesize the competitor comparison and evidence analysis into an executive research "
            "narrative. Highlight key trade-offs and recommendations without repetitive boilerplate."
        ),
        backstory=(
            "You are a Senior Industry Analyst known for writing concise, incisive market research "
            "reports that deliver executive-level clarity backed by empirical data."
        ),
        llm=create_agent_llm("synthesis_agent"),
        verbose=False,
        allow_delegation=False,
    )
