"""Comparison Agent for Market and Competitor Research.

Constructs structured competitive comparison matrices across architecture,
pricing, performance, and operational security dimensions.
"""

from crewai import Agent
from common.settings import create_agent_llm


def create_comparison_agent() -> Agent:
    return Agent(
        role="Competitor Comparison & Matrix Specialist",
        goal=(
            "Construct a multi-dimensional comparison matrix contrasting all 4 competitors. "
            "Clearly distinguish measured factual specifications from qualitative assessments. Cite doc_ids."
        ),
        backstory=(
            "You are a Competitive Strategy Director who synthesizes fragmented technical data "
            "into clear, actionable side-by-side matrices for executive evaluation."
        ),
        llm=create_agent_llm("comparison_agent"),
        verbose=False,
        allow_delegation=False,
    )
