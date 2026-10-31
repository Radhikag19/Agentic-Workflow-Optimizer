"""Feedback Agent for Market and Competitor Research.

Retrieves verified customer reviews, implementation feedback, and field observations.
Separates recorded quotes from interpretations and cites source doc_ids.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from crewai import Agent
from common.settings import create_agent_llm

DOC_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "market_research" / "source_documents.json"


def query_feedback_documents() -> List[Dict[str, Any]]:
    """Retrieve customer feedback and review documents."""
    if not DOC_PATH.exists():
        return []
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f).get("documents", [])
    return [d for d in docs if d.get("doc_type") == "customer_feedback"]


def create_feedback_agent() -> Agent:
    return Agent(
        role="Customer Feedback & Review Analyst",
        goal=(
            "Extract real-world customer feedback, reported production outages, praise, "
            "and usability complaints across competitors. Keep recorded quotes strictly separate from conclusions."
        ),
        backstory=(
            "You are a Voice of Customer (VoC) Market Researcher who analyzes verified user "
            "testimonials, G2 reviews, and community sentiment to assess customer satisfaction."
        ),
        llm=create_agent_llm("feedback_agent"),
        verbose=False,
        allow_delegation=False,
    )
