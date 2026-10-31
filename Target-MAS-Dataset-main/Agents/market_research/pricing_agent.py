"""Pricing Agent for Market and Competitor Research.

Retrieves commercial pricing schedules, subscription tiers, overage clauses,
and contract minimums. Preserves exact currency units and dates.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from crewai import Agent
from common.settings import create_agent_llm

DOC_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "market_research" / "source_documents.json"


def query_pricing_documents() -> List[Dict[str, Any]]:
    """Retrieve commercial pricing and licensing source documents."""
    if not DOC_PATH.exists():
        return []
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f).get("documents", [])
    return [d for d in docs if d.get("doc_type") == "pricing_and_licensing"]


def create_pricing_agent() -> Agent:
    return Agent(
        role="Pricing & Commercial Terms Specialist",
        goal=(
            "Extract commercial pricing schedules, monthly tiers, overage fees, contract minimums, "
            "and billing units. Explicitly mark unstated pricing as UNKNOWN and cite source doc_ids."
        ),
        backstory=(
            "You are a SaaS Pricing & Licensing Strategist who scrutinizes Master Services Agreements, "
            "rate cards, and hidden overage terms to establish true total cost of ownership (TCO)."
        ),
        llm=create_agent_llm("pricing_agent"),
        verbose=False,
        allow_delegation=False,
    )
