"""Product Agent for Market and Competitor Research.

Retrieves technical specifications, features, architectural models,
and product differentiators with stable source IDs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from crewai import Agent
from common.settings import create_agent_llm

DOC_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "market_research" / "source_documents.json"


def query_product_documents() -> List[Dict[str, Any]]:
    """Retrieve product capabilities and benchmark source documents."""
    if not DOC_PATH.exists():
        return []
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f).get("documents", [])
    return [d for d in docs if d.get("doc_type") in ["product_capabilities", "benchmark_report", "sla_specification"]]


def create_product_agent() -> Agent:
    return Agent(
        role="Product Capabilities Analyst",
        goal=(
            "Extract technical features, platform throughput, integrations, SDK support, "
            "and architectural limitations across competitors. Attribute every claim to its source doc_id."
        ),
        backstory=(
            "You are a Principal Solutions Architect who dissects technical whitepapers, "
            "feature sheets, and system SLAs to understand real engineering capabilities."
        ),
        llm=create_agent_llm("product_agent"),
        verbose=False,
        allow_delegation=False,
    )
