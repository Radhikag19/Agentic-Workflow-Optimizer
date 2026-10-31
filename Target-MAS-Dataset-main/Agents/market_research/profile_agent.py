"""Profile Agent for Market and Competitor Research.

Retrieves company profiles, headquarters, funding, headcount,
and target customer profiles with stable source document IDs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from crewai import Agent
from common.settings import create_agent_llm

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "market_research"


def query_company_profiles() -> Dict[str, Any]:
    """Retrieve competitor metadata and corporate profile documents."""
    comp_path = DATA_DIR / "competitors.json"
    doc_path = DATA_DIR / "source_documents.json"
    competitors = []
    if comp_path.exists():
        with open(comp_path, "r", encoding="utf-8") as f:
            competitors = json.load(f).get("competitors", [])

    profile_docs = []
    if doc_path.exists():
        with open(doc_path, "r", encoding="utf-8") as f:
            all_docs = json.load(f).get("documents", [])
            profile_docs = [d for d in all_docs if d.get("doc_type") == "company_profile"]

    return {"competitors": competitors, "profile_documents": profile_docs}


def create_profile_agent() -> Agent:
    return Agent(
        role="Company Profiling Specialist",
        goal=(
            "Extract competitor corporate identities, funding stage, employee count, "
            "market positioning, and Ideal Customer Profile (ICP). Always cite stable source IDs."
        ),
        backstory=(
            "You are a Venture Capital Intelligence Analyst who creates crisp corporate "
            "dossiers, strictly referencing primary corporate source documents."
        ),
        llm=create_agent_llm("profile_agent"),
        verbose=False,
        allow_delegation=False,
    )
