"""Runbook Agent for IT Incident Triage.

Retrieves relevant operational runbooks from the local dataset
matching the classified incident domain and symptoms.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from crewai import Agent
from common.settings import create_agent_llm

RUNBOOKS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "incident_triage" / "runbooks.json"


def query_candidate_runbooks(category: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Deterministic retrieval helper to fetch candidate runbooks from local dataset."""
    if not RUNBOOKS_PATH.exists():
        return []
    with open(RUNBOOKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    runbooks = data.get("runbooks", [])
    matches = [rb for rb in runbooks if rb.get("category") == category]
    return matches[:limit] if matches else runbooks[:limit]


def create_runbook_agent() -> Agent:
    return Agent(
        role="Runbook Retrieval Specialist",
        goal=(
            "Examine candidate runbooks from the local knowledge base, evaluate symptom matches, "
            "and select the optimal runbook IDs with supporting evidence."
        ),
        backstory=(
            "You are an Operations Reliability Engineer who knows every standard operating "
            "procedure and selects the most relevant, proven runbook for active incidents."
        ),
        llm=create_agent_llm("runbook_agent"),
        verbose=False,
        allow_delegation=False,
    )
