"""Symptom Agent for IT Incident Triage.

Correlates incident symptoms with historical incident logs and telemetry,
identifying recurrent patterns, matching past failures, and supporting record IDs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from crewai import Agent
from common.settings import create_agent_llm

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "incident_triage"


def query_historical_evidence(service: str, category: str, limit: int = 4) -> Dict[str, List[Dict[str, Any]]]:
    """Deterministic lookup for relevant historical incidents and logs."""
    inc_path = DATA_DIR / "historical_incidents.json"
    log_path = DATA_DIR / "log_snippets.json"

    matched_incidents = []
    if inc_path.exists():
        with open(inc_path, "r", encoding="utf-8") as f:
            all_inc = json.load(f).get("incidents", [])
        matched_incidents = [
            inc for inc in all_inc
            if inc.get("service") == service or inc.get("category") == category
        ][:limit]

    matched_logs = []
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            all_logs = json.load(f).get("logs", [])
        matched_logs = [
            lg for lg in all_logs
            if lg.get("service") == service or lg.get("category") == category
        ][:limit]

    return {"incidents": matched_incidents, "logs": matched_logs}


def create_symptom_agent() -> Agent:
    return Agent(
        role="Diagnostic & Symptom Correlation Specialist",
        goal=(
            "Correlate active incident symptoms with synthetic historical incidents and log records. "
            "Identify recurring patterns and return matching historical IDs and evidence."
        ),
        backstory=(
            "You are a Root Cause Diagnostics Specialist who performs deep telemetry correlation "
            "across past outages to spot systemic regressions and recurring anomalies."
        ),
        llm=create_agent_llm("symptom_agent"),
        verbose=False,
        allow_delegation=False,
    )
