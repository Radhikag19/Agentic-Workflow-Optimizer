"""Duplicate Agent for Employee Expense Reimbursement.

Detects potential duplicate claims against historical submissions,
differentiating between exact matches and plausible coincidences.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from crewai import Agent
from common.settings import create_agent_llm

HIST_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "expense_reimbursement" / "historical_submissions.json"


def query_candidate_duplicates(employee_id: str, merchant: str, amount: float) -> List[Dict[str, Any]]:
    """Deterministic lookup for potential duplicate claims."""
    if not HIST_PATH.exists():
        return []
    with open(HIST_PATH, "r", encoding="utf-8") as f:
        submissions = json.load(f).get("historical_submissions", [])

    candidates = []
    for sub in submissions:
        # Match if same employee or same merchant with identical amount
        same_emp = sub.get("employee_id") == employee_id
        same_merchant = sub.get("merchant", "").lower() == merchant.lower()
        same_amount = abs(float(sub.get("amount", 0.0)) - amount) < 0.05

        if (same_emp and same_amount) or (same_merchant and same_amount):
            candidates.append(sub)

    return candidates[:3]


def create_duplicate_agent() -> Agent:
    return Agent(
        role="Duplicate Claim Detection Specialist",
        goal=(
            "Cross-reference active claim against historical submissions. Distinguish "
            "between confirmed duplicate submissions and coincidentally similar expenses."
        ),
        backstory=(
            "You are a Fraud Prevention Analyst who scans transaction histories to prevent "
            "double-reimbursements and duplicate receipt submissions across accounting periods."
        ),
        llm=create_agent_llm("duplicate_agent"),
        verbose=False,
        allow_delegation=False,
    )
