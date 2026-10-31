"""Policy Agent for Employee Expense Reimbursement.

Retrieves corporate reimbursement policies, tests category limits,
per diem allowances, and itemization mandates.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from crewai import Agent
from common.settings import create_agent_llm

POLICIES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "expense_reimbursement" / "policies.json"


def query_policies_for_category(category: str) -> List[Dict[str, Any]]:
    """Retrieve applicable corporate policies for the given expense category."""
    if not POLICIES_PATH.exists():
        return []
    with open(POLICIES_PATH, "r", encoding="utf-8") as f:
        policies = json.load(f).get("policies", [])
    
    # Return matching policies or all if category is broad
    matches = [p for p in policies if p.get("category", "").lower() in category.lower() or category.lower() in p.get("category", "").lower()]
    return matches if matches else policies[:2]


def create_policy_agent() -> Agent:
    return Agent(
        role="Reimbursement Policy Compliance Specialist",
        goal=(
            "Apply corporate reimbursement policies against the claim details. Check caps, "
            "receipt requirements, and advance approval rules. Return policy IDs cited."
        ),
        backstory=(
            "You are a Corporate Compliance Officer with encyclopedic knowledge of company "
            "travel and expense policies, spending limits, and audit standards."
        ),
        llm=create_agent_llm("policy_agent"),
        verbose=False,
        allow_delegation=False,
    )
