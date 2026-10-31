"""
Agent 2: Policy Retrieval Agent (Customer Support)
Retrieves relevant customer service policies, SLAs, and troubleshooting SOPs.
"""

import json
from pathlib import Path
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent
from common.settings import DATA_DIR

AGENT_ID = "policy_retrieval_agent"

def deterministic_policy_retrieval(category: str) -> dict:
    """Matches ticket category against local synthetic support policies and troubleshooting SOPs."""
    pol_file = DATA_DIR / "customer_support" / "policies.json"
    tsg_file = DATA_DIR / "customer_support" / "troubleshooting_guides.json"

    matched_policies = []
    if pol_file.exists():
        with open(pol_file, "r", encoding="utf-8") as f:
            for p in json.load(f):
                cat_lower = p.get("category", "").lower()
                if category.lower() in cat_lower or (category == "refund_request" and "refund" in cat_lower):
                    matched_policies.append(p)

    matched_guides = []
    if tsg_file.exists():
        with open(tsg_file, "r", encoding="utf-8") as f:
            for g in json.load(f):
                if g.get("issue_category") == category:
                    matched_guides.append(g)

    return {
        "matched_policies": matched_policies,
        "matched_troubleshooting_guides": matched_guides,
        "has_coverage": len(matched_policies) > 0 or len(matched_guides) > 0
    }

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Customer Policy & Knowledge Base Specialist",
        goal="Retrieve authoritative support guidelines, SLAs, and troubleshooting procedures for the ticket.",
        backstory="A knowledge manager providing certified procedures to support agents without guessing.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, ticket_data: dict, classification_results: dict) -> Task:
    category = classification_results.get("category", ticket_data.get("category", "general"))
    retrieval_data = deterministic_policy_retrieval(category)
    description = (
        f"Retrieve applicable policies and SOPs for this support ticket.\n"
        f"Ticket Data:\n{json.dumps(ticket_data, indent=2)}\n\n"
        f"Retrieved Policy & Knowledge Base Records:\n{json.dumps(retrieval_data, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Identify the primary governing policy ID and troubleshooting guide ID.\n"
        f"2. Detail eligibility conditions, SLAs, and restrictions.\n"
        f"3. Note explicitly if no relevant policy covers this specific issue.\n"
        f"Return strictly valid JSON with keys: 'applicable_policy_ids', 'applicable_guide_ids', 'policy_conditions', and 'retrieval_summary'."
    )
    expected_output = "Strictly valid JSON containing applicable_policy_ids, applicable_guide_ids, policy_conditions, and retrieval_summary."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
