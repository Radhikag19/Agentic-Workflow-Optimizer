"""
Agent 1: Ticket Classification Agent (Customer Support)
Classifies support ticket category, urgency, complexity, and resolution type.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "ticket_classification_agent"

def deterministic_classification_hints(ticket_data: dict) -> dict:
    """Extracts deterministic classification signals from ticket fields."""
    subject = ticket_data.get("subject", "").lower()
    desc = ticket_data.get("description", "").lower()
    full_text = f"{subject} {desc}"

    category = "general"
    if any(k in full_text for k in ["refund", "money back", "credit"]):
        category = "refund_request"
    elif any(k in full_text for k in ["password", "login", "locked", "auth", "access"]):
        category = "account_access"
    elif any(k in full_text for k in ["invoice", "charge", "billed", "payment"]):
        category = "billing"
    elif any(k in full_text for k in ["ship", "delivery", "carrier", "order #"]):
        category = "delivery"
    elif any(k in full_text for k in ["500", "error", "api", "crash", "webhook"]):
        category = "technical_failure"

    is_simple = category in ["account_access", "billing"] and "intermittent" not in full_text
    apparent_complexity = "simple" if is_simple else "complex"

    return {
        "suggested_category": category,
        "suggested_complexity": apparent_complexity
    }

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Customer Support Triage Specialist",
        goal="Classify customer tickets into accurate operational categories, urgency tiers, and complexity levels.",
        backstory="An experienced support dispatcher who routes customer inquiries quickly and accurately.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, ticket_data: dict) -> Task:
    hints = deterministic_classification_hints(ticket_data)
    description = (
        f"Classify the following customer support ticket.\n"
        f"Ticket Data:\n{json.dumps(ticket_data, indent=2)}\n\n"
        f"Rule-Based Triage Signals:\n{json.dumps(hints, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Classify category (account_access, billing, payment, delivery, technical_failure, refund_request).\n"
        f"2. Assess urgency ('low', 'medium', 'high', 'critical') and customer impact.\n"
        f"3. Classify apparent complexity ('simple', 'moderate', 'complex').\n"
        f"4. Identify missing customer details or ambiguities.\n"
        f"Return strictly valid JSON with keys: 'category', 'urgency', 'complexity', 'customer_impact', and 'routing_recommendation'."
    )
    expected_output = "Strictly valid JSON containing category, urgency, complexity, customer_impact, and routing_recommendation."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
