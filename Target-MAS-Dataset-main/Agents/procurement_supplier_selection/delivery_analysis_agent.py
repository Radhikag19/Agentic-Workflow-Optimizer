"""
Agent 3: Delivery Analysis Agent (Procurement)
Assesses fulfillment timelines, lead times, and delivery schedule compliance.
"""

import json
from datetime import datetime
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "delivery_analysis_agent"

def deterministic_delivery_check(required_date_str: str, delivery_schedules: list) -> list:
    """Calculates deterministic date differentials and deadline adherence."""
    results = []
    req_dt = datetime.strptime(required_date_str, "%Y-%m-%d")
    for sched in delivery_schedules:
        promised_str = sched.get("promised_delivery_date")
        promised_dt = datetime.strptime(promised_str, "%Y-%m-%d")
        days_diff = (req_dt - promised_dt).days
        meets_deadline = days_diff >= 0
        results.append({
            "supplier_id": sched.get("supplier_id"),
            "lead_time_days": sched.get("lead_time_days"),
            "promised_date": promised_str,
            "required_date": required_date_str,
            "buffer_days": days_diff,
            "meets_deadline": meets_deadline
        })
    return results

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Supply Chain Fulfillment Analyst",
        goal="Evaluate supplier delivery schedules, shipping transit times, and milestone feasibility.",
        backstory="A logistics specialist focused on critical path analysis and preventing supply chain delays.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, required_date_str: str, delivery_schedules: list) -> Task:
    delivery_eval = deterministic_delivery_check(required_date_str, delivery_schedules)
    description = (
        f"Analyze delivery commitments and schedule risks for this procurement request.\n"
        f"Required Target Date: {required_date_str}\n"
        f"Deterministic Schedule Analysis from Python:\n{json.dumps(delivery_eval, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Highlight suppliers that meet vs. violate the delivery deadline.\n"
        f"2. Assess buffer margins and fulfillment risk.\n"
        f"Return strictly valid JSON with keys: 'delivery_evaluation', 'on_time_suppliers', 'late_risk_suppliers', and 'logistics_summary'."
    )
    expected_output = "Strictly valid JSON containing delivery_evaluation, on_time_suppliers, late_risk_suppliers, and logistics_summary."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
