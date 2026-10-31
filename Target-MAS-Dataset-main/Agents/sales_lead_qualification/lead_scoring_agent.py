"""
Agent 4: Lead Scoring Agent (Sales Lead Qualification)
Performs deterministic multi-signal numeric scoring for inbound leads.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "lead_scoring_agent"

def deterministic_lead_score(lead_data: dict, company_data: dict, events: list) -> dict:
    """Calculates deterministic points (0-100) based on predefined weights."""
    score = 0
    components = {}

    # 1. Industry Fit (25 pts max)
    target_ind = ["Financial Services", "Healthcare & Life Sciences", "Software & Technology", "Supply Chain & Logistics"]
    ind = company_data.get("industry", "")
    if ind in target_ind:
        ind_score = 25
    elif ind in ["Retail & Food Services", "Retail"]:
        ind_score = 0
    else:
        ind_score = 10
    score += ind_score
    components["industry_fit"] = ind_score

    # 2. Company Size (25 pts max)
    emps = int(company_data.get("employee_count", 0))
    if emps >= 500:
        size_score = 25
    elif emps >= 100:
        size_score = 18
    elif emps >= 20:
        size_score = 10
    else:
        size_score = 0
    score += size_score
    components["company_size_fit"] = size_score

    # 3. Role Authority (20 pts max)
    title = lead_data.get("job_title", "").lower()
    if any(k in title for k in ["cto", "vp", "chief", "head of", "director"]):
        role_score = 20
    elif any(k in title for k in ["manager", "lead", "architect"]):
        role_score = 12
    elif any(k in title for k in ["owner"]):
        role_score = 8 if emps >= 20 else 2
    else:
        role_score = 0
    score += role_score
    components["role_authority"] = role_score

    # 4. Intent Signals (20 pts max)
    event_types = [e.get("type", "") for e in events]
    if any("demo_request" in t or "executive_briefing" in t for t in event_types):
        intent_score = 20
    elif any("pricing" in t or "whitepaper" in t for t in event_types):
        intent_score = 12
    elif len(event_types) > 0:
        intent_score = 5
    else:
        intent_score = 0
    score += intent_score
    components["intent_signals"] = intent_score

    # 5. Budget & Timeline (10 pts max)
    budget = float(lead_data.get("stated_budget", 0))
    timeline = lead_data.get("purchase_timeline", "").lower()
    budget_score = 0
    if budget >= 50000.0:
        budget_score += 5
    elif budget >= 15000.0:
        budget_score += 2
    if any(k in timeline for k in ["immediate", "3 months", "q4"]):
        budget_score += 5
    elif "6 months" in timeline:
        budget_score += 3
    score += budget_score
    components["budget_timeline"] = budget_score

    tier = "DISQUALIFIED"
    if score >= 75:
        tier = "SALES_QUALIFIED_LEAD"
    elif score >= 45:
        tier = "MARKETING_NURTURE"

    return {
        "total_score": score,
        "qualification_tier": tier,
        "component_breakdown": components
    }

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Lead Qualification Scoring Specialist",
        goal="Apply mathematical scoring models to evaluate and grade inbound prospect readiness.",
        backstory="A RevOps scoring architect quantifying inbound lead potential using rigorous rubrics.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, lead_data: dict, company_data: dict, events: list) -> Task:
    scoring_result = deterministic_lead_score(lead_data, company_data, events)
    description = (
        f"Review and formalize the lead scoring breakdown.\n"
        f"Deterministic Scoring Model Results from Python:\n{json.dumps(scoring_result, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Explain the score calculation and tier classification ('SALES_QUALIFIED_LEAD', 'MARKETING_NURTURE', 'DISQUALIFIED').\n"
        f"2. Identify the highest-scoring and lowest-scoring dimensions.\n"
        f"Return strictly valid JSON with keys: 'total_score', 'qualification_tier', 'component_scores', and 'scoring_rationale'."
    )
    expected_output = "Strictly valid JSON containing total_score, qualification_tier, component_scores, and scoring_rationale."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
