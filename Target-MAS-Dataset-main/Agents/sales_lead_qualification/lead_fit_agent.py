"""
Agent 2: Lead Fit Agent (Sales Lead Qualification)
Assesses account and persona alignment against the synthetic Ideal Customer Profile (ICP).
"""

import json
from pathlib import Path
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent
from common.settings import DATA_DIR

AGENT_ID = "lead_fit_agent"

def deterministic_icp_check(lead_data: dict, company_data: dict) -> dict:
    """Evaluates rule-based ICP thresholds deterministically."""
    icp_file = DATA_DIR / "sales_lead_qualification" / "ideal_customer_profiles.json"
    icp = {
        "target_industries": ["Financial Services", "Healthcare & Life Sciences", "Software & Technology", "Supply Chain & Logistics"],
        "min_employees": 200,
        "disqualified_industries": ["Retail & Food Services", "Personal Services", "Independent Bodegas"]
    }
    if icp_file.exists():
        with open(icp_file, "r", encoding="utf-8") as f:
            icps = json.load(f)
            if icps:
                icp = icps[0]

    industry = company_data.get("industry", "")
    emp_count = int(company_data.get("employee_count", 0))

    if industry in icp.get("disqualified_industries", []) or emp_count < 10:
        return {
            "fit_tier": "DISQUALIFIED",
            "is_icp_match": False,
            "reason": f"Industry '{industry}' is in disqualified industries or company size ({emp_count} employees) is below micro-threshold."
        }

    if industry in icp.get("target_industries", []) and emp_count >= icp.get("min_employees", 200):
        return {
            "fit_tier": "STRONG_FIT",
            "is_icp_match": True,
            "reason": f"Matches target industry '{industry}' and enterprise headcount ({emp_count} >= {icp.get('min_employees')})."
        }

    return {
        "fit_tier": "MODERATE_FIT",
        "is_icp_match": True,
        "reason": f"Partial match: Industry '{industry}', employees: {emp_count}."
    }

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Ideal Customer Profile (ICP) Fit Evaluator",
        goal="Audit lead firmographics against ICP criteria to identify ideal buyers and flag unsuitable prospects.",
        backstory="A revenue operations specialist assessing inbound lead market segment fit.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, lead_data: dict, company_data: dict) -> Task:
    icp_eval = deterministic_icp_check(lead_data, company_data)
    description = (
        f"Evaluate the ICP fit for this sales lead.\n"
        f"Lead Profile:\n{json.dumps(lead_data, indent=2)}\n\n"
        f"Company Details:\n{json.dumps(company_data, indent=2)}\n\n"
        f"Deterministic ICP Rule Evaluation from Python:\n{json.dumps(icp_eval, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Evaluate industry fit, employee size, and decision-maker seniority.\n"
        f"2. Categorize fit: 'STRONG_FIT', 'MODERATE_FIT', or 'DISQUALIFIED'.\n"
        f"3. Highlight specific reasons supporting the fit classification.\n"
        f"Return strictly valid JSON with keys: 'fit_category', 'industry_fit_score', 'company_size_fit_score', and 'fit_rationale'."
    )
    expected_output = "Strictly valid JSON containing fit_category, industry_fit_score, company_size_fit_score, and fit_rationale."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
