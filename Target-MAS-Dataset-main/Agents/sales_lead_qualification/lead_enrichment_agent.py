"""
Agent 1: Lead Enrichment Agent (Sales Lead Qualification)
Enriches lead records with firmographic and company data from local synthetic datasets.
"""

import json
from pathlib import Path
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent
from common.settings import DATA_DIR

AGENT_ID = "lead_enrichment_agent"

def deterministic_enrichment(company_id: str) -> dict:
    """Retrieves synthetic company firmographic details from local files."""
    comp_file = DATA_DIR / "sales_lead_qualification" / "company_profiles.json"
    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            for c in json.load(f):
                if c.get("company_id") == company_id:
                    return c
    return {}

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Lead Enrichment & Intelligence Specialist",
        goal="Normalize contact and company fields using verified internal registry records.",
        backstory="A CRM operations specialist normalizing inbound sales records against known company databases.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, lead_data: dict) -> Task:
    comp_id = lead_data.get("company_id", "")
    company_info = deterministic_enrichment(comp_id)
    description = (
        f"Enrich this inbound sales lead record.\n"
        f"Inbound Lead Data:\n{json.dumps(lead_data, indent=2)}\n\n"
        f"Synthetic Company Profile from Internal Database:\n{json.dumps(company_info, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Normalize lead contact name, title, company industry, employee count, and annual revenue.\n"
        f"2. Retain company_id and lead_id references.\n"
        f"3. Note: Do not make external web searches or invent missing attributes.\n"
        f"Return strictly valid JSON with keys: 'lead_id', 'enriched_lead_profile', and 'missing_attributes'."
    )
    expected_output = "Strictly valid JSON containing lead_id, enriched_lead_profile, and missing_attributes."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
