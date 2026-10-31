"""
Agent 2: Supplier Profile Agent (Procurement)
Retrieves and assesses supplier capabilities, capacity, and operational eligibility.
"""

import json
from pathlib import Path
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent
from common.settings import DATA_DIR

AGENT_ID = "supplier_profile_agent"

def deterministic_profile_lookup(supplier_ids: list) -> dict:
    """Retrieves synthetic supplier profiles from local files."""
    prof_file = DATA_DIR / "procurement_supplier_selection" / "supplier_profiles.json"
    profiles = {}
    if prof_file.exists():
        with open(prof_file, "r", encoding="utf-8") as f:
            for s in json.load(f):
                if s.get("supplier_id") in supplier_ids:
                    profiles[s.get("supplier_id")] = s
    return profiles

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Supplier Intelligence Officer",
        goal="Evaluate supplier capabilities, manufacturing capacity, and operational standing.",
        backstory="A strategic sourcing specialist who audits supplier corporate health, certifications, and technical capacity.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, supplier_ids: list) -> Task:
    profiles = deterministic_profile_lookup(supplier_ids)
    description = (
        f"Evaluate the profiles of competing suppliers.\n"
        f"Target Supplier Profiles from Registry:\n{json.dumps(profiles, indent=2)}\n\n"
        f"Tasks:\n"
        f"1. Compare technical capability, ISO certifications, and manufacturing capacity.\n"
        f"2. Note any operational limitations or unverified profile data.\n"
        f"Return strictly valid JSON with keys: 'evaluated_profiles', 'profile_strengths', and 'operational_concerns'."
    )
    expected_output = "Strictly valid JSON containing evaluated_profiles, profile_strengths, and operational_concerns."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
