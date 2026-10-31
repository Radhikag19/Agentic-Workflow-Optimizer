"""
Agent 3: Intent Analysis Agent (Sales Lead Qualification)
Analyzes engagement events, demo requests, and timeline urgency to assess buying intent.
"""

import json
from pathlib import Path
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent
from common.settings import DATA_DIR

AGENT_ID = "intent_analysis_agent"

def deterministic_intent_lookup(lead_id: str) -> list:
    """Retrieves synthetic engagement event history from local files."""
    eng_file = DATA_DIR / "sales_lead_qualification" / "engagement_histories.json"
    if eng_file.exists():
        with open(eng_file, "r", encoding="utf-8") as f:
            for e in json.load(f):
                if e.get("lead_id") == lead_id:
                    return e.get("events", [])
    return []

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Buyer Intent & Behavioral Analyst",
        goal="Quantify purchase intent based on interaction history, content consumption, and stated timeline.",
        backstory="A growth marketing analyst interpreting digital engagement signals and buyer readiness.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, lead_data: dict) -> Task:
    lead_id = lead_data.get("lead_id", "")
    events = deterministic_intent_lookup(lead_id)
    description = (
        f"Analyze buyer intent for this lead.\n"
        f"Lead Details (Timeline & Stated Budget):\n{json.dumps(lead_data, indent=2)}\n\n"
        f"Recorded Engagement Events:\n{json.dumps(events, indent=2)}\n\n"
        f"Tasks:\n"
        f"1. Classify buyer intent: 'HIGH_INTENT', 'MEDIUM_INTENT', or 'LOW_INTENT'.\n"
        f"2. Cite high-value signals (e.g. demo request, pricing page views) vs. weak signals (newsletter signup).\n"
        f"Return strictly valid JSON with keys: 'intent_level', 'intent_score', 'high_value_signals', and 'behavioral_summary'."
    )
    expected_output = "Strictly valid JSON containing intent_level, intent_score, high_value_signals, and behavioral_summary."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
