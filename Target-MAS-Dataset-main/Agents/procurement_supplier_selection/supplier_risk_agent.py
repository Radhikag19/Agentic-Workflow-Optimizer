"""
Agent 4: Supplier Risk Agent (Procurement)
Evaluates supplier quality history, delivery failure rates, and audit compliance.
"""

import json
from pathlib import Path
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent
from common.settings import DATA_DIR

AGENT_ID = "supplier_risk_agent"

def deterministic_risk_lookup(supplier_ids: list) -> dict:
    """Retrieves synthetic risk records from local files."""
    risk_file = DATA_DIR / "procurement_supplier_selection" / "risk_records.json"
    records = {}
    if risk_file.exists():
        with open(risk_file, "r", encoding="utf-8") as f:
            for r in json.load(f):
                if r.get("supplier_id") in supplier_ids:
                    records[r.get("supplier_id")] = r
    return records

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Procurement Risk & Compliance Auditor",
        goal="Assess supplier quality defect rates, historical failure frequency, and audit status.",
        backstory="A risk governance analyst evaluating supplier reliability and preventing vendor failures.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, supplier_ids: list) -> Task:
    risk_data = deterministic_risk_lookup(supplier_ids)
    description = (
        f"Perform risk assessment across all competing suppliers.\n"
        f"Synthetic Supplier Risk Records:\n{json.dumps(risk_data, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Evaluate defect rates, delivery failure rates, and audit warnings.\n"
        f"2. Categorize each supplier's risk ('LOW', 'MEDIUM', 'HIGH').\n"
        f"3. Highlight red flags that should block supplier selection.\n"
        f"Return strictly valid JSON with keys: 'supplier_risk_scores', 'flagged_suppliers', and 'risk_mitigation_notes'."
    )
    expected_output = "Strictly valid JSON containing supplier_risk_scores, flagged_suppliers, and risk_mitigation_notes."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
