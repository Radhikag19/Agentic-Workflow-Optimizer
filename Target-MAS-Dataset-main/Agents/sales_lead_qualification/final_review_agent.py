"""
Agent 7: Final Review Agent (Sales Lead Qualification)
Verifies qualification consistency and produces the final lead evaluation report.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "final_review_agent_sales"

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Revenue Operations QA Lead",
        goal="Verify lead qualification accuracy, scoring consistency, and handoff validity.",
        backstory="An executive operations leader maintaining high pipeline hygiene and conversion accuracy.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, complete_lead_audit: dict) -> Task:
    description = (
        f"Perform the final quality review on this sales lead qualification.\n"
        f"Complete Lead Qualification Audit:\n{json.dumps(complete_lead_audit, indent=2)}\n\n"
        f"Tasks:\n"
        f"1. Verify that the next action strictly aligns with the qualification score tier.\n"
        f"2. Ensure obviously unsuitable leads are not marked as high priority.\n"
        f"3. Generate the final structured qualification report.\n"
        f"Output strictly valid JSON with keys: 'workflow_status' ('COMPLETED'), 'lead_id', 'final_qualification_tier', 'final_score', 'summary_report', and 'audit_findings'."
    )
    expected_output = "Strictly valid JSON containing workflow_status, lead_id, final_qualification_tier, final_score, summary_report, and audit_findings."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
