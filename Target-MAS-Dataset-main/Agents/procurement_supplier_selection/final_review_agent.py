"""
Agent 9: Final Review Agent (Procurement)
Verifies multi-agent evaluation integrity and produces the final procurement award report.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "final_review_agent_procurement"

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Chief Procurement Officer & Audit Lead",
        goal="Verify evaluation integrity, confirm weighted arithmetic, and authorize the final sourcing report.",
        backstory="The final governance authority ensuring procurement awards comply with organizational ethics and policy.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, complete_procurement_audit: dict) -> Task:
    description = (
        f"Perform the final governance audit on this supplier selection process.\n"
        f"Complete Sourcing Audit File:\n{json.dumps(complete_procurement_audit, indent=2)}\n\n"
        f"Tasks:\n"
        f"1. Verify that the recommended supplier actually meets mandatory criteria and passed weighted evaluation.\n"
        f"2. Confirm there are no unverified cost calculations or unaddressed risk flags.\n"
        f"3. Generate the final executive procurement report.\n"
        f"Output strictly valid JSON with keys: 'workflow_status' ('COMPLETED'), 'selected_supplier', 'compliance_verification', 'summary_report', and 'award_justification'."
    )
    expected_output = "Strictly valid JSON containing workflow_status, selected_supplier, compliance_verification, summary_report, and award_justification."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
