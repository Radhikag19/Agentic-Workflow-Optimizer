"""
Agent 6: Resolution Verification Agent (Customer Support)
Verifies resolution accuracy, policy adherence, and drafts final support report.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "resolution_verification_agent"

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Support QA & Verification Auditor",
        goal="Verify that the drafted response fully addresses the ticket within policy boundaries.",
        backstory="A quality assurance lead ensuring high customer satisfaction and strict policy adherence.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, full_ticket_audit: dict) -> Task:
    description = (
        f"Perform quality audit on the proposed ticket resolution and customer draft.\n"
        f"Full Support Audit Trail:\n{json.dumps(full_ticket_audit, indent=2)}\n\n"
        f"Verification Checks:\n"
        f"1. Does the drafted response completely solve the customer's stated complaint?\n"
        f"2. Are there any unsupported commitments, policy contradictions, or missing instructions?\n"
        f"3. Mark final status: 'RESOLVED', 'ACTION_REQUIRED_BY_USER', or 'ESCALATED'.\n"
        f"Return strictly valid JSON with keys: 'qa_status' ('APPROVED' or 'REJECTED'), 'final_ticket_status', 'verification_notes', and 'support_summary_report'."
    )
    expected_output = "Strictly valid JSON containing qa_status, final_ticket_status, verification_notes, and support_summary_report."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
