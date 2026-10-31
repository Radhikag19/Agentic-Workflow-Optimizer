"""
Agent 8: Final Review Agent (Invoice Processing)
Verifies end-to-end evidence consistency and produces the final accounts payable audit report.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "final_review_agent_invoice"

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Chief AP Quality & Verification Auditor",
        goal="Perform final consistency verification across all stages and produce the authoritative accounts payable report.",
        backstory="The final executive gatekeeper ensuring audit trails are complete, robust, and mathematically sound.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, all_stage_results: dict) -> Task:
    description = (
        f"Perform the final comprehensive review of this accounts payable workflow execution.\n"
        f"Complete Audit Trail:\n{json.dumps(all_stage_results, indent=2)}\n\n"
        f"Tasks:\n"
        f"1. Verify that the recommendation aligns with math validation, PO matching, policy checks, and duplicate findings.\n"
        f"2. Confirm there are no unaddressed critical exceptions.\n"
        f"3. Generate the final structured AP review report.\n"
        f"Output strictly valid JSON with keys: 'workflow_status' ('COMPLETED'), 'invoice_id', 'final_recommendation', 'risk_level', 'summary_report', and 'audit_findings'."
    )
    expected_output = "Strictly valid JSON containing workflow_status, invoice_id, final_recommendation, risk_level, summary_report, and audit_findings."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
