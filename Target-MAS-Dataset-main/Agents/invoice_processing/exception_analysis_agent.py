"""
Agent 6: Exception Analysis Agent
Consolidates and analyzes anomalies, policy exceptions, math discrepancies, and PO mismatches.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "exception_analysis_agent"

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Accounts Payable Exception Specialist",
        goal="Consolidate discrepancies, classify exception severities, and flag operational risks across all stages.",
        backstory="A senior risk analyst evaluating edge cases, discrepancies, and policy violations in payment workflows.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, upstream_evidence: dict) -> Task:
    description = (
        f"Analyze all upstream findings for accounts payable exceptions.\n"
        f"Consolidated Evidence:\n{json.dumps(upstream_evidence, indent=2)}\n\n"
        f"Tasks:\n"
        f"1. Consolidate math discrepancies, PO mismatch findings, policy violations, and duplicate alerts.\n"
        f"2. Assign an exception severity to each issue ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').\n"
        f"3. Note whether any exceptions are resolvable with further human review or are fatal blockers.\n"
        f"Return strictly valid JSON with keys: 'has_exceptions', 'exception_count', 'critical_exceptions', and 'detailed_exceptions'."
    )
    expected_output = "Strictly valid JSON containing has_exceptions, exception_count, critical_exceptions, and detailed_exceptions."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
