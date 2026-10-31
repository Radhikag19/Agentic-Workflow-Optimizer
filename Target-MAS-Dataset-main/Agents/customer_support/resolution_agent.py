"""
Agent 4: Resolution Agent (Customer Support)
Determines the structured resolution recommendation and concrete action steps.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "resolution_agent"

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Customer Resolution Strategist",
        goal="Synthesize policies and technical investigation into a clear, actionable customer resolution.",
        backstory="A tier-3 senior resolution specialist with authority to recommend credits, troubleshooting, or escalations.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, ticket_data: dict, stage_evidence: dict) -> Task:
    description = (
        f"Formulate a concrete resolution recommendation for this ticket.\n"
        f"Ticket Data:\n{json.dumps(ticket_data, indent=2)}\n\n"
        f"Upstream Evidence:\n{json.dumps(stage_evidence, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Recommend resolution type: 'immediate_resolution', 'requires_user_action', or 'escalate_to_engineering'.\n"
        f"2. Provide step-by-step resolution actions conforming strictly to retrieved policy rules.\n"
        f"3. Note: Never execute actual database alterations or financial credits.\n"
        f"Return strictly valid JSON with keys: 'resolution_type', 'recommended_action', 'policy_justification', and 'escalation_required'."
    )
    expected_output = "Strictly valid JSON containing resolution_type, recommended_action, policy_justification, and escalation_required."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
