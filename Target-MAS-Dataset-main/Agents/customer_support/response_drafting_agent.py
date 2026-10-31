"""
Agent 5: Response Drafting Agent (Customer Support)
Drafts customer-facing communications and action steps.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "response_drafting_agent"

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Customer Communications Specialist",
        goal="Draft clear, empathetic, and professional customer responses explaining the resolution.",
        backstory="An expert in customer experience communication ensuring clear guidance and professional empathy.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, ticket_data: dict, resolution_data: dict) -> Task:
    description = (
        f"Draft a formal, customer-facing response message.\n"
        f"Ticket Context:\n{json.dumps(ticket_data, indent=2)}\n\n"
        f"Approved Resolution Strategy:\n{json.dumps(resolution_data, indent=2)}\n\n"
        f"Guidelines:\n"
        f"1. Acknowledge the customer's specific issue empathetically.\n"
        f"2. Explain the resolution or required user action with clear bullet points.\n"
        f"3. Do not disclose internal reasoning, prompt names, or unverified claims.\n"
        f"4. Do not promise external actions the system cannot guarantee.\n"
        f"Return strictly valid JSON with keys: 'response_text', 'tone', 'next_steps_for_customer', and 'ticket_status'."
    )
    expected_output = "Strictly valid JSON containing response_text, tone, next_steps_for_customer, and ticket_status."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
