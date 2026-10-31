"""
Agent 1: Quotation Analysis Agent (Procurement)
Extracts and normalizes supplier quotations against the procurement request.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "quotation_analysis_agent"

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Senior Quotation Analyst",
        goal="Extract, normalize, and validate competing supplier bids and commercial terms.",
        backstory="A procurement analyst specializing in normalizing commercial bids across disparate supplier formats.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, procurement_request: dict, quotations: list) -> Task:
    description = (
        f"Analyze and normalize competing quotations for this procurement request.\n"
        f"Procurement Request:\n{json.dumps(procurement_request, indent=2)}\n\n"
        f"Submitted Supplier Quotations:\n{json.dumps(quotations, indent=2)}\n\n"
        f"Tasks:\n"
        f"1. Normalize item quantities, unit prices, shipping, and taxes per supplier.\n"
        f"2. Identify missing pricing elements, validity periods, or currency discrepancies.\n"
        f"3. Return normalized quotation summaries for all competing bidders.\n"
        f"Return strictly valid JSON with keys: 'request_id', 'normalized_quotations', and 'extraction_flags'."
    )
    expected_output = "Strictly valid JSON containing request_id, normalized_quotations, and extraction_flags."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
