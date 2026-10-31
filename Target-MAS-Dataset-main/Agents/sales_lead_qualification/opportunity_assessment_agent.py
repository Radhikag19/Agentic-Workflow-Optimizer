"""
Agent 5: Opportunity Assessment Agent (Sales Lead Qualification)
Assesses overall commercial opportunity viability and strategic value.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "opportunity_assessment_agent"

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Commercial Opportunity Assessor",
        goal="Synthesize fit, intent, and scoring into a commercial deal viability assessment.",
        backstory="A strategic account director evaluating enterprise deal feasibility and sales viability.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, upstream_evidence: dict) -> Task:
    description = (
        f"Evaluate the commercial opportunity viability for this lead.\n"
        f"Consolidated Lead Evidence:\n{json.dumps(upstream_evidence, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Evaluate whether this opportunity is high-potential, nurture-track, or fundamentally unviable.\n"
        f"2. Note whether any missing information blocks advancing the deal.\n"
        f"Return strictly valid JSON with keys: 'opportunity_viability' ('HIGH', 'MEDIUM', 'UNVIABLE'), 'estimated_deal_size', 'pipeline_stage', and 'key_risks'."
    )
    expected_output = "Strictly valid JSON containing opportunity_viability, estimated_deal_size, pipeline_stage, and key_risks."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
