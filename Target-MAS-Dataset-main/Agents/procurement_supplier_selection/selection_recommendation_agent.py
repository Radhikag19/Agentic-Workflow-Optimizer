"""
Agent 8: Selection Recommendation Agent (Procurement)
Aggregates multidimensional evaluation evidence and recommends supplier award.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "selection_recommendation_agent"

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Strategic Sourcing Committee Chair",
        goal="Synthesize technical, delivery, commercial, risk, and scoring evidence into an authoritative supplier recommendation.",
        backstory="An executive procurement director balancing strategic risk, quality, and commercial value.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, all_procurement_evidence: dict) -> Task:
    description = (
        f"Formulate an authoritative supplier award recommendation.\n"
        f"Consolidated Procurement Evidence:\n{json.dumps(all_procurement_evidence, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Recommend the winning supplier or explain why no supplier qualifies.\n"
        f"2. Explicitly detail trade-offs (e.g. lowest cost vs. delivery buffer).\n"
        f"3. Note: NEVER issue binding contracts or purchase orders.\n"
        f"Return strictly valid JSON with keys: 'recommended_supplier_id', 'commercial_rationale', 'risk_acceptance', and 'next_steps'."
    )
    expected_output = "Strictly valid JSON containing recommended_supplier_id, commercial_rationale, risk_acceptance, and next_steps."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
