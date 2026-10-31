"""
Agent 6: Risk and Ambiguity Agent for Contract Review and Compliance.

Responsibilities:
- Identify ambiguous wording, unclear obligations, and operational risks under the supplied policy.
- Distinguish explicit policy deviations from vague terms requiring human interpretation.
- Preserve source clause citations.
- Avoid unsupported legal conclusions.
"""

from typing import Any, Dict, List, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import RiskAmbiguityOutput
from common.metrics import MetricsTracker

AGENT_ID = "risk_ambiguity_agent"
ROLE = "Contract Risk and Ambiguity Analyst"
GOAL = (
    "Detect vague wording, undefined timelines, one-sided covenants, and latent "
    "operational risks within contractual text, highlighting issues requiring human legal counsel."
)
BACKSTORY = (
    "You are an operational risk analyst. You flag loose phrasing such as 'best commercial efforts', "
    "'reasonable timeframe', or vague indemnity scopes that expose the organization to unexpected liabilities."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Risk and Ambiguity Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_risk_ambiguity(
    contract_id: str,
    clause_extraction_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Risk and Ambiguity Agent."""
    active_agent = agent or create_agent()

    clauses = clause_extraction_data.get("clauses", [])

    task_desc = f"""
Contract ID: {contract_id}
Contract Clauses to Inspect for Ambiguity:
{clauses}

Identify ambiguous phrases, vague timelines, or open-ended liabilities:
- Look for non-committal language (e.g. 'endeavor to report in reasonable timeframe').
- Flag unquantified risk categories ('operational', 'legal_exposure', 'vague_terms', 'renewal_trap').
- Assign severity ('low', 'medium', 'high').
- Set human_interpretation_needed: true.

Return structured JSON matching:
{{
  "contract_id": "{contract_id}",
  "risk_findings": [
    {{
      "finding_id": "RSK-01",
      "clause_id": "CLS-xxxx-DATA",
      "risk_category": "vague_terms",
      "ambiguity_description": "Phrase 'reasonable timeframe' lacks defined hours/days threshold for security incidents.",
      "severity": "medium",
      "human_interpretation_needed": true
    }}
  ]
}}
"""
    expected_output = "Valid JSON matching RiskAmbiguityOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=RiskAmbiguityOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
