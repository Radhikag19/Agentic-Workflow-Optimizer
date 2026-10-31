"""
Agent 3: Policy Interpretation Agent for Contract Review and Compliance.

Responsibilities:
- Parse the supplied compliance policy into structured requirements.
- Assign stable policy requirement IDs.
- Distinguish mandatory rules from recommendations.
- Capture thresholds, conditions, and required evidence.
- Strictly do not introduce external legal rules that were not supplied in the policy.
"""

from typing import Any, Dict, List, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import PolicyInterpretationOutput
from common.metrics import MetricsTracker

AGENT_ID = "policy_interpretation_agent"
ROLE = "Policy Interpretation Specialist"
GOAL = (
    "Deconstruct internal organizational compliance policies into structured, "
    "discrete requirements with explicit thresholds, strictly avoiding external legal assumptions."
)
BACKSTORY = (
    "You are a compliance standards officer. You translate corporate governance policies "
    "into machine-actionable benchmarks, separating mandatory constraints from guidance recommendations."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Policy Interpretation Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_policy_interpretation(
    policy_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Policy Interpretation Agent."""
    active_agent = agent or create_agent()

    p_id = policy_data.get("policy_id", "POL-UNKNOWN")
    p_name = policy_data.get("policy_name", "Corporate Policy")
    p_ver = policy_data.get("version", "1.0")
    requirements = policy_data.get("requirements", [])

    task_desc = f"""
Policy ID: {p_id}
Policy Name: {p_name}
Version: {p_ver}
Policy Text Requirements:
{requirements}

Parse each policy rule:
- Assign stable requirement_id (e.g. REQ-SEC-01, REQ-FIN-01).
- Identify target clause_type ('data_protection', 'payment_terms', 'liability_cap', 'termination').
- Mark is_mandatory: true or false.
- Extract concrete requirement_text and measurable threshold_or_condition.

Return structured JSON matching:
{{
  "policy_id": "{p_id}",
  "policy_version": "{p_ver}",
  "policy_name": "{p_name}",
  "mandatory_requirements": [
    {{
      "requirement_id": "REQ-SEC-01",
      "clause_type": "data_protection",
      "is_mandatory": true,
      "requirement_text": "...",
      "threshold_or_condition": "48 hours notification",
      "exception_rules": []
    }}
  ],
  "recommended_requirements": [],
  "ambiguities_in_policy": []
}}
"""
    expected_output = "Valid JSON matching PolicyInterpretationOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=PolicyInterpretationOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
