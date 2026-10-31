"""
Agent 4: Clause-to-Policy Mapping Agent for Contract Review and Compliance.

Responsibilities:
- Map extracted contract clauses to applicable policy requirements.
- Classify fulfillment status as satisfied, partially_satisfied, conflicting, or unaddressed.
- Preserve clause IDs, policy requirement IDs, and supporting clause text excerpts.
- Distinguish missing clauses from clauses that fail a requirement.
"""

from typing import Any, Dict, List, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import ClauseMappingOutput
from common.metrics import MetricsTracker

AGENT_ID = "clause_mapping_agent"
ROLE = "Clause-to-Policy Mapping Specialist"
GOAL = (
    "Align extracted contract clauses against structured corporate policy requirements, "
    "classifying compliance alignment status while preserving bidirectional citation IDs."
)
BACKSTORY = (
    "You are a legal compliance auditor. You map contract provisions to policy mandates, "
    "determining whether each requirement is satisfied, partially satisfied, unaddressed, or in conflict."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Clause-to-Policy Mapping Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_clause_mapping(
    contract_id: str,
    clause_extraction_data: Dict[str, Any],
    policy_interpretation_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Clause-to-Policy Mapping Agent."""
    active_agent = agent or create_agent()

    clauses = clause_extraction_data.get("clauses", [])
    policy_id = policy_interpretation_data.get("policy_id", "POL-UNKNOWN")
    mand_reqs = policy_interpretation_data.get("mandatory_requirements", [])
    rec_reqs = policy_interpretation_data.get("recommended_requirements", [])
    all_reqs = mand_reqs + rec_reqs

    task_desc = f"""
Contract ID: {contract_id}
Policy ID: {policy_id}

Policy Requirements to Map:
{all_reqs}

Extracted Contract Clauses:
{clauses}

Map each requirement to matching contract clause(s):
- Set mapping_status to one of: 'satisfied', 'partially_satisfied', 'conflicting', 'unaddressed'.
- Link clause_id (or null if unaddressed/missing).
- Provide supporting_clause_text excerpt.
- Note any specific threshold comparisons (e.g. Net 90 vs policy Net 45).

Return structured JSON matching:
{{
  "contract_id": "{contract_id}",
  "policy_id": "{policy_id}",
  "mappings": [
    {{
      "mapping_id": "MAP-01",
      "clause_id": "CLS-xxxx-PAY",
      "requirement_id": "REQ-FIN-01",
      "mapping_status": "conflicting",
      "supporting_clause_text": "Invoices shall be payable within ninety (90) days...",
      "notes": "Contract specifies Net 90 which conflicts with Net 45 policy limit."
    }}
  ],
  "coverage_percentage": 100.0
}}
"""
    expected_output = "Valid JSON matching ClauseMappingOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=ClauseMappingOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
