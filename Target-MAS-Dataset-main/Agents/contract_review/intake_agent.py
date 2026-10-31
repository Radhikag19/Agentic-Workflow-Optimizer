"""
Agent 1: Contract Intake Agent for Contract Review and Compliance.

Responsibilities:
- Accept synthetic contract and supplied synthetic policy.
- Normalize inputs and identify contract type, document ID, version, and metadata.
- Validate document completeness and identify malformed or missing sections.
- Strictly never invent contract terms or policy requirements.
"""

from typing import Any, Dict, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import ContractIntakeOutput
from common.metrics import MetricsTracker

AGENT_ID = "contract_intake_agent"
ROLE = "Contract Intake and Ingestion Specialist"
GOAL = (
    "Ingest and validate the structure, metadata, versioning, and completeness "
    "of the target contract and associated compliance policy."
)
BACKSTORY = (
    "You are a contract operations analyst. You catalogue legal documents, "
    "verifying effective dates, executing parties, clause inventory count, and document completeness "
    "without ever assuming or injecting unstated terms."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Contract Intake Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_contract_intake(
    contract_data: Dict[str, Any],
    policy_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Contract Intake Agent."""
    active_agent = agent or create_agent()

    c_id = contract_data.get("contract_id", "CTR-UNKNOWN")
    c_title = contract_data.get("contract_title", "Commercial Agreement")
    c_type = contract_data.get("contract_type", "Standard Agreement")
    c_ver = contract_data.get("contract_version", "1.0")

    p_id = policy_data.get("policy_id", "POL-UNKNOWN")
    p_ver = policy_data.get("version", "1.0")

    clauses = contract_data.get("clauses", [])
    complete = len(clauses) > 0 and bool(contract_data.get("effective_date"))

    task_desc = f"""
Contract Record:
- Contract ID: {c_id}
- Title: {c_title}
- Type: {c_type}
- Version: {c_ver}
- Parties: {contract_data.get('party_a')} and {contract_data.get('party_b')}
- Effective Date: {contract_data.get('effective_date')}
- Expiration Date: {contract_data.get('expiration_date')}
- Clause Count: {len(clauses)}

Target Policy Record:
- Policy ID: {p_id}
- Policy Version: {p_ver}

Validate document structure and completeness.
Return structured JSON matching:
{{
  "document_id": "DOC-{c_id}",
  "contract_id": "{c_id}",
  "contract_title": "{c_title}",
  "contract_type": "{c_type}",
  "contract_version": "{c_ver}",
  "policy_id": "{p_id}",
  "policy_version": "{p_ver}",
  "document_complete": {str(complete).lower()},
  "section_count": {len(clauses)},
  "source_metadata": {{
    "party_a": "{contract_data.get('party_a')}",
    "party_b": "{contract_data.get('party_b')}",
    "effective_date": "{contract_data.get('effective_date')}",
    "expiration_date": "{contract_data.get('expiration_date')}"
  }},
  "intake_warnings": []
}}
"""
    expected_output = "Valid JSON matching ContractIntakeOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=ContractIntakeOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
