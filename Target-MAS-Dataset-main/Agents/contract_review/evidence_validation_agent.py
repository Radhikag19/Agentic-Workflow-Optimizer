"""
Agent 8: Evidence and Citation Validation Agent for Contract Review and Compliance.

Responsibilities:
- Verify that each finding references existing clauses and valid policy requirement IDs.
- Check that cited excerpts genuinely support downstream conclusions.
- Identify unsupported assertions, invalid IDs, or contradictory evidence.
- Return validated findings and unresolved evidence issues.
"""

from typing import Any, Dict, List, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import EvidenceValidationOutput
from common.metrics import MetricsTracker

AGENT_ID = "evidence_validation_agent"
ROLE = "Legal Evidence and Citation Validator"
GOAL = (
    "Audit all generated deviations, risks, and contradictions to confirm every finding "
    "cites a real, verified clause ID and that quoted excerpts directly support the claim."
)
BACKSTORY = (
    "You are a forensic legal proofreader. You verify that cited sections and IDs exist in the "
    "source contract and policy, catching any hallucinated clause numbers or unsupported conclusions."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Evidence and Citation Validation Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_evidence_validation(
    contract_id: str,
    clause_extraction_data: Dict[str, Any],
    policy_interpretation_data: Dict[str, Any],
    deviation_data: Dict[str, Any],
    risk_data: Dict[str, Any],
    consistency_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Evidence and Citation Validation Agent."""
    active_agent = agent or create_agent()

    valid_clause_ids = [c.get("clause_id") for c in clause_extraction_data.get("clauses", [])]
    valid_req_ids = [
        r.get("requirement_id")
        for r in policy_interpretation_data.get("mandatory_requirements", [])
        + policy_interpretation_data.get("recommended_requirements", [])
    ]

    deviations = deviation_data.get("deviations", [])
    risks = risk_data.get("risk_findings", [])
    inconsistencies = consistency_data.get("inconsistencies", [])

    task_desc = f"""
Contract ID: {contract_id}
Valid Clause IDs in Document: {valid_clause_ids}
Valid Policy Requirement IDs: {valid_req_ids}

Findings to Validate:
- Deviations: {deviations}
- Risks: {risks}
- Inconsistencies: {inconsistencies}

Verify evidence integrity:
1. Check that clause_id in each finding is among Valid Clause IDs (or null if missing clause).
2. Check that requirement_id is among Valid Policy Requirement IDs.
3. Confirm whether all citations are valid.
4. Return verified finding IDs and note any invalid or unsupported references.

Return structured JSON matching:
{{
  "contract_id": "{contract_id}",
  "all_citations_valid": true,
  "verified_finding_ids": ["DEV-01", "RSK-01", "INC-01"],
  "invalid_or_unsupported_findings": [],
  "evidence_notes": ["All cited clauses verified against primary document inventory."]
}}
"""
    expected_output = "Valid JSON matching EvidenceValidationOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=EvidenceValidationOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
