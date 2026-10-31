"""
Agent 5: Compliance Deviation Agent for Contract Review and Compliance.

Responsibilities:
- Evaluate mapped clauses against supplied policy requirements.
- Identify deviations, omitted protections, incompatible terms, and conditional conflicts.
- Assign severity (low, medium, high, critical) using documented risk criteria.
- Explain concrete evidence supporting each finding.
"""

from typing import Any, Dict, List, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import ComplianceDeviationOutput
from common.metrics import MetricsTracker

AGENT_ID = "compliance_deviation_agent"
ROLE = "Contract Compliance Deviation Evaluator"
GOAL = (
    "Identify and grade specific deviations where contractual terms fall short of or "
    "contradict corporate policy requirements, detailing factual evidence and severity."
)
BACKSTORY = (
    "You are a regulatory and corporate compliance reviewer. You inspect clause alignments, "
    "pinpointing precise contractual shortfalls and assigning structured severity levels without legal speculation."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Compliance Deviation Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_compliance_deviation(
    contract_id: str,
    clause_mapping_data: Dict[str, Any],
    policy_interpretation_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Compliance Deviation Agent."""
    active_agent = agent or create_agent()

    mappings = clause_mapping_data.get("mappings", [])

    task_desc = f"""
Contract ID: {contract_id}
Clause Mappings: {mappings}

Identify all compliance deviations:
- For any mapping with status 'conflicting' or 'unaddressed' (on a mandatory requirement):
  - Assign severity: 'critical' (for mandatory breach notification or unlimited liability), 'high' (for payment term violations > 30 days deviation), 'medium', or 'low'.
  - Provide deviation_summary.
  - Quote supporting_evidence.
  - Recommend clear remediation language.

Return structured JSON matching:
{{
  "contract_id": "{contract_id}",
  "deviations": [
    {{
      "deviation_id": "DEV-01",
      "requirement_id": "REQ-FIN-01",
      "clause_id": "CLS-xxxx-PAY",
      "severity": "high",
      "deviation_summary": "Payment term of Net 90 exceeds policy ceiling of Net 45.",
      "supporting_evidence": "Invoices shall be payable within ninety (90) days...",
      "recommended_remediation": "Amend payment terms to Net 45 or shorter."
    }}
  ],
  "has_critical_deviations": false
}}
"""
    expected_output = "Valid JSON matching ComplianceDeviationOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=ComplianceDeviationOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
