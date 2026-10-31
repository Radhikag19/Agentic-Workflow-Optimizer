"""
Agent 10: Final Compliance Report Agent for Contract Review and Compliance.

Responsibilities:
- Synthesize all findings into the final comprehensive structured compliance report.
- Deliver executive summary, policy coverage, clause inventory, deviations, risks, and cross-clause contradictions.
- Separate explicit policy mismatches from ambiguous concerns.
- Include unresolved questions, recommended actions, and clear legal disclaimers.
- Produce output matching Section 7 ContractFinalReportOutput schema.
"""

from typing import Any, Dict, List, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import ContractFinalReportOutput
from common.metrics import MetricsTracker

AGENT_ID = "final_compliance_report_agent"
ROLE = "Lead Contract Compliance Report Author"
GOAL = (
    "Assemble and author the definitive, evidence-backed contract compliance report, "
    "structuring deviations, risks, and recommended actions with explicit disclaimers "
    "stating that AI review does not replace licensed legal counsel."
)
BACKSTORY = (
    "You are an executive legal operations director. You synthesize multi-agent review findings "
    "into polished, structured governance reports that enable corporate risk and legal teams "
    "to make informed contracting decisions."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Final Compliance Report Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_final_report(
    run_id: str,
    intake_data: Dict[str, Any],
    clause_extraction_data: Dict[str, Any],
    policy_interpretation_data: Dict[str, Any],
    clause_mapping_data: Dict[str, Any],
    deviation_data: Dict[str, Any],
    risk_data: Dict[str, Any],
    consistency_data: Dict[str, Any],
    evidence_validation_data: Dict[str, Any],
    independent_review_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Final Compliance Report Agent."""
    active_agent = agent or create_agent()

    c_id = intake_data.get("contract_id", "CTR-UNKNOWN")
    c_ver = intake_data.get("contract_version", "1.0")
    p_id = intake_data.get("policy_id", "POL-UNKNOWN")
    p_ver = intake_data.get("policy_version", "1.0")
    c_type = intake_data.get("contract_type", "Standard Agreement")

    clauses = clause_extraction_data.get("clauses", [])
    mappings = clause_mapping_data.get("mappings", [])
    deviations = deviation_data.get("deviations", [])
    risks = risk_data.get("risk_findings", [])
    inconsistencies = consistency_data.get("inconsistencies", [])
    ev_val = evidence_validation_data

    coverage = {
        "total_requirements": len(mappings),
        "satisfied": len([m for m in mappings if m.get("mapping_status") == "satisfied"]),
        "conflicting": len([m for m in mappings if m.get("mapping_status") == "conflicting"]),
        "unaddressed": len([m for m in mappings if m.get("mapping_status") == "unaddressed"]),
        "coverage_percentage": clause_mapping_data.get("coverage_percentage", 100.0),
    }

    has_issues = len(deviations) > 0 or len(risks) > 0 or len(inconsistencies) > 0

    task_desc = f"""
Assemble the definitive final contract review report:
- Run ID: {run_id}
- Contract ID: {c_id} (Version {c_ver}, Type: {c_type})
- Policy ID: {p_id} (Version {p_ver})
- Clause Count: {len(clauses)}
- Policy Coverage: {coverage}
- Compliance Deviations: {deviations}
- Operational Risks & Ambiguities: {risks}
- Cross-Clause Contradictions: {inconsistencies}
- Evidence Validation: {ev_val}
- Review Reviewer Notes: {independent_review_data.get('unresolved_matters')}

Requirements:
1. Executive summary synthesizing contract compliance posture.
2. Distinct sections for explicit policy deviations vs ambiguous concerns.
3. Recommended follow-up actions for counsel/procurement.
4. Set human_review_required: true if deviations or risks exist.
5. Emphasize that zero detected issues does NOT constitute a warranty of legal validity.

Return structured JSON matching:
{{
  "run_id": "{run_id}",
  "contract_id": "{c_id}",
  "contract_version": "{c_ver}",
  "policy_id": "{p_id}",
  "policy_version": "{p_ver}",
  "contract_type": "{c_type}",
  "executive_summary": "...",
  "policy_coverage": {coverage},
  "clause_inventory": {clauses},
  "compliance_findings": {deviations},
  "risk_findings": {risks},
  "cross_clause_inconsistencies": {inconsistencies},
  "evidence_validation": {ev_val},
  "unresolved_questions": ["Requires confirmation of whether payment terms can be adjusted to Net 45"],
  "recommended_actions": ["Request amendment to align payment terms with corporate policy"],
  "human_review_required": {str(has_issues).lower()},
  "validation_warnings": [],
  "legal_disclaimer": "This AI-assisted review is for preliminary compliance screening only and is NOT a substitute for qualified legal or compliance counsel. No detected issues does NOT imply complete legal validity."
}}
"""
    expected_output = "Valid JSON matching ContractFinalReportOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=ContractFinalReportOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
