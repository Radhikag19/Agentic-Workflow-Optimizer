"""
Agent 9: Independent Review Agent for Contract Review and Compliance.

Responsibilities:
- Check findings for duplication, inconsistent severity, and missing requirements.
- Resolve duplicate findings while preserving distinct legal/operational issues.
- Identify disagreements among preceding agents.
- Flag matters requiring qualified human legal counsel.
"""

from typing import Any, Dict, List, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import IndependentReviewOutput
from common.metrics import MetricsTracker

AGENT_ID = "contract_independent_review_agent"
ROLE = "Senior Contract Compliance Reviewer"
GOAL = (
    "Perform a critical second-pass quality review across all extracted issues, "
    "deduplicating redundant findings, calibrating severity ratings, and highlighting human review items."
)
BACKSTORY = (
    "You are a managing corporate legal auditor. You sanity check the work of junior reviewers, "
    "ensuring findings are succinct, severity levels are proportional, and genuine legal ambiguities "
    "are escalated to human attorneys."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Independent Review Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_independent_review(
    contract_id: str,
    deviation_data: Dict[str, Any],
    risk_data: Dict[str, Any],
    consistency_data: Dict[str, Any],
    evidence_validation_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Independent Review Agent."""
    active_agent = agent or create_agent()

    deviations = deviation_data.get("deviations", [])
    risks = risk_data.get("risk_findings", [])
    inconsistencies = consistency_data.get("inconsistencies", [])
    ev_valid = evidence_validation_data.get("all_citations_valid", True)

    total_issues = len(deviations) + len(risks) + len(inconsistencies)

    task_desc = f"""
Contract ID: {contract_id}
Deviations Count: {len(deviations)} ({deviations})
Risks Count: {len(risks)} ({risks})
Inconsistencies Count: {len(inconsistencies)} ({inconsistencies})
Evidence Verification Status: {ev_valid}

Conduct comprehensive QA review:
1. Merge any duplicate or overlapping findings.
2. Verify severity consistency (e.g., critical breaches vs minor payment delays).
3. Identify unresolved matters requiring human legal review.
4. Set review_status to 'approved_for_report' if findings are sound.

Return structured JSON matching:
{{
  "contract_id": "{contract_id}",
  "review_status": "approved_for_report",
  "duplicate_findings_merged": 0,
  "severity_adjustments": [],
  "unresolved_matters": ["Formal legal review required for identified deviations and risk clauses"]
}}
"""
    expected_output = "Valid JSON matching IndependentReviewOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=IndependentReviewOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
