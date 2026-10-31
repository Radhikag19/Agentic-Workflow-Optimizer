"""
Agent 7: Screening Review Agent for Recruitment and Resume Screening.

Responsibilities:
- Check for contradictory assessments, duplicated findings, and unsupported claims.
- Verify that mandatory and preferred criteria are strictly segregated.
- Check evidence traceability (valid evidence IDs for all claims).
- Identify issues requiring mandatory recruiter review.
"""

from typing import Any, Dict, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import ScreeningReviewOutput
from common.metrics import MetricsTracker

AGENT_ID = "screening_review_agent"
ROLE = "Screening Quality and Integrity Reviewer"
GOAL = (
    "Conduct an independent quality check across prior agent findings, ensuring "
    "evidence traceability, non-contradiction, and distinct treatment of mandatory vs preferred criteria."
)
BACKSTORY = (
    "You are a recruitment quality assurance lead. You verify that every assessment made by "
    "upstream screeners is supported by traceable resume evidence, that mandatory criteria have not "
    "been conflated with optional preferences, and that no contradictory conclusions exist."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Screening Review Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_screening_review(
    candidate_id: str,
    evidence_mapping_data: Dict[str, Any],
    assessment_data: Dict[str, Any],
    missing_evidence_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Screening Review Agent."""
    active_agent = agent or create_agent()

    mappings = evidence_mapping_data.get("mappings", [])
    strengths = assessment_data.get("strengths", [])
    gaps = assessment_data.get("experience_gaps", [])
    inconsistencies = missing_evidence_data.get("inconsistencies", [])

    # Deterministic pre-check for mandatory criteria
    mandatory_unsupported = [
        m for m in mappings if m.get("requirement_type") == "mandatory" and m.get("status") in ["unsupported", "unclear"]
    ]
    mandatory_met_deterministic = len(mandatory_unsupported) == 0

    task_desc = f"""
Candidate ID: {candidate_id}
Criteria Mappings: {mappings}
Identified Strengths: {strengths}
Identified Gaps: {gaps}
Flagged Inconsistencies: {inconsistencies}

Conduct independent quality check:
1. Verify if any finding contradicts another (e.g., claimed as strength but mapped as unsupported).
2. Check evidence traceability: Are all mapped criteria linked to valid evidence IDs?
3. Verify whether all mandatory criteria are met (mandatory_criteria_met: {str(mandatory_met_deterministic).lower()}).
4. List specific issues requiring human recruiter review.
5. Set review_passed to true if findings are consistent and traceable.

Return structured JSON matching:
{{
  "candidate_id": "{candidate_id}",
  "review_passed": true,
  "contradictions_detected": [],
  "traceability_verified": true,
  "mandatory_criteria_met": {str(mandatory_met_deterministic).lower()},
  "issues_requiring_human_review": ["Recruiter review needed for date inconsistency in EXP1", ...]
}}
"""
    expected_output = "Valid JSON matching ScreeningReviewOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=ScreeningReviewOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
