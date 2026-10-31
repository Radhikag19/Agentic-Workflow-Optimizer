"""
Agent 6: Missing Evidence and Consistency Agent for Recruitment and Resume Screening.

Responsibilities:
- Identify missing information, ambiguous claims, inconsistent dates, and unsupported proficiency claims.
- Identify requirements that cannot be evaluated reliably due to incomplete documentation.
- Distinguish confirmed inconsistencies from questions requiring recruiter clarification.
- Intentionally overlap with evidence mapping to support future consolidation experiments.
"""

from typing import Any, Dict, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import MissingEvidenceOutput
from common.metrics import MetricsTracker

AGENT_ID = "missing_evidence_agent"
ROLE = "Recruitment Consistency and Verification Auditor"
GOAL = (
    "Audit the resume and qualification records for chronological anomalies, "
    "unsupported buzzword claims, missing mandatory evidence, and ambiguous statements."
)
BACKSTORY = (
    "You are a forensic resume auditor. You cross-check timeline continuity (such as graduation "
    "dates vs employment dates), unbacked keyword stuffing, and unverified leadership claims, "
    "clearly categorizing confirmed discrepancies vs questions that need recruiter follow-up."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Missing Evidence and Consistency Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_missing_evidence(
    intake_data: Dict[str, Any],
    qualification_data: Dict[str, Any],
    evidence_mapping_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Missing Evidence and Consistency Agent."""
    active_agent = agent or create_agent()

    cand_id = intake_data.get("candidate_id", "CAND-UNKNOWN")
    raw_sections = intake_data.get("raw_sections", {})
    mappings = evidence_mapping_data.get("mappings", [])

    task_desc = f"""
Candidate ID: {cand_id}
Raw Resume Sections: {raw_sections}
Criteria Mappings: {mappings}

Audit candidate records for evidence gaps and inconsistencies:
1. Identify missing evidence items (e.g., degree institution omitted, or missing dates).
2. Identify unreliable requirements (criteria that cannot be evaluated reliably from the resume).
3. Detect inconsistencies:
   - Check date continuity: Are employment start dates later than end dates?
   - Check keyword stuffing: Are skills listed without any supporting project or role?
   - Set 'is_confirmed_inconsistency': true for definite chronological or factual errors.
   - Set 'requires_recruiter_clarification': true for ambiguous wording.

Return structured JSON matching:
{{
  "candidate_id": "{cand_id}",
  "missing_evidence_items": ["Missing employment end date", ...],
  "unreliable_requirements": ["Cannot verify high-throughput scale claim", ...],
  "inconsistencies": [
    {{
      "issue_type": "inconsistent_dates",
      "description": "Start date 2025 is after end date 2021 in work history",
      "is_confirmed_inconsistency": true,
      "requires_recruiter_clarification": false,
      "evidence_ids": ["EVD-xxx-EXP1"]
    }}
  ]
}}
"""
    expected_output = "Valid JSON matching MissingEvidenceOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=MissingEvidenceOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
