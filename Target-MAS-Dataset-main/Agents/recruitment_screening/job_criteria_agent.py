"""
Agent 3: Job Criteria Agent for Recruitment and Resume Screening.

Responsibilities:
- Parse the supplied job description.
- Separate mandatory and preferred qualifications, responsibilities, and experience requirements.
- Assign stable criterion IDs.
- Flag ambiguous requirements without inventing new criteria.
"""

from typing import Any, Dict, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import JobCriteriaOutput
from common.metrics import MetricsTracker

AGENT_ID = "job_criteria_agent"
ROLE = "Job Criteria Analyst"
GOAL = (
    "Parse the job description into discrete mandatory and preferred evaluation criteria, "
    "assigning stable criterion IDs and flagging any ambiguous requirements."
)
BACKSTORY = (
    "You are a job specification analyst. You decompose job descriptions into explicit, "
    "verifiable criteria, categorizing them cleanly into mandatory requirements and preferred "
    "qualifications while never inventing unstated prerequisites."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Job Criteria Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_job_criteria(
    job_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Job Criteria Agent on job description."""
    active_agent = agent or create_agent()

    job_id = job_data.get("job_id", "JOB-UNKNOWN")
    job_title = job_data.get("title", "Software Engineer")
    mand_list = job_data.get("mandatory_criteria", [])
    pref_list = job_data.get("preferred_criteria", [])

    task_desc = f"""
Job ID: {job_id}
Job Title: {job_title}
Mandatory Requirements: {mand_list}
Preferred Requirements: {pref_list}

Parse and structure each requirement into distinct evaluation criteria:
- Assign stable criterion_id (e.g. {job_id}-CRIT-M1, {job_id}-CRIT-P1).
- Identify category ('technical_skill', 'education', 'certification', 'experience').
- Mark requirement_type ('mandatory' or 'preferred').
- Flag is_ambiguous (true/false) if requirements contain vague terms like 'rockstar' or unspecified scope.

Return structured JSON matching:
{{
  "job_id": "{job_id}",
  "job_title": "{job_title}",
  "mandatory_criteria": [
    {{
      "criterion_id": "{job_id}-CRIT-M1",
      "category": "technical_skill",
      "requirement_type": "mandatory",
      "description": "...",
      "is_ambiguous": false,
      "ambiguity_notes": null
    }}
  ],
  "preferred_criteria": [
    {{
      "criterion_id": "{job_id}-CRIT-P1",
      "category": "technical_skill",
      "requirement_type": "preferred",
      "description": "...",
      "is_ambiguous": false,
      "ambiguity_notes": null
    }}
  ],
  "total_criteria_count": {len(mand_list) + len(pref_list)},
  "parsing_notes": []
}}
"""
    expected_output = "Valid JSON matching JobCriteriaOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=JobCriteriaOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
