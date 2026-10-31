"""
Agent 1: Resume Intake Agent for Recruitment and Resume Screening.

Responsibilities:
- Accept synthetic resume and job application identifier.
- Normalize input and identify available resume sections.
- Extract candidate metadata and detect missing or malformed information.
- Identify education, experience, projects, skills, and certifications.
- Never invent qualifications.
"""

from typing import Any, Dict, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import ResumeIntakeOutput
from common.metrics import MetricsTracker

AGENT_ID = "resume_intake_agent"
ROLE = "Resume Intake Specialist"
GOAL = (
    "Accurately normalize raw resume text, identify present and missing sections, "
    "and catalog candidate metadata without ever inventing qualifications."
)
BACKSTORY = (
    "You are a meticulous recruitment intake specialist. Your role is strictly to catalog "
    "the sections and metadata provided in the candidate's resume, noting any formatting gaps "
    "or omitted standard sections with high factual fidelity."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Resume Intake Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_intake(
    resume_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Resume Intake Agent on raw resume data."""
    active_agent = agent or create_agent()

    app_id = resume_data.get("application_id", "APP-UNKNOWN")
    cand_id = resume_data.get("candidate_id", "CAND-UNKNOWN")
    cand_name = resume_data.get("candidate_name", "Unknown Candidate")
    sections = resume_data.get("sections", {})
    source_ids = resume_data.get("source_evidence_ids", [])

    task_desc = f"""
Analyze the candidate application and resume input below:
- Application ID: {app_id}
- Candidate ID: {cand_id}
- Candidate Name: {cand_name}
- Available Resume Sections: {list(sections.keys())}
- Raw Section Content: {sections}
- Source Evidence IDs: {source_ids}

Extract and normalize the intake metadata.
Return a structured JSON object with the following fields:
{{
  "application_id": "{app_id}",
  "candidate_id": "{cand_id}",
  "candidate_name": "{cand_name}",
  "contact_info_present": true or false,
  "sections_identified": ["education", "experience", "skills", ...],
  "missing_sections": ["contact_information", ...],
  "raw_sections": {{ "education": "...", "experience": "..." }},
  "source_evidence_ids": {source_ids},
  "intake_notes": ["Note on any malformed or incomplete sections"]
}}
"""
    expected_output = "Valid JSON matching ResumeIntakeOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=ResumeIntakeOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
