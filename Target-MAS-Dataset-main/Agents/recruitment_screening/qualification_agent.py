"""
Agent 2: Qualification Extraction Agent for Recruitment and Resume Screening.

Responsibilities:
- Extract technical skills, education, certifications, projects, experience, accomplishments.
- Preserve supporting resume excerpts and evidence IDs.
- Record proficiency or duration only when supported by the resume.
- Avoid repeating normalization completed by the intake agent.
"""

from typing import Any, Dict, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import QualificationExtractionOutput
from common.metrics import MetricsTracker

AGENT_ID = "qualification_agent"
ROLE = "Candidate Qualification Extractor"
GOAL = (
    "Extract factual technical skills, degrees, certifications, and work experience "
    "from the normalized resume sections, linking each item to exact excerpts and evidence IDs."
)
BACKSTORY = (
    "You are an expert technical resume parser. You extract qualifications with high precision, "
    "faithfully recording evidence IDs and verbatim excerpts. You never infer unstated proficiency."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Qualification Extraction Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_qualification_extraction(
    intake_output: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Qualification Extraction Agent."""
    active_agent = agent or create_agent()

    cand_id = intake_output.get("candidate_id", "CAND-UNKNOWN")
    raw_sections = intake_output.get("raw_sections", {})
    evidence_ids = intake_output.get("source_evidence_ids", [])

    task_desc = f"""
Candidate ID: {cand_id}
Normalized Resume Sections:
{raw_sections}
Available Evidence IDs: {evidence_ids}

Extract all concrete qualifications. For each item:
- Identify category: 'technical_skill', 'education', 'certification', 'experience', 'project'.
- Name of skill/degree/experience.
- Supporting excerpt from the resume text.
- Associated evidence ID (e.g., EVD-xxx-EDU, EVD-xxx-EXP1, EVD-xxx-SKL).

Return a structured JSON object matching:
{{
  "candidate_id": "{cand_id}",
  "technical_skills": [
    {{
      "category": "technical_skill",
      "item_name": "Python",
      "proficiency_or_duration": "5 years" or null,
      "supporting_excerpt": "Designed scalable services in Python...",
      "evidence_id": "EVD-xxx-SKL"
    }}
  ],
  "education": [
    {{
      "category": "education",
      "item_name": "B.S. in Computer Science",
      "proficiency_or_duration": "Graduated 2020",
      "supporting_excerpt": "B.S. in Computer Science from State University",
      "evidence_id": "EVD-xxx-EDU"
    }}
  ],
  "certifications": [],
  "experience_records": [
    {{
      "category": "experience",
      "item_name": "Software Engineer at Apex Dynamics Corp",
      "proficiency_or_duration": "2018 - 2024",
      "supporting_excerpt": "Designed and deployed scalable services...",
      "evidence_id": "EVD-xxx-EXP1"
    }}
  ],
  "projects": [],
  "extraction_warnings": []
}}
"""
    expected_output = "Valid JSON matching QualificationExtractionOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=QualificationExtractionOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
