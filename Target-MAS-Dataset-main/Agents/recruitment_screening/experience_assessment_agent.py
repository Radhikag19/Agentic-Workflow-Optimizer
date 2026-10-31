"""
Agent 5: Experience and Skills Assessment Agent for Recruitment and Resume Screening.

Responsibilities:
- Evaluate relevance of experience, projects, skills, and accomplishments to the target role.
- Identify relevant transferable skills where supported by evidence.
- Explain strengths and gaps against stated job requirements.
- Avoid unsupported judgments about candidate competence.
"""

from typing import Any, Dict, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import ExperienceAssessmentOutput
from common.metrics import MetricsTracker

AGENT_ID = "experience_assessment_agent"
ROLE = "Experience Assessment Evaluator"
GOAL = (
    "Evaluate candidate career trajectory, domain depth, and transferable skills "
    "relative to the target position, documenting strengths and concrete gaps without bias."
)
BACKSTORY = (
    "You are a seasoned technical hiring evaluator. You examine work history for depth, "
    "relevance, and practical complexity. You identify transferable competencies grounded in evidence, "
    "avoiding speculative judgments regarding candidate character or unstated capabilities."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Experience Assessment Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_experience_assessment(
    qualification_data: Dict[str, Any],
    evidence_mapping_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Experience and Skills Assessment Agent."""
    active_agent = agent or create_agent()

    cand_id = qualification_data.get("candidate_id", "CAND-UNKNOWN")
    mappings = evidence_mapping_data.get("mappings", [])
    exp_records = qualification_data.get("experience_records", [])

    task_desc = f"""
Candidate ID: {cand_id}
Work Experience Records: {exp_records}
Criteria Fulfillment Mappings: {mappings}

Evaluate candidate experience relevance:
1. Provide a succinct relevance_summary of candidate career history vs target requirements.
2. Identify 2-4 key candidate strengths supported by work evidence.
3. Identify notable experience gaps (e.g., lacking cloud scale, insufficient years in role, or irrelevant industry).
4. Identify transferable skills supported by evidence.
5. List evidence reference IDs cited.

Return structured JSON matching:
{{
  "candidate_id": "{cand_id}",
  "relevance_summary": "...",
  "strengths": ["Strong backend Python architecture experience", ...],
  "experience_gaps": ["No demonstrated experience with AWS Kubernetes", ...],
  "transferable_skills": ["SQL query optimization transferable to PostgreSQL", ...],
  "evidence_references": ["EVD-xxx-EXP1", "EVD-xxx-SKL"]
}}
"""
    expected_output = "Valid JSON matching ExperienceAssessmentOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=ExperienceAssessmentOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
