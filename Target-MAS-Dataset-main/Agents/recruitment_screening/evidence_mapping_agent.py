"""
Agent 4: Evidence Mapping Agent for Recruitment and Resume Screening.

Responsibilities:
- Map each job criterion to candidate evidence.
- Classify each criterion as supported, partially supported, unsupported, or unclear.
- Preserve supporting resume excerpts or evidence IDs.
- Distinguish a missing qualification from missing evidence of a qualification.
"""

from typing import Any, Dict, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import EvidenceMappingOutput
from common.metrics import MetricsTracker

AGENT_ID = "evidence_mapping_agent"
ROLE = "Candidate Evidence Mapping Specialist"
GOAL = (
    "Cross-reference every job criterion against extracted candidate qualifications, "
    "classifying fulfillment status and linking evidence IDs while clearly distinguishing "
    "a confirmed missing qualification from mere missing documentation."
)
BACKSTORY = (
    "You are an objective evidence auditor. You never guess or assume. For each criterion, "
    "you look for direct evidence in the candidate's record. If evidence is lacking, you verify "
    "whether the candidate lacks the skill or simply omitted proof in the text."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Evidence Mapping Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_evidence_mapping(
    qualification_data: Dict[str, Any],
    job_criteria_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Evidence Mapping Agent."""
    active_agent = agent or create_agent()

    cand_id = qualification_data.get("candidate_id", "CAND-UNKNOWN")
    job_id = job_criteria_data.get("job_id", "JOB-UNKNOWN")
    mandatory = job_criteria_data.get("mandatory_criteria", [])
    preferred = job_criteria_data.get("preferred_criteria", [])

    all_criteria = mandatory + preferred
    skills = qualification_data.get("technical_skills", [])
    edu = qualification_data.get("education", [])
    exp = qualification_data.get("experience_records", [])

    task_desc = f"""
Candidate ID: {cand_id}
Job ID: {job_id}

Job Criteria to Evaluate:
{all_criteria}

Candidate Extracted Qualifications:
- Technical Skills: {skills}
- Education: {edu}
- Experience: {exp}

For each criterion:
1. Status must be one of: 'supported', 'partially_supported', 'unsupported', 'unclear'.
2. Provide candidate_evidence excerpt or explanation.
3. List matching evidence_ids (e.g. EVD-xxx-SKL, EVD-xxx-EDU, EVD-xxx-EXP1).
4. Set 'is_missing_qualification': true if the candidate explicitly lacks the skill or domain.
5. Set 'is_missing_evidence_only': true if the candidate mentions the domain but provides insufficient verifiable detail.

Return structured JSON matching:
{{
  "candidate_id": "{cand_id}",
  "job_id": "{job_id}",
  "mappings": [
    {{
      "criterion_id": "...",
      "requirement_type": "mandatory",
      "status": "supported",
      "candidate_evidence": "...",
      "evidence_ids": ["EVD-xxx-..."],
      "is_missing_qualification": false,
      "is_missing_evidence_only": false,
      "notes": "..."
    }}
  ],
  "supported_count": 0,
  "unsupported_count": 0,
  "partially_supported_count": 0,
  "unclear_count": 0
}}
"""
    expected_output = "Valid JSON matching EvidenceMappingOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=EvidenceMappingOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
