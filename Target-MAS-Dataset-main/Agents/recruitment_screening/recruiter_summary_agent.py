"""
Agent 8: Recruiter Summary Agent for Recruitment and Resume Screening.

Responsibilities:
- Produce a structured, recruiter-facing screening report.
- Synthesize candidate overview, criteria matrix, strengths, evidence gaps, inconsistencies, and follow-up questions.
- Include a transparent, deterministic match score with documented formula and limitations.
- Strictly never autonomously reject a candidate or make a final hiring decision.
"""

from typing import Any, Dict, List, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import RecruiterSummaryOutput
from common.metrics import MetricsTracker

AGENT_ID = "recruiter_summary_agent"
ROLE = "Executive Recruiter Summary Synthesizer"
GOAL = (
    "Synthesize all screening findings into a structured, transparent recruiter report, "
    "providing actionable follow-up interview questions and deterministic scoring while "
    "explicitly preserving final decision authority for human hiring teams."
)
BACKSTORY = (
    "You are an executive talent partner. You synthesize complex candidate evaluations into clear, "
    "balanced screening summaries. You highlight both verified strengths and critical gaps, providing "
    "tailored interview questions and transparent scoring without ever making autonomous hire/reject rulings."
)


def calculate_deterministic_match_score(mappings: List[Dict[str, Any]]) -> float:
    """
    Calculate deterministic match score (0.0 to 100.0).
    Formula:
    - Mandatory criteria account for 70% of total score.
    - Preferred criteria account for 30% of total score.
    - Points: supported=1.0, partially_supported=0.5, unclear=0.25, unsupported=0.0.
    """
    mand_items = [m for m in mappings if m.get("requirement_type") == "mandatory"]
    pref_items = [m for m in mappings if m.get("requirement_type") == "preferred"]

    def score_items(items):
        if not items:
            return 1.0  # No criteria in category gives neutral full points
        total = 0.0
        for item in items:
            status = item.get("status", "unsupported")
            if status == "supported":
                total += 1.0
            elif status == "partially_supported":
                total += 0.5
            elif status == "unclear":
                total += 0.25
        return total / len(items)

    mand_score = score_items(mand_items)
    pref_score = score_items(pref_items)

    final_score = (mand_score * 70.0) + (pref_score * 30.0)
    return round(final_score, 1)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Recruiter Summary Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_recruiter_summary(
    run_id: str,
    intake_data: Dict[str, Any],
    job_criteria_data: Dict[str, Any],
    evidence_mapping_data: Dict[str, Any],
    assessment_data: Dict[str, Any],
    missing_evidence_data: Dict[str, Any],
    review_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Recruiter Summary Agent."""
    active_agent = agent or create_agent()

    app_id = intake_data.get("application_id", "APP-UNKNOWN")
    cand_id = intake_data.get("candidate_id", "CAND-UNKNOWN")
    cand_name = intake_data.get("candidate_name", "Unknown Candidate")
    job_id = job_criteria_data.get("job_id", "JOB-UNKNOWN")
    job_title = job_criteria_data.get("job_title", "Unknown Job")

    mappings = evidence_mapping_data.get("mappings", [])
    strengths = assessment_data.get("strengths", [])
    gaps = assessment_data.get("experience_gaps", [])
    inconsistencies = [i.get("description", str(i)) for i in missing_evidence_data.get("inconsistencies", [])]
    issues_for_review = review_data.get("issues_requiring_human_review", [])

    # Deterministic arithmetic for match score
    match_score = calculate_deterministic_match_score(mappings)
    score_method = "Deterministic weighted ratio: Mandatory criteria (70%), Preferred criteria (30%). Supported=1.0, Partially Supported=0.5, Unclear=0.25, Unsupported=0.0."
    score_limitations = "Numerical score reflects documented resume keyword and experience overlap only. It does not measure soft skills, motivation, or unlisted capabilities."

    task_desc = f"""
Application ID: {app_id}
Candidate ID: {cand_id}
Candidate Name: {cand_name}
Job ID: {job_id}
Job Title: {job_title}

Screening Findings:
- Criteria Mappings: {mappings}
- Strengths: {strengths}
- Experience Gaps: {gaps}
- Inconsistencies: {inconsistencies}
- Review Issues: {issues_for_review}
- Deterministic Match Score: {match_score}

Generate the final structured Recruiter Summary:
1. Executive overview of the candidate's background and suitability.
2. Provide 3-5 concrete recommended follow-up questions for the recruiter's phone screen.
3. Include the deterministic score ({match_score}) and its documented limitations.
4. Set human_review_required: true.
5. Emphasize that this summary is a screening assistance aid and does NOT make autonomous hiring decisions.

Return structured JSON matching:
{{
  "run_id": "{run_id}",
  "application_id": "{app_id}",
  "candidate_id": "{cand_id}",
  "candidate_name": "{cand_name}",
  "job_id": "{job_id}",
  "job_title": "{job_title}",
  "executive_candidate_overview": "...",
  "criteria_matrix": {mappings},
  "key_strengths": {strengths},
  "evidence_gaps": {gaps},
  "identified_inconsistencies": {inconsistencies},
  "recommended_follow_up_questions": [
    "Could you elaborate on your experience deploying services with Docker and Kubernetes?",
    ...
  ],
  "deterministic_match_score": {match_score},
  "score_calculation_method": "{score_method}",
  "score_limitations": "{score_limitations}",
  "human_review_required": true,
  "non_autonomous_decision_statement": "This is an automated screening assistant summary. It does NOT make autonomous hiring decisions.",
  "validation_warnings": []
}}
"""
    expected_output = "Valid JSON matching RecruiterSummaryOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=RecruiterSummaryOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
