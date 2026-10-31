"""
Workflow Orchestrator: Recruitment and Resume Screening (Workflow 1).

Executes exactly 8 CrewAI agents in sequential baseline order:
1. Resume Intake Agent (intake_agent.py)
2. Qualification Extraction Agent (qualification_agent.py)
3. Job Criteria Agent (job_criteria_agent.py)
4. Evidence Mapping Agent (evidence_mapping_agent.py)
5. Experience and Skills Assessment Agent (experience_assessment_agent.py)
6. Missing Evidence and Consistency Agent (missing_evidence_agent.py)
7. Screening Review Agent (review_agent.py)
8. Recruiter Summary Agent (recruiter_summary_agent.py)
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional
from common.metrics import MetricsTracker
from common.tracing import init_tracing

# Import all 8 agents
from Agents.recruitment_screening import (
    intake_agent,
    qualification_agent,
    job_criteria_agent,
    evidence_mapping_agent,
    experience_assessment_agent,
    missing_evidence_agent,
    review_agent,
    recruiter_summary_agent,
)

WORKFLOW_NAME = "recruitment_screening"
EXPECTED_AGENT_COUNT = 8


def load_sample_recruitment_input(
    candidate_index: int = 0,
    job_index: int = 0,
    data_dir: str = "data/recruitment_screening",
) -> Dict[str, Any]:
    """Load sample resume and job description from local synthetic datasets."""
    resumes_path = os.path.join(data_dir, "resumes.json")
    jobs_path = os.path.join(data_dir, "jobs.json")

    with open(resumes_path, "r", encoding="utf-8") as f:
        resumes = json.load(f)
    with open(jobs_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    resume = resumes[candidate_index % len(resumes)]
    job = jobs[job_index % len(jobs)]

    return {
        "resume": resume,
        "job": job,
    }


class RecruitmentScreeningWorkflow:
    """Manages baseline execution of the 8-agent recruitment screening workflow."""

    def __init__(self, llm: Optional[Any] = None, enable_tracing: bool = True):
        self.llm = llm
        if enable_tracing:
            init_tracing(project_name="awo-recruitment-screening")

        # Instantiate all 8 CrewAI agents
        self.agent_1_intake = intake_agent.create_agent(llm=self.llm)
        self.agent_2_qualification = qualification_agent.create_agent(llm=self.llm)
        self.agent_3_job_criteria = job_criteria_agent.create_agent(llm=self.llm)
        self.agent_4_evidence_mapping = evidence_mapping_agent.create_agent(llm=self.llm)
        self.agent_5_experience = experience_assessment_agent.create_agent(llm=self.llm)
        self.agent_6_missing_evidence = missing_evidence_agent.create_agent(llm=self.llm)
        self.agent_7_review = review_agent.create_agent(llm=self.llm)
        self.agent_8_recruiter_summary = recruiter_summary_agent.create_agent(llm=self.llm)

    def get_agent_count(self) -> int:
        """Return the exact number of agents."""
        return EXPECTED_AGENT_COUNT

    def run(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        runs_base_dir: str = "runs",
    ) -> Dict[str, Any]:
        """
        Execute the 8-agent screening workflow sequentially.
        """
        active_run_id = run_id or f"rec_{uuid.uuid4().hex[:8]}"
        tracker = MetricsTracker(workflow_name=WORKFLOW_NAME, run_id=active_run_id)

        if input_data is None:
            input_data = load_sample_recruitment_input()

        resume_data = input_data.get("resume", {})
        job_data = input_data.get("job", {})

        try:
            # Stage 1: Resume Intake Agent
            intake_res = intake_agent.run_intake(
                resume_data=resume_data,
                agent=self.agent_1_intake,
                tracker=tracker,
            )

            # Stage 2: Qualification Extraction Agent
            qual_res = qualification_agent.run_qualification_extraction(
                intake_output=intake_res,
                agent=self.agent_2_qualification,
                tracker=tracker,
            )

            # Stage 3: Job Criteria Agent
            job_crit_res = job_criteria_agent.run_job_criteria(
                job_data=job_data,
                agent=self.agent_3_job_criteria,
                tracker=tracker,
            )

            # Stage 4: Evidence Mapping Agent
            ev_map_res = evidence_mapping_agent.run_evidence_mapping(
                qualification_data=qual_res,
                job_criteria_data=job_crit_res,
                agent=self.agent_4_evidence_mapping,
                tracker=tracker,
            )

            # Stage 5: Experience and Skills Assessment Agent
            exp_assess_res = experience_assessment_agent.run_experience_assessment(
                qualification_data=qual_res,
                evidence_mapping_data=ev_map_res,
                agent=self.agent_5_experience,
                tracker=tracker,
            )

            # Stage 6: Missing Evidence and Consistency Agent
            missing_ev_res = missing_evidence_agent.run_missing_evidence(
                intake_data=intake_res,
                qualification_data=qual_res,
                evidence_mapping_data=ev_map_res,
                agent=self.agent_6_missing_evidence,
                tracker=tracker,
            )

            # Stage 7: Screening Review Agent
            review_res = review_agent.run_screening_review(
                candidate_id=intake_res.get("candidate_id", ""),
                evidence_mapping_data=ev_map_res,
                assessment_data=exp_assess_res,
                missing_evidence_data=missing_ev_res,
                agent=self.agent_7_review,
                tracker=tracker,
            )

            # Stage 8: Recruiter Summary Agent
            final_summary = recruiter_summary_agent.run_recruiter_summary(
                run_id=active_run_id,
                intake_data=intake_res,
                job_criteria_data=job_crit_res,
                evidence_mapping_data=ev_map_res,
                assessment_data=exp_assess_res,
                missing_evidence_data=missing_ev_res,
                review_data=review_res,
                agent=self.agent_8_recruiter_summary,
                tracker=tracker,
            )

            # Finalize metrics and persist run record
            record = tracker.finalize(
                output=final_summary,
                status="success",
                runs_base_dir=runs_base_dir,
            )
            output_dict = dict(final_summary) if isinstance(final_summary, dict) else (final_summary.model_dump() if hasattr(final_summary, "model_dump") else {})
            output_dict.update({
                "run_id": active_run_id,
                "status": "SUCCESS",
                "duration_seconds": record.total_duration_seconds,
                "total_tokens": record.token_usage.get("total_tokens", 0) if record.token_usage else 0,
                "prompt_tokens": record.token_usage.get("prompt_tokens", 0) if record.token_usage else 0,
                "completion_tokens": record.token_usage.get("completion_tokens", 0) if record.token_usage else 0,
                "estimated_cost_usd": record.estimated_cost_usd or 0.0,
                "agent_executions": [m.model_dump() for m in record.agent_metrics],
                "final_output": final_summary,
            })
            return output_dict

        except Exception as exc:
            tracker.finalize(
                output=None,
                status="failed",
                error=str(exc),
                runs_base_dir=runs_base_dir,
            )
            raise


def run_recruitment_screening_workflow(
    input_data: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute recruitment screening workflow."""
    wf = RecruitmentScreeningWorkflow(enable_tracing=True)
    return wf.run(input_data=input_data, run_id=run_id)

