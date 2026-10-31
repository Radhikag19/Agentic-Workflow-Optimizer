"""
Unit Tests for Workflow 1: Recruitment and Resume Screening (8 Agents).

Tests execute with mocks to avoid consuming Groq API credits.
Smoke test is marked with @pytest.mark.smoke for live testing.
"""

import json
import os
from unittest.mock import MagicMock, patch
import pytest

from common.validation import (
    ResumeIntakeOutput,
    QualificationExtractionOutput,
    JobCriteriaOutput,
    EvidenceMappingOutput,
    ExperienceAssessmentOutput,
    MissingEvidenceOutput,
    ScreeningReviewOutput,
    RecruiterSummaryOutput,
)
from Agents.recruitment_screening.workflow import (
    RecruitmentScreeningWorkflow,
    load_sample_recruitment_input,
    EXPECTED_AGENT_COUNT,
)
from Agents.recruitment_screening.recruiter_summary_agent import calculate_deterministic_match_score


def test_recruitment_agent_count():
    """Verify workflow implements exactly 8 agents."""
    workflow = RecruitmentScreeningWorkflow(enable_tracing=False)
    assert workflow.get_agent_count() == 8
    assert EXPECTED_AGENT_COUNT == 8


def test_recruitment_datasets():
    """Verify local synthetic recruitment datasets exist and adhere to schema."""
    jobs_path = "data/recruitment_screening/jobs.json"
    resumes_path = "data/recruitment_screening/resumes.json"

    assert os.path.exists(jobs_path)
    assert os.path.exists(resumes_path)

    with open(jobs_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    with open(resumes_path, "r", encoding="utf-8") as f:
        resumes = json.load(f)

    assert len(jobs) >= 10
    assert len(resumes) >= 200

    # Verify resume structure
    sample_res = resumes[0]
    assert "candidate_id" in sample_res
    assert "application_id" in sample_res
    assert "sections" in sample_res
    assert "source_evidence_ids" in sample_res
    assert len(sample_res["source_evidence_ids"]) > 0


def test_deterministic_match_score_calculation():
    """Verify deterministic match score calculation logic."""
    # Test case 1: all mandatory supported, all preferred supported -> 100.0
    mappings_full = [
        {"requirement_type": "mandatory", "status": "supported"},
        {"requirement_type": "mandatory", "status": "supported"},
        {"requirement_type": "preferred", "status": "supported"},
    ]
    score_full = calculate_deterministic_match_score(mappings_full)
    assert score_full == 100.0

    # Test case 2: mandatory half supported, preferred unsupported
    mappings_partial = [
        {"requirement_type": "mandatory", "status": "supported"},
        {"requirement_type": "mandatory", "status": "unsupported"},
        {"requirement_type": "preferred", "status": "unsupported"},
    ]
    score_partial = calculate_deterministic_match_score(mappings_partial)
    # (0.5 * 70) + (0.0 * 30) = 35.0
    assert score_partial == 35.0


def test_recruitment_workflow_mock_execution(tmp_path):
    """
    Test end-to-end execution of the 8-agent recruitment workflow using mocked agent outputs.
    Ensures data contracts and structured schemas are honored.
    """
    sample_input = load_sample_recruitment_input()

    # Create mock return values for each agent matching their Pydantic schemas
    mock_intake = {
        "application_id": "APP-001",
        "candidate_id": "CAND-001",
        "candidate_name": "Alex Smith",
        "contact_info_present": True,
        "sections_identified": ["education", "experience", "skills"],
        "missing_sections": [],
        "raw_sections": {"education": "B.S. in CS", "experience": "Software Engineer at Apex"},
        "source_evidence_ids": ["EVD-001-EDU", "EVD-001-EXP1"],
        "intake_notes": [],
    }

    mock_qual = {
        "candidate_id": "CAND-001",
        "technical_skills": [
            {
                "category": "technical_skill",
                "item_name": "Python",
                "proficiency_or_duration": "5 years",
                "supporting_excerpt": "Designed scalable services in Python",
                "evidence_id": "EVD-001-SKL",
            }
        ],
        "education": [
            {
                "category": "education",
                "item_name": "B.S. in Computer Science",
                "proficiency_or_duration": "2020",
                "supporting_excerpt": "B.S. in Computer Science from State University",
                "evidence_id": "EVD-001-EDU",
            }
        ],
        "certifications": [],
        "experience_records": [
            {
                "category": "experience",
                "item_name": "Software Engineer",
                "proficiency_or_duration": "2018 - 2024",
                "supporting_excerpt": "Designed scalable services",
                "evidence_id": "EVD-001-EXP1",
            }
        ],
        "projects": [],
        "extraction_warnings": [],
    }

    mock_crit = {
        "job_id": "JOB-01",
        "job_title": "Senior Cloud Backend Engineer",
        "mandatory_criteria": [
            {
                "criterion_id": "JOB-01-CRIT-M1",
                "category": "technical_skill",
                "requirement_type": "mandatory",
                "description": "Python or Go backend experience",
                "is_ambiguous": False,
                "ambiguity_notes": None,
            }
        ],
        "preferred_criteria": [
            {
                "criterion_id": "JOB-01-CRIT-P1",
                "category": "technical_skill",
                "requirement_type": "preferred",
                "description": "gRPC event streaming",
                "is_ambiguous": False,
                "ambiguity_notes": None,
            }
        ],
        "total_criteria_count": 2,
        "parsing_notes": [],
    }

    mock_map = {
        "candidate_id": "CAND-001",
        "job_id": "JOB-01",
        "mappings": [
            {
                "criterion_id": "JOB-01-CRIT-M1",
                "requirement_type": "mandatory",
                "status": "supported",
                "candidate_evidence": "5 years Python development",
                "evidence_ids": ["EVD-001-SKL"],
                "is_missing_qualification": False,
                "is_missing_evidence_only": False,
                "notes": "Direct match",
            },
            {
                "criterion_id": "JOB-01-CRIT-P1",
                "requirement_type": "preferred",
                "status": "unsupported",
                "candidate_evidence": None,
                "evidence_ids": [],
                "is_missing_qualification": True,
                "is_missing_evidence_only": False,
                "notes": "No gRPC mention",
            },
        ],
        "supported_count": 1,
        "unsupported_count": 1,
        "partially_supported_count": 0,
        "unclear_count": 0,
    }

    mock_exp = {
        "candidate_id": "CAND-001",
        "relevance_summary": "Strong backend Python match with solid systems tenure.",
        "strengths": ["Python expertise", "CS degree"],
        "experience_gaps": ["Lacks gRPC streaming"],
        "transferable_skills": ["SQL query optimization"],
        "evidence_references": ["EVD-001-SKL", "EVD-001-EXP1"],
    }

    mock_missing = {
        "candidate_id": "CAND-001",
        "missing_evidence_items": [],
        "unreliable_requirements": [],
        "inconsistencies": [],
    }

    mock_review = {
        "candidate_id": "CAND-001",
        "review_passed": True,
        "contradictions_detected": [],
        "traceability_verified": True,
        "mandatory_criteria_met": True,
        "issues_requiring_human_review": ["Confirm gRPC familiarity in phone screen"],
    }

    mock_summary = {
        "run_id": "rec_test_001",
        "application_id": "APP-001",
        "candidate_id": "CAND-001",
        "candidate_name": "Alex Smith",
        "job_id": "JOB-01",
        "job_title": "Senior Cloud Backend Engineer",
        "executive_candidate_overview": "Candidate demonstrates required 5+ years Python backend experience.",
        "criteria_matrix": mock_map["mappings"],
        "key_strengths": ["Python systems design", "Production scalability"],
        "evidence_gaps": ["No explicit gRPC background"],
        "identified_inconsistencies": [],
        "recommended_follow_up_questions": ["How have you handled inter-service messaging without gRPC?"],
        "deterministic_match_score": 70.0,
        "score_calculation_method": "Deterministic weighted ratio",
        "score_limitations": "Measures stated resume text only",
        "human_review_required": True,
        "non_autonomous_decision_statement": "This is an automated screening assistant summary. It does NOT make autonomous hiring decisions.",
        "validation_warnings": [],
    }

    # Patch run_agent_task to return mocked schemas sequentially
    returns = [
        mock_intake,
        mock_qual,
        mock_crit,
        mock_map,
        mock_exp,
        mock_missing,
        mock_review,
        mock_summary,
    ]
    # Use set_mock_task_handler to return mocked schemas sequentially
    from common.agent_runner import set_mock_task_handler
    iterator = iter(returns)
    set_mock_task_handler(lambda agent_id, desc: next(iterator))
    try:
        workflow = RecruitmentScreeningWorkflow(enable_tracing=False)
        result = workflow.run(input_data=sample_input, run_id="rec_test_001", runs_base_dir=str(tmp_path))
    finally:
        set_mock_task_handler(None)

    assert result["run_id"] == "rec_test_001"
    assert result["candidate_id"] == "CAND-001"
    assert result["deterministic_match_score"] == 70.0
    assert result["human_review_required"] is True
    assert "autonomous hiring decisions" in result["non_autonomous_decision_statement"]


@pytest.mark.smoke
def test_recruitment_live_smoke():
    """Live smoke test calling Groq API (requires valid GROQ_API_KEY)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not configured; skipping live smoke test")

    workflow = RecruitmentScreeningWorkflow(enable_tracing=True)
    result = workflow.run()
    assert result is not None
    assert "deterministic_match_score" in result
    assert result["human_review_required"] is True
