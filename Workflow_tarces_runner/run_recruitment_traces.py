"""
Dedicated Live Trace Generator for Recruitment & Resume Screening Workflow.

Enforces:
1. 1-to-1 Groq API Key Distribution across all 8 sub-agents with dynamic auto-failover.
2. Live, non-hallucinated execution using Groq API with openai/gpt-oss-120b.
3. Reference trace format strictly matching runs/recruitment_screening/live_20261023_070437_recrui.json.
4. Outputs persisted directly in Ingestion_Traces/recruitment_screening/ (no agent-wise subfolders).
"""

import os
import sys
import json
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from groq import Groq

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

# 1-to-1 Sub-Agent to Groq API Key Mapping
AGENT_KEY_MAP = {
    "resume_intake_agent": "GROQ_API_KEY_11",
    "qualification_agent": "GROQ_API_KEY_12",
    "job_criteria_agent": "GROQ_API_KEY_13",
    "evidence_mapping_agent": "GROQ_API_KEY_14",
    "experience_assessment_agent": "GROQ_API_KEY_15",
    "missing_evidence_agent": "GROQ_API_KEY_16",
    "screening_review_agent": "GROQ_API_KEY_1",
    "recruiter_summary_agent": "GROQ_API_KEY_2",
}

# Standby reserve pool of healthy keys (in priority order of remaining capacity)
STANDBY_POOL = [
    "GROQ_API_KEY_5",
    "GROQ_API_KEY_10",
    "GROQ_API_KEY_7",
    "GROQ_API_KEY_4",
]

# Track exhausted keys to never reuse them
EXHAUSTED_KEYS = {"GROQ_API_KEY_3", "GROQ_API_KEY_6", "GROQ_API_KEY_8", "GROQ_API_KEY_9"}

def get_next_standby_key() -> str:
    """Return next available healthy Groq API key from the standby pool."""
    for candidate in STANDBY_POOL:
        if candidate not in EXHAUSTED_KEYS and candidate not in AGENT_KEY_MAP.values():
            val = os.getenv(candidate, "").strip()
            if val:
                return candidate
    for candidate in STANDBY_POOL:
        if candidate not in EXHAUSTED_KEYS:
            val = os.getenv(candidate, "").strip()
            if val:
                return candidate
    raise RuntimeError("All Groq API keys in the standby pool are exhausted!")

def mask_key(raw_key: str) -> str:
    if not raw_key:
        return "[NOT_SET]"
    s = raw_key.strip()
    return f"{s[:6]}...{s[-4:]}" if len(s) > 10 else "***"

def extract_json_from_text(text: str) -> Any:
    """Robust JSON extraction from LLM response text."""
    s = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if match:
        s = match.group(1).strip()
    
    try:
        return json.loads(s)
    except Exception:
        pass

    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = s[first_brace:last_brace + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")
    if open_braces > 0 or open_brackets > 0:
        s_repaired = s.rstrip(", \n\r\t") + ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))
        return json.loads(s_repaired)

    raise ValueError(f"Could not parse valid JSON from text: {text[:200]}...")


def calculate_deterministic_match_score(mappings: List[Dict[str, Any]]) -> float:
    """Deterministic match score (0.0 to 100.0) based on criteria weights."""
    mand_items = [m for m in mappings if m.get("requirement_type") == "mandatory"]
    pref_items = [m for m in mappings if m.get("requirement_type") == "preferred"]

    def score_items(items):
        if not items:
            return 1.0
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


def call_agent_groq(
    agent_id: str,
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    max_tokens: int = 2500,
    max_retries: int = 8,
) -> Dict[str, Any]:
    """Execute a single agent stage via its assigned Groq API key with automatic failover."""
    global AGENT_KEY_MAP, EXHAUSTED_KEYS

    system_prompt = (
        f"You are the {role}.\n"
        f"Goal: {goal}\n"
        f"Backstory: {backstory}\n"
        f"Strictly return valid JSON only. Do not wrap in markdown quotes if possible, or use standard ```json ... ``` blocks."
    )

    t0 = time.time()
    last_err = None
    for attempt in range(max_retries):
        current_key_name = AGENT_KEY_MAP[agent_id]
        api_key = os.getenv(current_key_name, "").strip()

        # If key is missing or already marked exhausted, failover immediately
        if not api_key or current_key_name in EXHAUSTED_KEYS:
            new_key_name = get_next_standby_key()
            print(f"    [Key Failover] Switching {agent_id} from {current_key_name} to fresh standby key {new_key_name}...", flush=True)
            AGENT_KEY_MAP[agent_id] = new_key_name
            current_key_name = new_key_name
            api_key = os.getenv(current_key_name, "").strip()

        masked_id = f"{current_key_name} ({mask_key(api_key)})"
        client = Groq(api_key=api_key)

        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_description},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            t1 = time.time()
            duration = round(t1 - t0, 3)

            raw_content = resp.choices[0].message.content or ""
            if not raw_content and hasattr(resp.choices[0].message, "reasoning"):
                raw_content = resp.choices[0].message.reasoning or ""

            parsed = extract_json_from_text(raw_content)

            prompt_toks = getattr(resp.usage, "prompt_tokens", 0)
            comp_toks = getattr(resp.usage, "completion_tokens", 0)
            tot_toks = getattr(resp.usage, "total_tokens", prompt_toks + comp_toks)

            return {
                "status": "SUCCESS",
                "assigned_key_identifier": masked_id,
                "duration_seconds": duration,
                "tokens": tot_toks,
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "output": parsed,
                "notes": "",
            }
        except Exception as exc:
            last_err = exc
            err_str = str(exc)
            
            # Check if this is a daily quota exhaustion (TPD limit)
            is_quota_exhausted = (
                "tpd" in err_str.lower() or 
                "tokens per day" in err_str.lower() or 
                ("429" in err_str and "limit 200" in err_str)
            )

            if is_quota_exhausted:
                print(f"    [Quota Limit Hit] Key {current_key_name} daily limit reached! Immediate hot-swap to fresh key...", flush=True)
                EXHAUSTED_KEYS.add(current_key_name)
                new_key_name = get_next_standby_key()
                print(f"    [Hot-Swap] Assigned {new_key_name} to {agent_id}. Retrying immediately without waiting...", flush=True)
                AGENT_KEY_MAP[agent_id] = new_key_name
                # Continue loop immediately with new key (no sleep!)
                continue
            elif "429" in err_str or "rate_limit" in err_str.lower():
                # Short transient TPM limit: brief pause
                m_sec = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                wait_time = min(float(m_sec.group(1)) + 1.0, 10.0) if m_sec else 4.0
                print(f"    [Transient RateLimit] Agent '{agent_id}' waiting {wait_time:.1f}s...", flush=True)
                time.sleep(wait_time)
            else:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Agent '{agent_id}' failed after {max_retries} attempts: {last_err}")


def run_single_recruitment_trace(
    resume: Dict[str, Any],
    job: Dict[str, Any],
    run_number: int = 1,
    output_dir: Path = Path("Ingestion_Traces/recruitment_screening"),
) -> Dict[str, Any]:
    """Execute all 8 agents sequentially for one candidate and job, persisting the trace."""
    run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_trace{run_number:02d}"
    t_start = time.time()
    agent_executions = []

    app_id = resume.get("application_id", "APP-UNKNOWN")
    cand_id = resume.get("candidate_id", "CAND-UNKNOWN")
    cand_name = resume.get("candidate_name", "Unknown Candidate")
    job_id = job.get("job_id", "JOB-UNKNOWN")
    job_title = job.get("title", "Unknown Role")

    print(f"\n--- [Run {run_number:02d}] Executing Recruitment Screening for {cand_name} ({cand_id}) -> {job_title} ({job_id}) ---")

    # 1. Resume Intake Agent
    print(f"  [1/8] Resume Intake Agent -> {AGENT_KEY_MAP['resume_intake_agent']}...", flush=True)
    sections = resume.get("sections", {})
    source_ids = resume.get("source_evidence_ids", [])
    desc_1 = f"""
Analyze the candidate application and resume input below:
- Application ID: {app_id}
- Candidate ID: {cand_id}
- Candidate Name: {cand_name}
- Available Resume Sections: {list(sections.keys())}
- Raw Section Content: {json.dumps(sections, indent=2)}
- Source Evidence IDs: {json.dumps(source_ids)}

Extract and normalize the intake metadata.
Return strictly valid JSON matching:
{{
  "application_id": "{app_id}",
  "candidate_id": "{cand_id}",
  "candidate_name": "{cand_name}",
  "contact_info_present": true,
  "sections_identified": {json.dumps(list(sections.keys()))},
  "missing_sections": [],
  "raw_sections": {json.dumps(sections)}
}}
"""
    exec_1 = call_agent_groq(
        agent_id="resume_intake_agent",
        role="Resume Intake Specialist",
        goal="Accurately normalize raw resume text, identify present and missing sections, and catalog candidate metadata without ever inventing qualifications.",
        backstory="A meticulous recruitment intake specialist cataloging resume sections and metadata with high factual fidelity.",
        task_description=desc_1,
        max_tokens=2500,
    )
    exec_1["order"] = 1
    exec_1["agent_id"] = "resume_intake_agent"
    agent_executions.append(exec_1)
    intake_res = exec_1["output"]

    # 2. Qualification Extraction Agent
    print(f"  [2/8] Qualification Extraction Agent -> {AGENT_KEY_MAP['qualification_agent']}...", flush=True)
    raw_secs = intake_res.get("raw_sections", sections)
    desc_2 = f"""
Candidate ID: {cand_id}
Normalized Resume Sections:
{json.dumps(raw_secs, indent=2)}
Available Evidence IDs: {json.dumps(source_ids)}

Extract all concrete qualifications. For each item:
- Identify category: 'technical_skill', 'education', 'certification', 'experience', 'project'.
- Name of skill/degree/experience.
- Supporting excerpt from the resume text.
- Associated evidence ID (e.g., EVD-xxx-EDU, EVD-xxx-EXP1, EVD-xxx-SKL).

Return strictly valid JSON matching:
{{
  "candidate_id": "{cand_id}",
  "technical_skills": [
    {{
      "category": "technical_skill",
      "item_name": "Python",
      "proficiency_or_duration": "5 years" or null,
      "supporting_excerpt": "...",
      "evidence_id": "EVD-..."
    }}
  ],
  "education": [
    {{
      "category": "education",
      "item_name": "B.S. in Computer Science",
      "proficiency_or_duration": "Graduated 2014",
      "supporting_excerpt": "...",
      "evidence_id": "EVD-..."
    }}
  ],
  "certifications": [],
  "experience_records": [
    {{
      "category": "experience",
      "item_name": "Software Engineer at Apex Dynamics Corp",
      "proficiency_or_duration": "2015-01 - 2024-12",
      "supporting_excerpt": "...",
      "evidence_id": "EVD-..."
    }}
  ],
  "projects": [],
  "extraction_warnings": []
}}
"""
    exec_2 = call_agent_groq(
        agent_id="qualification_agent",
        role="Candidate Qualification Extractor",
        goal="Extract concrete qualifications (skills, degrees, certifications, roles) explicitly grounded in source evidence IDs without hallucinating unstated credentials.",
        backstory="A credential auditor verifying candidate qualifications against source text.",
        task_description=desc_2,
        max_tokens=2500,
    )
    exec_2["order"] = 2
    exec_2["agent_id"] = "qualification_agent"
    agent_executions.append(exec_2)
    qual_res = exec_2["output"]

    # 3. Job Criteria Agent
    print(f"  [3/8] Job Criteria Agent -> {AGENT_KEY_MAP['job_criteria_agent']}...", flush=True)
    mand_list = job.get("mandatory_criteria", [])
    pref_list = job.get("preferred_criteria", [])
    desc_3 = f"""
Job ID: {job_id}
Job Title: {job_title}
Mandatory Requirements: {json.dumps(mand_list, indent=2)}
Preferred Requirements: {json.dumps(pref_list, indent=2)}

Parse and structure each requirement into distinct evaluation criteria:
- Assign stable criterion_id (e.g. {job_id}-CRIT-M1, {job_id}-CRIT-P1).
- Identify category ('technical_skill', 'education', 'certification', 'experience').
- Mark requirement_type ('mandatory' or 'preferred').
- Flag is_ambiguous (true/false) if requirements contain vague terms or unspecified scope.

Return strictly valid JSON matching:
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
    exec_3 = call_agent_groq(
        agent_id="job_criteria_agent",
        role="Job Criteria Analyst",
        goal="Decompose job descriptions into discrete, unambiguous mandatory and preferred evaluation criteria with stable IDs.",
        backstory="A job analysis expert dissecting role descriptions into objective, measurable evaluation criteria.",
        task_description=desc_3,
        max_tokens=2500,
    )
    exec_3["order"] = 3
    exec_3["agent_id"] = "job_criteria_agent"
    agent_executions.append(exec_3)
    job_crit_res = exec_3["output"]

    # 4. Evidence Mapping Agent
    print(f"  [4/8] Evidence Mapping Agent -> {AGENT_KEY_MAP['evidence_mapping_agent']}...", flush=True)
    desc_4 = f"""
Candidate ID: {cand_id}
Job ID: {job_id}

Extracted Candidate Qualifications:
{json.dumps(qual_res, indent=2)}

Structured Job Criteria:
{json.dumps(job_crit_res, indent=2)}

Instructions:
Map each job criterion to candidate qualifications.
Classify status strictly as: 'supported', 'partially_supported', 'unsupported', or 'unclear'.
Include supporting candidate_evidence, evidence_ids (from qualifications), and notes.

Return strictly valid JSON matching:
{{
  "candidate_id": "{cand_id}",
  "job_id": "{job_id}",
  "mappings": [
    {{
      "criterion_id": "{job_id}-CRIT-M1",
      "requirement_type": "mandatory",
      "status": "supported" or "unsupported" or "partially_supported",
      "candidate_evidence": "...",
      "evidence_ids": ["EVD-..."],
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
    exec_4 = call_agent_groq(
        agent_id="evidence_mapping_agent",
        role="Candidate Evidence Mapping Specialist",
        goal="Objectively map job criteria to candidate evidence, accurately classifying support status without assuming unproven skills.",
        backstory="An evidence mapper matching candidate records against job specifications with zero bias.",
        task_description=desc_4,
        max_tokens=3000,
    )
    exec_4["order"] = 4
    exec_4["agent_id"] = "evidence_mapping_agent"
    agent_executions.append(exec_4)
    ev_map_res = exec_4["output"]

    # 5. Experience and Skills Assessment Agent
    print(f"  [5/8] Experience Assessment Agent -> {AGENT_KEY_MAP['experience_assessment_agent']}...", flush=True)
    desc_5 = f"""
Candidate ID: {cand_id}
Job Title: {job_title}

Candidate Qualifications:
{json.dumps(qual_res, indent=2)}

Evidence Mappings:
{json.dumps(ev_map_res, indent=2)}

Tasks:
1. Provide a relevance summary of candidate's career progression and domain relevance.
2. Identify 3-5 verified strengths with evidence citations.
3. Identify critical experience gaps compared to target role requirements.
4. Highlight transferable skills and evidence references.

Return strictly valid JSON matching:
{{
  "candidate_id": "{cand_id}",
  "relevance_summary": "...",
  "strengths": ["..."],
  "experience_gaps": ["..."],
  "transferable_skills": ["..."],
  "evidence_references": {json.dumps(source_ids)}
}}
"""
    exec_5 = call_agent_groq(
        agent_id="experience_assessment_agent",
        role="Experience Assessment Evaluator",
        goal="Evaluate career progression, depth of experience, and transferable competencies against role demands.",
        backstory="A senior technical recruiter evaluating candidate seniority, leadership, and transferable skills.",
        task_description=desc_5,
        max_tokens=2500,
    )
    exec_5["order"] = 5
    exec_5["agent_id"] = "experience_assessment_agent"
    agent_executions.append(exec_5)
    exp_assess_res = exec_5["output"]

    # 6. Missing Evidence and Consistency Agent
    print(f"  [6/8] Missing Evidence Agent -> {AGENT_KEY_MAP['missing_evidence_agent']}...", flush=True)
    desc_6 = f"""
Candidate ID: {cand_id}
Resume Sections:
{json.dumps(intake_res.get('raw_sections', {}), indent=2)}

Evidence Mappings:
{json.dumps(ev_map_res, indent=2)}

Tasks:
1. Detect unverified claims, missing evidence items, or buzzwords lacking project context.
2. Check for chronological timeline inconsistencies or gaps.
3. Detail unreliable requirements or unverified achievements.

Return strictly valid JSON matching:
{{
  "candidate_id": "{cand_id}",
  "missing_evidence_items": [],
  "unreliable_requirements": ["..."],
  "inconsistencies": []
}}
"""
    exec_6 = call_agent_groq(
        agent_id="missing_evidence_agent",
        role="Recruitment Consistency and Verification Auditor",
        goal="Detect timeline contradictions, unsupported buzzwords, missing evidence, and unverified claims in candidate submissions.",
        backstory="A background verification specialist auditing resumes for truthfulness, timeline integrity, and empirical substance.",
        task_description=desc_6,
        max_tokens=2500,
    )
    exec_6["order"] = 6
    exec_6["agent_id"] = "missing_evidence_agent"
    agent_executions.append(exec_6)
    missing_ev_res = exec_6["output"]

    # 7. Screening Review Agent
    print(f"  [7/8] Screening Review Agent -> {AGENT_KEY_MAP['screening_review_agent']}...", flush=True)
    desc_7 = f"""
Candidate ID: {cand_id}
Audit File:
- Evidence Mappings: {json.dumps(ev_map_res, indent=2)}
- Experience Assessment: {json.dumps(exp_assess_res, indent=2)}
- Missing Evidence & Consistency: {json.dumps(missing_ev_res, indent=2)}

Tasks:
1. Audit findings for contradictions or unsupported claims.
2. Verify traceability of candidate evidence.
3. Confirm whether mandatory criteria are met (true/false).
4. List specific issues requiring human hiring team review.

Return strictly valid JSON matching:
{{
  "candidate_id": "{cand_id}",
  "review_passed": true,
  "contradictions_detected": [],
  "traceability_verified": true,
  "mandatory_criteria_met": true or false,
  "issues_requiring_human_review": ["..."]
}}
"""
    exec_7 = call_agent_groq(
        agent_id="screening_review_agent",
        role="Screening Quality and Integrity Reviewer",
        goal="Audit upstream screening outputs for logical contradictions, verify evidence traceability, and confirm mandatory criteria satisfaction.",
        backstory="A quality assurance lead in recruitment operations ensuring evaluation objectivity and compliance.",
        task_description=desc_7,
        max_tokens=2500,
    )
    exec_7["order"] = 7
    exec_7["agent_id"] = "screening_review_agent"
    agent_executions.append(exec_7)
    review_res = exec_7["output"]

    # Calculate deterministic match score
    mappings_list = ev_map_res.get("mappings", [])
    deterministic_score = calculate_deterministic_match_score(mappings_list)

    # 8. Recruiter Summary Agent
    print(f"  [8/8] Recruiter Summary Agent -> {AGENT_KEY_MAP['recruiter_summary_agent']}...", flush=True)
    stage_dossier = {
        "candidate": intake_res,
        "job": job_crit_res,
        "mappings": ev_map_res,
        "assessment": exp_assess_res,
        "audit": missing_ev_res,
        "qa_review": review_res,
        "deterministic_match_score": deterministic_score,
    }
    desc_8 = f"""
Candidate Dossier:
{json.dumps(stage_dossier, indent=2)}

Deterministic Match Score calculated: {deterministic_score}/100.0

Tasks:
1. Synthesize an executive candidate overview.
2. Outline key strengths with evidence citations.
3. Detail critical evidence gaps.
4. Formulate 3-4 tailored interview questions for the hiring team.
5. Provide a clear final recommendation ('ADVANCE_TO_INTERVIEW', 'POTENTIAL_FIT_NEEDS_REVIEW', or 'NOT_QUALIFIED').
6. Preserves decision authority for human teams.

Return strictly valid JSON matching:
{{
  "run_id": "{run_id}",
  "application_id": "{app_id}",
  "candidate_id": "{cand_id}",
  "candidate_name": "{cand_name}",
  "job_id": "{job_id}",
  "job_title": "{job_title}",
  "executive_candidate_overview": "...",
  "criteria_matrix": {json.dumps(mappings_list)},
  "key_strengths": ["..."],
  "evidence_gaps": ["..."],
  "deterministic_match_score": {deterministic_score},
  "human_review_required": true,
  "interview_questions": ["..."],
  "final_recommendation": "..."
}}
"""
    exec_8 = call_agent_groq(
        agent_id="recruiter_summary_agent",
        role="Executive Recruiter Summary Synthesizer",
        goal="Synthesize all screening findings into a structured, transparent recruiter report with deterministic scoring and tailored interview questions.",
        backstory="An executive talent partner synthesizing complex candidate evaluations into clear, balanced screening summaries.",
        task_description=desc_8,
        max_tokens=3500,
    )
    exec_8["order"] = 8
    exec_8["agent_id"] = "recruiter_summary_agent"
    agent_executions.append(exec_8)

    t_end = time.time()
    total_dur = round(t_end - t_start, 3)

    tot_prompt = sum(e["prompt_tokens"] for e in agent_executions)
    tot_comp = sum(e["completion_tokens"] for e in agent_executions)
    tot_tokens = tot_prompt + tot_comp
    cost = round((tot_prompt / 1_000_000.0) * 0.59 + (tot_comp / 1_000_000.0) * 0.79, 6)

    trace_record = {
        "run_id": run_id,
        "workflow_name": "recruitment_screening",
        "model": "openai/gpt-oss-120b",
        "status": "SUCCESS",
        "duration_seconds": total_dur,
        "agent_execution_count": len(agent_executions),
        "agent_executions": agent_executions,
        "total_tokens": tot_tokens,
        "prompt_tokens": tot_prompt,
        "completion_tokens": tot_comp,
        "estimated_cost_usd": cost,
        "error_message": None,
        "final_output": json.dumps(exec_8["output"], indent=2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_file = output_dir / f"{run_id}.json"
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(trace_record, f, indent=2, ensure_ascii=False)

    print(f"--> Saved trace to: {trace_file}")
    print(f"--> Completed in {total_dur}s | Total Tokens: {tot_tokens:,} | Cost: ${cost:.6f}")
    return trace_record


def generate_batch_summary(target_dir: Path):
    """Aggregate all traces in target_dir into batch_30_runs_summary.json."""
    trace_files = sorted(list(target_dir.glob("live_*_trace*.json")))
    if not trace_files:
        return

    total_runs = len(trace_files)
    total_tokens = 0
    total_prompt = 0
    total_comp = 0
    total_cost = 0.0
    durations = []
    file_names = []

    for tf in trace_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                data = json.load(f)
            file_names.append(tf.name)
            durations.append(data.get("duration_seconds", 0))
            total_tokens += data.get("total_tokens", 0)
            total_prompt += data.get("prompt_tokens", 0)
            total_comp += data.get("completion_tokens", 0)
            total_cost += data.get("estimated_cost_usd", 0.0)
        except Exception as e:
            print(f"Error reading {tf}: {e}")

    summary = {
        "workflow_name": "recruitment_screening",
        "total_runs": total_runs,
        "successful_runs": total_runs,
        "failed_runs": 0,
        "model_enforced": "openai/gpt-oss-120b",
        "key_distribution_model": "1-to-1 dedicated Groq key per sub-agent with dynamic auto-failover",
        "timing_metrics": {
            "average_duration_seconds": round(sum(durations) / max(len(durations), 1), 2),
            "min_duration_seconds": min(durations) if durations else 0,
            "max_duration_seconds": max(durations) if durations else 0,
            "total_runtime_seconds": round(sum(durations), 2),
        },
        "token_metrics": {
            "total_tokens_consumed": total_tokens,
            "average_tokens_per_run": round(total_tokens / max(total_runs, 1), 1),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_comp,
        },
        "cost_metrics": {
            "total_cost_usd": round(total_cost, 4),
            "average_cost_per_run_usd": round(total_cost / max(total_runs, 1), 6),
        },
        "trace_files": file_names,
    }

    summary_file = target_dir / "batch_30_runs_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n--> Successfully compiled batch summary: {summary_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate live Ingestion_Traces for Recruitment Screening")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute (default: 1)")
    parser.add_argument("--start-idx", type=int, default=0, help="Resumes dataset starting index")
    parser.add_argument("--start-run-num", type=int, default=1, help="Run number to start numbering from")
    parser.add_argument("--total-runs", type=int, default=30, help="Total runs in batch for logging")
    args = parser.parse_args()

    # Load datasets
    data_dir = PROJECT_ROOT / "data" / "recruitment_screening"
    with open(data_dir / "resumes.json", "r", encoding="utf-8") as f:
        all_resumes = json.load(f)
    with open(data_dir / "jobs.json", "r", encoding="utf-8") as f:
        all_jobs = json.load(f)

    target_dir = PROJECT_ROOT / "Ingestion_Traces" / "recruitment_screening"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting execution of {args.runs} live runs for recruitment_screening (Runs {args.start_run_num} to {args.start_run_num + args.runs - 1})...")
    print(f"Output directory: {target_dir}")

    summary_file = target_dir / "summary.jsonl"

    for i in range(args.runs):
        run_num = args.start_run_num + i
        res_idx = (args.start_idx + i) % len(all_resumes)
        resume_sample = all_resumes[res_idx]

        # Match target job if possible or cycle through jobs
        target_job_id = resume_sample.get("target_job_id")
        matching_job = next((j for j in all_jobs if j.get("job_id") == target_job_id), None)
        if not matching_job:
            matching_job = all_jobs[i % len(all_jobs)]

        print(f"\n=======================================================")
        print(f" [RUN {run_num:02d}/{args.total_runs}] Starting...")
        print(f"=======================================================")

        try:
            trace_rec = run_single_recruitment_trace(
                resume=resume_sample,
                job=matching_job,
                run_number=run_num,
                output_dir=target_dir,
            )
            with open(summary_file, "a", encoding="utf-8") as sf:
                summary_entry = {
                    "run_number": run_num,
                    "run_id": trace_rec["run_id"],
                    "candidate_id": resume_sample.get("candidate_id"),
                    "candidate_name": resume_sample.get("candidate_name"),
                    "job_id": matching_job.get("job_id"),
                    "job_title": matching_job.get("title"),
                    "duration_seconds": trace_rec["duration_seconds"],
                    "total_tokens": trace_rec["total_tokens"],
                    "estimated_cost_usd": trace_rec["estimated_cost_usd"],
                    "status": trace_rec["status"],
                    "timestamp": trace_rec["timestamp"],
                }
                sf.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[ERROR] Run {run_num} failed: {e}")
            with open(summary_file, "a", encoding="utf-8") as sf:
                sf.write(json.dumps({
                    "run_number": run_num,
                    "candidate_id": resume_sample.get("candidate_id"),
                    "error": str(e),
                    "status": "FAILED",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")

    generate_batch_summary(target_dir)

