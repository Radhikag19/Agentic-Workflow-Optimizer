# TraceProfiler Ingestion Report: `recruitment_screening`

**Workflow Name:** `recruitment_screening`  
**Description:** Baseline 8-agent recruitment and resume screening workflow for candidate qualification evaluation.  
**Assigned Key:** `None`  
**Fixed Model:** `openai/gpt-oss-120b`  
**Total Agents:** 8 (Implemented: 8/8)  

## 1. Execution Order

| Step | Agent ID | Status | Implementation File |
|---|---|---|---|
| 1 | `resume_intake_agent` | Implemented | `Agents/recruitment_screening/intake_agent.py` |
| 2 | `qualification_agent` | Implemented | `Agents/recruitment_screening/qualification_agent.py` |
| 3 | `job_criteria_agent` | Implemented | `Agents/recruitment_screening/job_criteria_agent.py` |
| 4 | `evidence_mapping_agent` | Implemented | `Agents/recruitment_screening/evidence_mapping_agent.py` |
| 5 | `experience_assessment_agent` | Implemented | `Agents/recruitment_screening/experience_assessment_agent.py` |
| 6 | `missing_evidence_agent` | Implemented | `Agents/recruitment_screening/missing_evidence_agent.py` |
| 7 | `screening_review_agent` | Implemented | `Agents/recruitment_screening/review_agent.py` |
| 8 | `recruiter_summary_agent` | Implemented | `Agents/recruitment_screening/recruiter_summary_agent.py` |

## 2. Architectural Findings & Discrepancies

- No discrepancies detected between declared manifest and source code orchestration.

## 3. Optimization Hypotheses & Opportunities

- parallel_candidates
- pruning_candidates
- consolidation_candidates

## 4. Telemetry History Artifacts

Found **4** run artifacts recorded in `runs/recruitment_screening/`.

- `runs/recruitment_screening/live_20261023_070437_recrui.json` (Run ID: `live_20261023_070437_recrui`, Status: `success`, Duration: `53.34s`)
- `runs/recruitment_screening/rec_7581eef9.json` (Run ID: `rec_7581eef9`, Status: `success`, Duration: `56.40s`)
- `runs/recruitment_screening/rec_be76e5c5.json` (Run ID: `rec_be76e5c5`, Status: `failed`, Duration: `32.45s`)
- `runs/recruitment_screening/summary.jsonl` (Run ID: `N/A`, Status: `N/A`, Duration: `N/A`)