# TraceProfiler Ingestion Report: `contract_review`

**Workflow Name:** `contract_review`  
**Description:** Baseline 10-agent contract review and compliance workflow with evidence linking.  
**Assigned Key:** `None`  
**Fixed Model:** `openai/gpt-oss-120b`  
**Total Agents:** 10 (Implemented: 10/10)  

## 1. Execution Order

| Step | Agent ID | Status | Implementation File |
|---|---|---|---|
| 1 | `contract_intake_agent` | Implemented | `Agents/contract_review/intake_agent.py` |
| 2 | `clause_extraction_agent` | Implemented | `Agents/contract_review/clause_extraction_agent.py` |
| 3 | `policy_interpretation_agent` | Implemented | `Agents/contract_review/policy_interpretation_agent.py` |
| 4 | `clause_mapping_agent` | Implemented | `Agents/contract_review/clause_mapping_agent.py` |
| 5 | `compliance_deviation_agent` | Implemented | `Agents/contract_review/deviation_agent.py` |
| 6 | `risk_ambiguity_agent` | Implemented | `Agents/contract_review/risk_ambiguity_agent.py` |
| 7 | `cross_clause_consistency_agent` | Implemented | `Agents/contract_review/consistency_agent.py` |
| 8 | `evidence_validation_agent` | Implemented | `Agents/contract_review/evidence_validation_agent.py` |
| 9 | `contract_independent_review_agent` | Implemented | `Agents/contract_review/review_agent.py` |
| 10 | `final_compliance_report_agent` | Implemented | `Agents/contract_review/final_report_agent.py` |

## 2. Architectural Findings & Discrepancies

- No discrepancies detected between declared manifest and source code orchestration.

## 3. Optimization Hypotheses & Opportunities

- potential_parallel_branches
- consolidation_candidates
- pruning_candidates

## 4. Telemetry History Artifacts

Found **3** run artifacts recorded in `runs/contract_review/`.

- `runs/contract_review/ctr_97b38060.json` (Run ID: `ctr_97b38060`, Status: `success`, Duration: `53.71s`)
- `runs/contract_review/live_20261023_070547_contra.json` (Run ID: `live_20261023_070547_contra`, Status: `success`, Duration: `50.40s`)
- `runs/contract_review/summary.jsonl` (Run ID: `N/A`, Status: `N/A`, Duration: `N/A`)