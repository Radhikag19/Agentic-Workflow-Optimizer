# TraceProfiler Ingestion Report: `invoice_processing`

**Workflow Name:** `invoice_processing`  
**Description:** Baseline Accounts Payable Invoice Processing Multi-Agent Workflow  
**Assigned Key:** `GROQ_API_KEY_1`  
**Fixed Model:** `openai/gpt-oss-120b`  
**Total Agents:** 8 (Implemented: 8/8)  

## 1. Execution Order

| Step | Agent ID | Status | Implementation File |
|---|---|---|---|
| 1 | `invoice_extraction_agent` | Implemented | `Agents/invoice_processing/invoice_extraction_agent.py` |
| 2 | `invoice_validation_agent` | Implemented | `Agents/invoice_processing/invoice_validation_agent.py` |
| 3 | `po_matching_agent` | Implemented | `Agents/invoice_processing/po_matching_agent.py` |
| 4 | `policy_compliance_agent` | Implemented | `Agents/invoice_processing/policy_compliance_agent.py` |
| 5 | `duplicate_detection_agent` | Implemented | `Agents/invoice_processing/duplicate_detection_agent.py` |
| 6 | `exception_analysis_agent` | Implemented | `Agents/invoice_processing/exception_analysis_agent.py` |
| 7 | `approval_recommendation_agent` | Implemented | `Agents/invoice_processing/approval_recommendation_agent.py` |
| 8 | `final_review_agent_invoice` | Implemented | `Agents/invoice_processing/final_review_agent.py` |

## 2. Architectural Findings & Discrepancies

- **[WARNING] MULTIPLE_KICKOFF_CALLS**: Workflow invokes 8 sequential Crew kickoff calls instead of single execution graph.

## 3. Optimization Hypotheses & Opportunities

- Prune exception_analysis_agent if validation and compliance provide sufficient evidence
- Consolidate invoice_validation_agent and exception_analysis_agent
- Parallelize invoice_validation, po_matching, policy_compliance, and duplicate_detection after extraction
- Optimize prompts to eliminate repeated context across agents

## 4. Telemetry History Artifacts

Found **4** run artifacts recorded in `runs/invoice_processing/`.

- `runs/invoice_processing/invoice_processing_20261020_131708_80802c.json` (Run ID: `invoice_processing_20261020_131708_80802c`, Status: `SUCCESS`, Duration: `42.84s`)
- `runs/invoice_processing/invoice_processing_20261023_043543_f5b68b.json` (Run ID: `invoice_processing_20261023_043543_f5b68b`, Status: `SUCCESS`, Duration: `36.35s`)
- `runs/invoice_processing/live_20261023_070056_invoic.json` (Run ID: `live_20261023_070056_invoic`, Status: `SUCCESS`, Duration: `39.74s`)
- `runs/invoice_processing/test_inv_run_01.json` (Run ID: `test_inv_run_01`, Status: `SUCCESS`, Duration: `0.17s`)