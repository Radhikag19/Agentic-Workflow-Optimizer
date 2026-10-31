# TraceProfiler Ingestion Report: `sales_lead_qualification`

**Workflow Name:** `sales_lead_qualification`  
**Description:** Baseline Sales Lead Qualification Multi-Agent Workflow  
**Assigned Key:** `GROQ_API_KEY_4`  
**Fixed Model:** `openai/gpt-oss-120b`  
**Total Agents:** 7 (Implemented: 7/7)  

## 1. Execution Order

| Step | Agent ID | Status | Implementation File |
|---|---|---|---|
| 1 | `lead_enrichment_agent` | Implemented | `Agents/sales_lead_qualification/lead_enrichment_agent.py` |
| 2 | `lead_fit_agent` | Implemented | `Agents/sales_lead_qualification/lead_fit_agent.py` |
| 3 | `intent_analysis_agent` | Implemented | `Agents/sales_lead_qualification/intent_analysis_agent.py` |
| 4 | `lead_scoring_agent` | Implemented | `Agents/sales_lead_qualification/lead_scoring_agent.py` |
| 5 | `opportunity_assessment_agent` | Implemented | `Agents/sales_lead_qualification/opportunity_assessment_agent.py` |
| 6 | `next_action_agent` | Implemented | `Agents/sales_lead_qualification/next_action_agent.py` |
| 7 | `final_review_agent_sales` | Implemented | `Agents/sales_lead_qualification/final_review_agent.py` |

## 2. Architectural Findings & Discrepancies

- **[WARNING] MULTIPLE_KICKOFF_CALLS**: Workflow invokes 7 sequential Crew kickoff calls instead of single execution graph.

## 3. Optimization Hypotheses & Opportunities

- Dynamic routing: Early-stop and disqualify immediately after lead_fit_agent if the company is in disqualified industries
- Pruning opportunity_assessment_agent when qualification score is under 45 points
- Parallelize lead_fit_agent and intent_analysis_agent immediately after enrichment
- Prompt compression: Pass only structured numeric feature vectors to lead_scoring_agent

## 4. Telemetry History Artifacts

Found **3** run artifacts recorded in `runs/sales_lead_qualification/`.

- `runs/sales_lead_qualification/live_20261023_070410_sales_.json` (Run ID: `live_20261023_070410_sales_`, Status: `SUCCESS`, Duration: `23.18s`)
- `runs/sales_lead_qualification/sales_lead_qualification_20261023_044003_dd3df9.json` (Run ID: `sales_lead_qualification_20261023_044003_dd3df9`, Status: `SUCCESS`, Duration: `21.19s`)
- `runs/sales_lead_qualification/test_sales_run_01.json` (Run ID: `test_sales_run_01`, Status: `SUCCESS`, Duration: `0.13s`)