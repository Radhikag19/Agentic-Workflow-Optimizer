"""
Comprehensive Live Test Runner for all 10 Workflows (78 Functional Agents).

Executes each workflow with live Groq API calls using openai/gpt-oss-120b,
enforcing per-workflow and per-agent API key isolation across GROQ_API_KEY_1..14.
Outputs are saved workflow-wise in runs/<workflow_name>/.
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on Windows stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_DIR = PROJECT_ROOT / "data"

from common.settings import MANDATORY_MODEL
from common.api_key_manager import validate_all_workflow_keys

# Workflows 1-7
from Agents.invoice_processing.workflow import run_invoice_processing_workflow
from Agents.customer_support.workflow import run_customer_support_workflow
from Agents.procurement_supplier_selection.workflow import run_procurement_workflow
from Agents.sales_lead_qualification.workflow import run_sales_lead_workflow
from Agents.recruitment_screening.workflow import (
    run_recruitment_screening_workflow,
    load_sample_recruitment_input,
)
from Agents.inventory_replenishment.workflow import (
    run_inventory_replenishment_workflow,
    load_sample_inventory_input,
)
from Agents.contract_review.workflow import (
    run_contract_review_workflow,
    load_sample_contract_input,
)

# Workflows 8-10
from Agents.incident_triage.workflow import run_incident_triage_workflow
from Agents.expense_reimbursement.workflow import run_expense_workflow
from Agents.market_research.workflow import run_market_research_workflow
from main import get_sample_incident, get_sample_expense, get_sample_market_query


def get_sample_inputs():
    """Load clean samples for all 10 workflows."""
    # 1. Invoice
    with open(DATA_DIR / "invoice_processing" / "invoices.json", "r", encoding="utf-8") as f:
        inv_sample = json.load(f)[0]

    # 2. Support
    with open(DATA_DIR / "customer_support" / "tickets.json", "r", encoding="utf-8") as f:
        cs_sample = json.load(f)[0]

    # 3. Procurement
    pro_dir = DATA_DIR / "procurement_supplier_selection"
    with open(pro_dir / "procurement_requests.json", "r", encoding="utf-8") as f:
        req = json.load(f)[0]
    with open(pro_dir / "quotations.json", "r", encoding="utf-8") as f:
        quotes = [q for q in json.load(f) if q.get("request_id") == req.get("request_id")]
    with open(pro_dir / "delivery_schedules.json", "r", encoding="utf-8") as f:
        schs = [s for s in json.load(f) if any(q["quotation_id"] == s["quotation_id"] for q in quotes)]
    pro_sample = {"procurement_request": req, "quotations": quotes, "delivery_schedules": schs}

    # 4. Sales
    with open(DATA_DIR / "sales_lead_qualification" / "leads.json", "r", encoding="utf-8") as f:
        sales_sample = json.load(f)[0]

    # 5. Recruitment
    rec_sample = load_sample_recruitment_input()

    # 6. Inventory
    inv_plan_sample = load_sample_inventory_input()

    # 7. Contract
    contract_sample = load_sample_contract_input()

    # 8. Incident
    inc_sample = get_sample_incident()

    # 9. Expense
    exp_sample = get_sample_expense()

    # 10. Market
    mkt_sample = get_sample_market_query()

    return {
        "invoice_processing": inv_sample,
        "customer_support": cs_sample,
        "procurement_supplier_selection": pro_sample,
        "sales_lead_qualification": sales_sample,
        "recruitment_screening": rec_sample,
        "inventory_replenishment": inv_plan_sample,
        "contract_review": contract_sample,
        "incident_triage": inc_sample,
        "expense_reimbursement": exp_sample,
        "market_research": mkt_sample,
    }


WORKFLOW_SPECS = [
    {
        "index": 1,
        "id": "invoice_processing",
        "name": "Invoice Processing and Accounts Payable",
        "agents": 8,
        "key": "GROQ_API_KEY_1",
        "runner": run_invoice_processing_workflow,
    },
    {
        "index": 2,
        "id": "customer_support",
        "name": "Customer Support Ticket Resolution",
        "agents": 6,
        "key": "GROQ_API_KEY_2",
        "runner": run_customer_support_workflow,
    },
    {
        "index": 3,
        "id": "procurement_supplier_selection",
        "name": "Procurement and Supplier Selection",
        "agents": 9,
        "key": "GROQ_API_KEY_3",
        "runner": run_procurement_workflow,
    },
    {
        "index": 4,
        "id": "sales_lead_qualification",
        "name": "Sales Lead Qualification",
        "agents": 7,
        "key": "GROQ_API_KEY_4",
        "runner": run_sales_lead_workflow,
    },
    {
        "index": 5,
        "id": "recruitment_screening",
        "name": "Recruitment and Resume Screening",
        "agents": 8,
        "key": "GROQ_API_KEY_5",
        "runner": run_recruitment_screening_workflow,
    },
    {
        "index": 6,
        "id": "inventory_replenishment",
        "name": "Inventory Replenishment Planning",
        "agents": 6,
        "key": "GROQ_API_KEY_6",
        "runner": run_inventory_replenishment_workflow,
    },
    {
        "index": 7,
        "id": "contract_review",
        "name": "Contract Review and Compliance",
        "agents": 10,
        "key": "GROQ_API_KEY_7",
        "runner": run_contract_review_workflow,
    },
    {
        "index": 8,
        "id": "incident_triage",
        "name": "IT Incident Triage and Remediation",
        "agents": 7,
        "key": "GROQ_API_KEY_8..14",
        "runner": run_incident_triage_workflow,
    },
    {
        "index": 9,
        "id": "expense_reimbursement",
        "name": "Employee Expense Reimbursement",
        "agents": 8,
        "key": "GROQ_API_KEY_8..14",
        "runner": run_expense_workflow,
    },
    {
        "index": 10,
        "id": "market_research",
        "name": "Market and Competitor Research",
        "agents": 9,
        "key": "GROQ_API_KEY_8..14",
        "runner": run_market_research_workflow,
    },
]


def extract_metrics(result):
    """Normalize metrics from either dict or WorkflowExecutionResult."""
    if hasattr(result, "to_dict"):
        d = result.to_dict()
        toks = d.get("token_usage", {})
        return {
            "run_id": d.get("run_id"),
            "status": d.get("status"),
            "duration": round(float(d.get("total_duration_seconds", 0.0)), 2),
            "prompt_tokens": int(toks.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(toks.get("completion_tokens", 0) or 0),
            "total_tokens": int(toks.get("total_tokens", 0) or 0),
            "cost_usd": d.get("estimated_cost_usd", 0.0) or 0.0,
            "error": d.get("error"),
        }
    else:
        return {
            "run_id": result.get("run_id"),
            "status": result.get("status"),
            "duration": round(float(result.get("duration_seconds", 0.0)), 2),
            "prompt_tokens": int(result.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(result.get("completion_tokens", 0) or 0),
            "total_tokens": int(result.get("total_tokens", 0) or 0),
            "cost_usd": result.get("estimated_cost_usd", 0.0) or 0.0,
            "error": result.get("error_message"),
        }


def main():
    print("\n" + "=" * 78)
    print(" AGENTIC WORKFLOW OPTIMIZER (AWO) - LIVE 10-WORKFLOW TEST SUITE")
    print(f" Mandatory Model: {MANDATORY_MODEL} (Groq)")
    print(f" Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 78)

    print("\n[Step 1/3] Validating Groq API Keys...")
    keys = validate_all_workflow_keys()
    for wf, k in keys.items():
        print(f"  [OK] {wf.ljust(32)}: {k}")
    print("All required API keys validated.\n")

    print("[Step 2/3] Loading input datasets for all 10 workflows...")
    samples = get_sample_inputs()
    print("Sample datasets loaded.\n")

    print("[Step 3/3] Executing all 10 workflows sequentially...\n")

    results_summary = []
    total_tokens_all = 0
    total_prompt_all = 0
    total_completion_all = 0
    total_duration_all = 0.0

    for spec in WORKFLOW_SPECS:
        idx = spec["index"]
        wf_id = spec["id"]
        wf_name = spec["name"]
        agents_count = spec["agents"]
        assigned_key = spec["key"]
        runner = spec["runner"]
        sample_input = samples[wf_id]

        divider = "-" * 78
        print(f"\n{divider}")
        print(f" [{idx}/10] RUNNING WORKFLOW: {wf_name.upper()}")
        print(f" Agents: {agents_count} | Assigned Key: {assigned_key}")
        print(divider)

        t_start = time.perf_counter()
        run_tag = f"live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{wf_id[:6]}"

        try:
            res = runner(sample_input, run_id=run_tag)
            m = extract_metrics(res)
            dur = m["duration"]
            status = m["status"]
            tot_toks = m["total_tokens"]
            p_toks = m["prompt_tokens"]
            c_toks = m["completion_tokens"]

            total_tokens_all += tot_toks
            total_prompt_all += p_toks
            total_completion_all += c_toks
            total_duration_all += dur

            # Locate saved artifact
            wf_folder = RUNS_DIR / wf_id
            wf_folder.mkdir(parents=True, exist_ok=True)
            saved_file = wf_folder / f"{m['run_id']}.json"

            print(f"\n >>> [{status}] Completed in {dur:.2f}s | Tokens Generated: {tot_toks:,} (Prompt: {p_toks:,}, Completion: {c_toks:,})")
            print(f" >>> Run Output Artifact: {saved_file}")

            results_summary.append({
                "index": idx,
                "workflow_id": wf_id,
                "workflow_name": wf_name,
                "agents": agents_count,
                "assigned_key": assigned_key,
                "status": status,
                "run_id": m["run_id"],
                "duration_seconds": dur,
                "prompt_tokens": p_toks,
                "completion_tokens": c_toks,
                "total_tokens": tot_toks,
                "artifact_path": str(saved_file),
                "error": None,
            })

        except Exception as exc:
            dur = round(time.perf_counter() - t_start, 2)
            total_duration_all += dur
            print(f"\n >>> [FAILED] {wf_name} failed after {dur:.2f}s: {exc}")
            results_summary.append({
                "index": idx,
                "workflow_id": wf_id,
                "workflow_name": wf_name,
                "agents": agents_count,
                "assigned_key": assigned_key,
                "status": "FAILED",
                "run_id": run_tag,
                "duration_seconds": dur,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "artifact_path": None,
                "error": str(exc),
            })

        # Inter-workflow pacing pause to reset Groq RPM/TPM windows
        if idx < len(WORKFLOW_SPECS):
            print("\n [Pacing] Pausing 4 seconds for Groq rate window reset...")
            time.sleep(4.0)

    # Save grand summary artifact
    summary_file = RUNS_DIR / "combined_10_workflows_test_summary.json"
    grand_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MANDATORY_MODEL,
        "total_workflows": len(WORKFLOW_SPECS),
        "total_agents": sum(s["agents"] for s in WORKFLOW_SPECS),
        "successful_workflows": sum(1 for r in results_summary if r["status"] == "SUCCESS"),
        "failed_workflows": sum(1 for r in results_summary if r["status"] == "FAILED"),
        "total_duration_seconds": round(total_duration_all, 2),
        "total_tokens": total_tokens_all,
        "prompt_tokens": total_prompt_all,
        "completion_tokens": total_completion_all,
        "runs": results_summary,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(grand_summary, f, indent=2)

    # Console Summary Table
    print("\n\n" + "=" * 90)
    print(" LIVE 10-WORKFLOW EXECUTION SUMMARY MATRIX")
    print("=" * 90)
    header = f"{'#':<3} | {'Workflow Name':<34} | {'Agents':<6} | {'Status':<9} | {'Duration':<8} | {'Tokens':<10}"
    print(header)
    print("-" * 90)
    for r in results_summary:
        st_color = "[SUCCESS]" if r["status"] == "SUCCESS" else "[FAILED]"
        print(f"{r['index']:<3} | {r['workflow_name'][:34]:<34} | {r['agents']:<6} | {st_color:<9} | {r['duration_seconds']:>6.2f}s | {r['total_tokens']:>10,}")
    print("-" * 90)
    print(f" TOTAL TOKENS GENERATED ACROSS ALL 10 WORKFLOWS: {total_tokens_all:,} tokens")
    print(f" TOTAL EXECUTION TIME: {total_duration_all:.2f}s")
    print(f" GRAND SUMMARY PERSISTED AT: {summary_file}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
