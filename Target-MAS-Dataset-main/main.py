"""
Main CLI Entry Point for Agentic Workflow Optimizer Workflows.

Supports all 10 production-grade multi-agent workflows (78 functional agents):
  1. Invoice Processing and Accounts Payable (8 agents)
  2. Customer Support Ticket Resolution (6 agents)
  3. Procurement and Supplier Selection (9 agents)
  4. Sales Lead Qualification (7 agents)
  5. Recruitment and Resume Screening (8 agents)
  6. Inventory Replenishment Planning (6 agents)
  7. Contract Review and Compliance (10 agents)
  8. IT Incident Triage and Remediation (7 agents)
  9. Employee Expense Reimbursement (8 agents)
  10. Market and Competitor Research (9 agents)

Commands:
  python main.py list
  python main.py run <workflow> [--id <sample_id>]
  python main.py run-incident
  python main.py run-expense
  python main.py run-market
  python main.py live-smoke [--workflow <name>]
  python main.py launch-phoenix [--port 6006]
  python main.py validate-keys
  python main.py ast-analyze [--output-dir <path>] [--json]
"""

import sys
import json
import argparse
from pathlib import Path

# Force UTF-8 on Windows stdout/stderr to avoid charmap encoding issues
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from common.settings import DATA_DIR, MANDATORY_MODEL
from common.api_key_manager import validate_all_workflow_keys
from common.tracing import setup_tracing, launch_phoenix_server

# Workflows 1 to 7 and 8 to 10 (Meta_MAS suite)
try:
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
    from Agents.incident_triage.workflow import run_incident_triage_workflow
    from Agents.expense_reimbursement.workflow import run_expense_workflow
    from Agents.market_research.workflow import run_market_research_workflow
    _WORKFLOW_IMPORTS_AVAILABLE = True
    _WORKFLOW_IMPORT_ERROR = None
except ImportError as err:
    _WORKFLOW_IMPORTS_AVAILABLE = False
    _WORKFLOW_IMPORT_ERROR = err
    run_invoice_processing_workflow = None
    run_customer_support_workflow = None
    run_procurement_workflow = None
    run_sales_lead_workflow = None
    run_recruitment_screening_workflow = None
    load_sample_recruitment_input = None
    run_inventory_replenishment_workflow = None
    load_sample_inventory_input = None
    run_contract_review_workflow = None
    load_sample_contract_input = None
    run_incident_triage_workflow = None
    run_expense_workflow = None
    run_market_research_workflow = None


def get_sample_incident(incident_id=None) -> dict:
    inc_path = DATA_DIR / "incident_triage" / "historical_incidents.json"
    if inc_path.exists():
        with open(inc_path, "r", encoding="utf-8") as f:
            incidents = json.load(f).get("incidents", [])
            if incident_id:
                matched = [i for i in incidents if i.get("incident_id") == incident_id]
                if matched:
                    return matched[0]
            if incidents:
                return incidents[0]
    return {
        "incident_id": "INC-SAMPLE-101",
        "title": "database-primary: PostgreSQL active connection pool reached 100% (250/250)",
        "service": "database-primary",
        "service_tier": "Tier-1",
        "category": "database_problem",
        "impact_users": 6500,
        "outage_minutes": 45,
        "symptoms": ["PostgreSQL active connection pool reached 100% (250/250)", "High p99 write latency"],
        "synthetic": True,
    }


def get_sample_expense(claim_id=None) -> dict:
    exp_path = DATA_DIR / "expense_reimbursement" / "expense_claims.json"
    if exp_path.exists():
        with open(exp_path, "r", encoding="utf-8") as f:
            claims = json.load(f).get("claims", [])
            if claim_id:
                matched = [c for c in claims if c.get("claim_id") == claim_id]
                if matched:
                    return matched[0]
            if claims:
                return claims[0]
    return {
        "claim_id": "CLM-SAMPLE-201",
        "employee_id": "EMP-401",
        "employee_name": "Elena Rostova",
        "department": "Engineering",
        "submission_date": "2026-09-12",
        "category": "Meals & Entertainment",
        "merchant": "The Grillhouse Steakhouse",
        "currency": "USD",
        "subtotal": 65.00,
        "tax": 5.77,
        "total": 70.77,
        "business_purpose": "Client technical architecture dinner",
        "receipt_text": "THE GRILLHOUSE - Date: 2026-09-12 - Total: $70.77",
        "synthetic": True,
    }


def get_sample_market_query() -> dict:
    return {
        "query": (
            "Compare CloudScale AI, DataPulse Systems, NexusOps Technologies, and Veloce Security "
            "across enterprise latency, pricing predictability, advertised vs actual SLAs, and on-prem readiness."
        ),
        "synthetic": True,
    }


WORKFLOW_CATALOG = {
    "invoice": {
        "id": "invoice_processing",
        "name": "Invoice Processing and Accounts Payable",
        "agents": 8,
        "key_env": "GROQ_API_KEY_1",
        "runner": run_invoice_processing_workflow,
    },
    "support": {
        "id": "customer_support",
        "name": "Customer Support Ticket Resolution",
        "agents": 6,
        "key_env": "GROQ_API_KEY_2",
        "runner": run_customer_support_workflow,
    },
    "procurement": {
        "id": "procurement_supplier_selection",
        "name": "Procurement and Supplier Selection",
        "agents": 9,
        "key_env": "GROQ_API_KEY_3",
        "runner": run_procurement_workflow,
    },
    "sales": {
        "id": "sales_lead_qualification",
        "name": "Sales Lead Qualification",
        "agents": 7,
        "key_env": "GROQ_API_KEY_4",
        "runner": run_sales_lead_workflow,
    },
    "recruitment": {
        "id": "recruitment_screening",
        "name": "Recruitment and Resume Screening",
        "agents": 8,
        "key_env": "GROQ_API_KEY_5",
        "runner": run_recruitment_screening_workflow,
    },
    "inventory": {
        "id": "inventory_replenishment",
        "name": "Inventory Replenishment Planning",
        "agents": 6,
        "key_env": "GROQ_API_KEY_6",
        "runner": run_inventory_replenishment_workflow,
    },
    "contract": {
        "id": "contract_review",
        "name": "Contract Review and Compliance",
        "agents": 10,
        "key_env": "GROQ_API_KEY_7",
        "runner": run_contract_review_workflow,
    },
    "incident": {
        "id": "incident_triage",
        "name": "IT Incident Triage and Remediation",
        "agents": 7,
        "key_env": "GROQ_API_KEY_8..14",
        "runner": run_incident_triage_workflow,
    },
    "expense": {
        "id": "expense_reimbursement",
        "name": "Employee Expense Reimbursement",
        "agents": 8,
        "key_env": "GROQ_API_KEY_8..14",
        "runner": run_expense_workflow,
    },
    "market": {
        "id": "market_research",
        "name": "Market and Competitor Research",
        "agents": 9,
        "key_env": "GROQ_API_KEY_8..14",
        "runner": run_market_research_workflow,
    },
}


def print_generic_result(result):
    """Cleanly print execution summary regardless of dict or WorkflowExecutionResult format."""
    print("\n" + "=" * 70)
    print("FINAL STRUCTURED RUN RESULT")
    print("=" * 70)

    if hasattr(result, "to_dict"):
        d = result.to_dict()
        print(f"Workflow: {d.get('workflow_name')}")
        print(f"Run ID: {d.get('run_id')}")
        print(f"Status: {d.get('status')}")
        print(f"Duration: {d.get('total_duration_seconds'):.2f}s")
        toks = d.get("token_usage", {})
        print(f"Total Tokens: {toks.get('total_tokens', 0)} (Prompt: {toks.get('prompt_tokens', 0)}, Completion: {toks.get('completion_tokens', 0)})")
        print(f"Estimated Cost: ${d.get('estimated_cost_usd', 0.0) or 0.0:.6f}")
        print(f"Run Summary Log: runs/{d.get('workflow_name')}/{d.get('run_id')}.json")
        print("\nFinal Output Preview:")
        out_str = json.dumps(d.get("output", {}), indent=2)
        preview_lines = out_str.splitlines()[:25]
        for line in preview_lines:
            print(f"  {line}")
        if len(out_str.splitlines()) > 25:
            print("  ... [truncated preview, full output stored in runs/]")
    else:
        print(f"Run ID: {result.get('run_id')}")
        print(f"Duration: {result.get('duration_seconds')}s")
        print(f"Total Tokens: {result.get('total_tokens', 0)} (Prompt: {result.get('prompt_tokens', 0)}, Completion: {result.get('completion_tokens', 0)})")
        print(f"Estimated Cost: ${result.get('estimated_cost_usd', 0.0):.6f}")
        print(f"Status: {result.get('status')}")
        print(f"Run Summary Log: runs/{result.get('workflow_name')}/{result.get('run_id')}.json")
        print("\nFinal Output:")
        print(str(result.get("final_output")))

    print("=" * 70 + "\n")


def cmd_list(args):
    print("\n" + "=" * 72)
    print(" Agentic Workflow Optimizer (AWO) - Complete Baseline Workflows")
    print(f" Mandatory Model: {MANDATORY_MODEL} (Groq)")
    print("=" * 72)
    total_agents = 0
    for alias, wf in WORKFLOW_CATALOG.items():
        total_agents += wf["agents"]
        print(f"[*] Alias: {alias.ljust(12)} | Workflow: {wf['name']}")
        print(f"    Agents: {wf['agents']} baseline agents | Assigned Key: {wf['key_env']}")
    print("-" * 72)
    print(f" Total Workflows: {len(WORKFLOW_CATALOG)} | Total Baseline Agents: {total_agents}")
    print("=" * 72 + "\n")


def cmd_run(args):
    target = args.workflow.lower()
    if target not in WORKFLOW_CATALOG:
        print(f"Error: Unknown workflow '{target}'. Choose from: {list(WORKFLOW_CATALOG.keys())}")
        sys.exit(1)

    if not _WORKFLOW_IMPORTS_AVAILABLE:
        print(f"\n[ERROR] Workflow execution runtime dependencies are missing: {_WORKFLOW_IMPORT_ERROR}")
        print("Please ensure your environment has all requirements installed: pip install -r requirements.txt\n")
        sys.exit(1)

    wf_info = WORKFLOW_CATALOG[target]
    print(f"\n[INFO] Initializing workflow: {wf_info['name']}")

    # 1. Invoice
    if target == "invoice":
        inv_file = DATA_DIR / "invoice_processing" / "invoices.json"
        if not inv_file.exists():
            print(f"Error: Dataset not found at {inv_file}. Run 'python generate_dummy_data.py' first.")
            sys.exit(1)
        with open(inv_file, "r", encoding="utf-8") as f:
            invoices = json.load(f)
        sample = invoices[0]
        if args.id:
            matched = [i for i in invoices if i.get("invoice_id") == args.id or i.get("invoice_number") == args.id]
            if matched:
                sample = matched[0]
        result = wf_info["runner"](sample)

    # 2. Support
    elif target == "support":
        tick_file = DATA_DIR / "customer_support" / "tickets.json"
        if not tick_file.exists():
            print(f"Error: Dataset not found at {tick_file}. Run 'python generate_dummy_data.py' first.")
            sys.exit(1)
        with open(tick_file, "r", encoding="utf-8") as f:
            tickets = json.load(f)
        sample = tickets[0]
        if args.id:
            matched = [t for t in tickets if t.get("ticket_id") == args.id]
            if matched:
                sample = matched[0]
        result = wf_info["runner"](sample)

    # 3. Procurement
    elif target == "procurement":
        pro_dir = DATA_DIR / "procurement_supplier_selection"
        req_file = pro_dir / "procurement_requests.json"
        quo_file = pro_dir / "quotations.json"
        sch_file = pro_dir / "delivery_schedules.json"
        with open(req_file, "r", encoding="utf-8") as f:
            requests = json.load(f)
        with open(quo_file, "r", encoding="utf-8") as f:
            quotations = json.load(f)
        with open(sch_file, "r", encoding="utf-8") as f:
            schedules = json.load(f)
        sample_req = requests[0]
        if args.id:
            matched = [r for r in requests if r.get("request_id") == args.id]
            if matched:
                sample_req = matched[0]
        req_id = sample_req.get("request_id")
        sample_quotes = [q for q in quotations if q.get("request_id") == req_id]
        sample_sch = [s for s in schedules if any(q["quotation_id"] == s["quotation_id"] for q in sample_quotes)]
        procurement_case = {
            "procurement_request": sample_req,
            "quotations": sample_quotes,
            "delivery_schedules": sample_sch
        }
        result = wf_info["runner"](procurement_case)

    # 4. Sales
    elif target == "sales":
        lead_file = DATA_DIR / "sales_lead_qualification" / "leads.json"
        if not lead_file.exists():
            print(f"Error: Dataset not found at {lead_file}. Run 'python generate_dummy_data.py' first.")
            sys.exit(1)
        with open(lead_file, "r", encoding="utf-8") as f:
            leads = json.load(f)
        sample = leads[0]
        if args.id:
            matched = [l for l in leads if l.get("lead_id") == args.id]
            if matched:
                sample = matched[0]
        result = wf_info["runner"](sample)

    # 5. Recruitment
    elif target == "recruitment":
        sample = load_sample_recruitment_input()
        if args.id:
            res_file = DATA_DIR / "recruitment_screening" / "resumes.json"
            if res_file.exists():
                with open(res_file, "r", encoding="utf-8") as f:
                    resumes = json.load(f)
                matched = [r for r in resumes if r.get("candidate_id") == args.id or r.get("application_id") == args.id]
                if matched:
                    sample["resume"] = matched[0]
        result = wf_info["runner"](sample)

    # 6. Inventory
    elif target == "inventory":
        sample = load_sample_inventory_input()
        if args.id:
            inv_file = DATA_DIR / "inventory_replenishment" / "inventory_records.json"
            if inv_file.exists():
                with open(inv_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
                matched = [r for r in records if r.get("sku_id") == args.id]
                if matched:
                    sample["inventory_record"] = matched[0]
        result = wf_info["runner"](sample)

    # 7. Contract
    elif target == "contract":
        sample = load_sample_contract_input()
        if args.id:
            c_file = DATA_DIR / "contract_review" / "contracts.json"
            if c_file.exists():
                with open(c_file, "r", encoding="utf-8") as f:
                    contracts = json.load(f)
                matched = [c for c in contracts if c.get("contract_id") == args.id]
                if matched:
                    sample["contract"] = matched[0]
        result = wf_info["runner"](sample)

    # 8. Incident Triage
    elif target == "incident":
        setup_tracing()
        sample = get_sample_incident(args.id)
        result = wf_info["runner"](sample)

    # 9. Expense Reimbursement
    elif target == "expense":
        setup_tracing()
        sample = get_sample_expense(args.id)
        result = wf_info["runner"](sample)

    # 10. Market Research
    elif target == "market":
        setup_tracing()
        sample = get_sample_market_query()
        result = wf_info["runner"](sample)

    print_generic_result(result)


def cmd_run_incident(args):
    setup_tracing()
    sample = get_sample_incident(args.id if hasattr(args, "id") else None)
    print(f"\n[AWO] Running IT Incident Triage workflow (Ticket: {sample.get('incident_id')})...")
    result = run_incident_triage_workflow(sample)
    print_generic_result(result)


def cmd_run_expense(args):
    setup_tracing()
    sample = get_sample_expense(args.id if hasattr(args, "id") else None)
    print(f"\n[AWO] Running Employee Expense Reimbursement workflow (Claim: {sample.get('claim_id')})...")
    result = run_expense_workflow(sample)
    print_generic_result(result)


def cmd_run_market(args):
    setup_tracing()
    sample = get_sample_market_query()
    print(f"\n[AWO] Running Market & Competitor Research workflow...")
    result = run_market_research_workflow(sample)
    print_generic_result(result)


def cmd_live_smoke(args):
    setup_tracing()
    print("\n" + "=" * 70)
    print(f" RUNNING LIVE SMOKE TESTS (Groq Model: {MANDATORY_MODEL})")
    print("=" * 70)

    target = args.workflow or "all"

    if target in ["all", "incident", "incident_triage"]:
        print("\n--> Testing IT Incident Triage (7 agents)...")
        res1 = run_incident_triage_workflow(get_sample_incident())
        print(f"    [Status: {res1.status}] Completed in {res1.total_duration_seconds:.2f}s across {res1.agent_count} agents.")

    if target in ["all", "expense", "expense_reimbursement"]:
        print("\n--> Testing Employee Expense Reimbursement (8 agents)...")
        res2 = run_expense_workflow(get_sample_expense())
        print(f"    [Status: {res2.status}] Completed in {res2.total_duration_seconds:.2f}s across {res2.agent_count} agents.")

    if target in ["all", "market", "market_research"]:
        print("\n--> Testing Market & Competitor Research (9 agents)...")
        res3 = run_market_research_workflow(get_sample_market_query())
        print(f"    [Status: {res3.status}] Completed in {res3.total_duration_seconds:.2f}s across {res3.agent_count} agents.")

    print("\n" + "=" * 70)
    print(" Live smoke tests finished.")
    print("=" * 70 + "\n")


def cmd_launch_phoenix(args):
    port = args.port or 6006
    print(f"[AWO] Launching local Arize Phoenix tracing server on port {port}...")
    launch_phoenix_server(port=port)
    print("Keep this terminal window open to preserve local tracing UI. Press Ctrl+C to exit.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[AWO] Stopped Phoenix tracing server.")


def cmd_ast_analyze(args):
    from tools.ast_workflow_analyzer import ASTWorkflowAnalyzer, print_cli_summary
    analyzer = ASTWorkflowAnalyzer()
    out_dir = analyzer.export_reports(output_dir=Path(args.output_dir) if args.output_dir else None)
    report_path = out_dir / "target_mas_ingestion_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_cli_summary(report)
        print(f"Reports saved to: {out_dir}")
        for p in sorted(out_dir.iterdir()):
            if p.is_file():
                print(f"  - {p.name} ({p.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Agentic Workflow Optimizer - Multi-Agent Workflows CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    parser_list = subparsers.add_parser("list", help="List all 10 baseline multi-agent workflows")
    parser_list.set_defaults(func=cmd_list)

    # AST analysis command
    p_ast = subparsers.add_parser("ast-analyze", help="Run static AST analysis across all 10 workflows")
    p_ast.add_argument("--output-dir", help="Optional custom directory to export reports")
    p_ast.add_argument("--json", action="store_true", help="Print raw JSON report to stdout")
    p_ast.set_defaults(func=cmd_ast_analyze)

    # Run command
    parser_run = subparsers.add_parser("run", help="Run a specific workflow")
    parser_run.add_argument("workflow", choices=list(WORKFLOW_CATALOG.keys()), help="Workflow alias to execute")
    parser_run.add_argument("--id", help="Optional sample record ID to process")
    parser_run.set_defaults(func=cmd_run)

    # Convenience shortcuts for workflows 8-10
    p_inc = subparsers.add_parser("run-incident", help="Run IT Incident Triage workflow")
    p_inc.add_argument("--id", help="Optional incident ID")
    p_inc.set_defaults(func=cmd_run_incident)

    p_exp = subparsers.add_parser("run-expense", help="Run Employee Expense Reimbursement workflow")
    p_exp.add_argument("--id", help="Optional claim ID")
    p_exp.set_defaults(func=cmd_run_expense)

    p_mkt = subparsers.add_parser("run-market", help="Run Market and Competitor Research workflow")
    p_mkt.set_defaults(func=cmd_run_market)

    # Smoke test command
    p_smoke = subparsers.add_parser("live-smoke", help="Run live smoke tests against Groq")
    p_smoke.add_argument("--workflow", choices=["incident", "expense", "market", "all"], default="all")
    p_smoke.set_defaults(func=cmd_live_smoke)

    # Phoenix server command
    p_px = subparsers.add_parser("launch-phoenix", help="Start Arize Phoenix local tracing UI")
    p_px.add_argument("--port", type=int, default=6006, help="Port to run Phoenix UI on (default: 6006)")
    p_px.set_defaults(func=cmd_launch_phoenix)

    # Key validation command
    parser_keys = subparsers.add_parser("validate-keys", help="Validate workflow Groq API key configuration")
    def _validate(args):
        print("\nValidating Workflow Groq API Keys...")
        keys = validate_all_workflow_keys()
        for wf, k in keys.items():
            print(f"  [OK] {wf.ljust(32)}: {k}")
        print("All workflow API keys successfully configured and isolated.\n")
    parser_keys.set_defaults(func=_validate)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        cmd_list(args)


if __name__ == "__main__":
    main()
