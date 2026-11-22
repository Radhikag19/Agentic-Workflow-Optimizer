"""
Standalone Test Runner: JEV Model Selection for all Agents in a Workflow.

Usage:
    python run_jev_workflow.py <workflow_name>
    python run_jev_workflow.py contract_review

Outputs:
    Saves structured JSON recommendations for each functional agent under:
    JEV_testing/<workflow_name>/<agent_id>.json
"""

import os
import sys
import subprocess
import json
import time
import argparse
from pathlib import Path

# Auto-detect and switch to Python 3.12 if running under a Python without langgraph
try:
    import langgraph
except ImportError:
    py312_candidates = [
        Path(r"C:\Users\radhi\AppData\Local\Programs\Python\Python312\python.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python312" / "python.exe",
    ]
    py312 = next((p for p in py312_candidates if p.exists()), None)
    if py312 and Path(sys.executable).resolve() != py312.resolve():
        cmd = [str(py312)] + sys.argv
        sys.exit(subprocess.call(cmd))

# Add project roots to sys.path so it works from anywhere
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "Target-MAS-Dataset-main"))

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VALID_WORKFLOWS = [
    "contract_review",
    "customer_support",
    "expense_reimbursement",
    "incident_triage",
    "inventory_replenishment",
    "invoice_processing",
    "market_research",
    "procurement_supplier_selection",
    "recruitment_screening",
    "sales_lead_qualification",
]


def get_workflow_agents(workflow_id: str) -> list[str]:
    """Retrieve all functional agent IDs for the given workflow."""
    # Look for workflow.json under Agents/
    candidates = [
        CURRENT_DIR / "Target-MAS-Dataset-main" / "Agents" / workflow_id / "workflow.json",
        CURRENT_DIR / "Agents" / workflow_id / "workflow.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                data = json.loads(c.read_text(encoding="utf-8"))
                agents = [
                    a.get("agent_id") or a.get("id")
                    for a in data.get("agents", [])
                    if a.get("agent_id") or a.get("id")
                ]
                if agents:
                    return agents
            except Exception:
                pass

    # Fallback to MCP tools
    try:
        from mcp_server import tools
        prof = tools.get_agent_profiles(workflow_id)
        return [p["agent_id"] for p in prof.get("profiles", []) if "agent_id" in p]
    except Exception as e:
        raise RuntimeError(f"Could not load agents for workflow '{workflow_id}': {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run JEV Model Selection for all functional agents in a target workflow."
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        choices=VALID_WORKFLOWS,
        help=f"Target workflow name. Choices: {', '.join(VALID_WORKFLOWS)}",
    )
    parser.add_argument(
        "--output-dir",
        default="JEV_testing",
        help="Directory to save agent JSON results (default: JEV_testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between agent executions (default: 1.0s)",
    )

    args = parser.parse_args()

    if not args.workflow:
        print("\n[!] Please specify a workflow name.")
        print("Available workflows:")
        for wf in VALID_WORKFLOWS:
            print(f"  - {wf}")
        print("\nExample:")
        print("  python run_jev_workflow.py contract_review\n")
        sys.exit(1)

    wf_name = args.workflow
    agents = get_workflow_agents(wf_name)

    # Output directory: JEV_testing/<workflow_name>/
    out_dir = CURRENT_DIR / args.output_dir / wf_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(f" JEV WORKFLOW TEST RUNNER: {wf_name.upper()}")
    print("=" * 75)
    print(f" Total Functional Agents: {len(agents)}")
    print(f" Results Output Folder:   {out_dir}")
    print("=" * 75 + "\n")

    from optimization_agent.model_selection import run_model_selection

    summary_rows = []

    for idx, aid in enumerate(agents, 1):
        print(f"[{idx}/{len(agents)}] Running JEV Model Selection for agent: '{aid}'...")
        t0 = time.perf_counter()

        try:
            res = run_model_selection(
                workflow_id=wf_name,
                agent_id=aid,
                export_report=False,  # We manage export directly into JEV_testing
            )
            elapsed = time.perf_counter() - t0

            # Save individual JSON file
            agent_json_path = out_dir / f"{aid}.json"
            agent_json_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

            reqs = res.get("agent_requirements", {})
            recom = res.get("model_recommendation", {})
            val_info = res.get("selection_validation", {})

            t_comp = reqs.get("task_complexity", {}).get("class", "N/A")
            p_comp = reqs.get("prompt_complexity", {}).get("class", "N/A")
            tok_c = reqs.get("token_consumption", {}).get("class", "N/A")
            sel_model = recom.get("selected_model") or "None"
            is_valid = recom.get("selection_validated", False)
            is_downgrade = val_info.get("is_downgrade_candidate", False)

            print(f"    -> Status: SUCCESS ({elapsed:.2f}s)")
            print(f"    -> Complexity: Task={t_comp} | Prompt={p_comp} | Tokens={tok_c}")
            print(f"    -> Recommendation: {sel_model} (Validated: {is_valid}, Downgrade: {is_downgrade})")
            print(f"    -> Saved JSON: {agent_json_path}\n")

            summary_rows.append({
                "agent_id": aid,
                "task_complexity": t_comp,
                "prompt_complexity": p_comp,
                "token_consumption": tok_c,
                "selected_model": sel_model,
                "validated": is_valid,
                "is_downgrade": is_downgrade,
                "file": str(agent_json_path),
            })

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"    [ERROR] Failed for agent '{aid}': {e}\n")
            summary_rows.append({
                "agent_id": aid,
                "task_complexity": "ERROR",
                "prompt_complexity": "ERROR",
                "token_consumption": "ERROR",
                "selected_model": "ERROR",
                "validated": False,
                "is_downgrade": False,
                "error": str(e),
            })

        if idx < len(agents) and args.delay > 0:
            time.sleep(args.delay)

    # Print final summary table
    print("\n" + "=" * 90)
    print(f" EXECUTION SUMMARY: {wf_name.upper()} ({len(summary_rows)} agents)")
    print("=" * 90)
    print(f"{'Agent ID':<35} {'Task':<8} {'Prompt':<8} {'Tokens':<8} {'Selected Model':<22} {'Downgrade?'}")
    print("-" * 90)
    for row in summary_rows:
        aid = row["agent_id"][:33]
        tc = row["task_complexity"]
        pc = row["prompt_complexity"]
        tok = row["token_consumption"]
        sm = (row.get("selected_model") or "None")[:21]
        dg = "YES" if row.get("is_downgrade") else "NO"
        print(f"{aid:<35} {tc:<8} {pc:<8} {tok:<8} {sm:<22} {dg}")
    print("=" * 90)
    print(f"\nAll {len(summary_rows)} agent JSON files generated in:\n  {out_dir}\n")


if __name__ == "__main__":
    main()

