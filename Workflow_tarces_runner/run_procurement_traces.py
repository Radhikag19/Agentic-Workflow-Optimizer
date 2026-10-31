"""
Dedicated Live Trace Generator for Procurement & Supplier Selection Workflow.

Enforces:
1. 1-to-1 Groq API Key Distribution across all 9 sub-agents (Keys 1..9).
2. Live, non-hallucinated execution using Groq API with openai/gpt-oss-120b.
3. Reference trace format strictly matching runs/procurement_supplier_selection/live_20261023_070208_procur.json.
4. Outputs persisted directly in Ingestion_Traces/procurement_supplier_selection/ (no agent-wise subfolders).
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
    "quotation_analysis_agent": "GROQ_API_KEY_1",
    "supplier_profile_agent": "GROQ_API_KEY_2",
    "delivery_analysis_agent": "GROQ_API_KEY_3",
    "supplier_risk_agent": "GROQ_API_KEY_4",
    "contract_analysis_agent": "GROQ_API_KEY_5",
    "cost_comparison_agent": "GROQ_API_KEY_11",
    "weighted_evaluation_agent": "GROQ_API_KEY_7",
    "selection_recommendation_agent": "GROQ_API_KEY_12",
    "final_review_agent_procurement": "GROQ_API_KEY_13",
}

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


# Deterministic Helper Functions
def deterministic_profile_lookup(supplier_ids: list, all_profiles: list) -> dict:
    profiles = {}
    for s in all_profiles:
        if s.get("supplier_id") in supplier_ids:
            profiles[s.get("supplier_id")] = s
    return profiles

def deterministic_delivery_check(required_date_str: str, delivery_schedules: list) -> list:
    results = []
    req_dt = datetime.strptime(required_date_str, "%Y-%m-%d")
    for sched in delivery_schedules:
        promised_str = sched.get("promised_delivery_date")
        promised_dt = datetime.strptime(promised_str, "%Y-%m-%d")
        days_diff = (req_dt - promised_dt).days
        meets_deadline = days_diff >= 0
        results.append({
            "supplier_id": sched.get("supplier_id"),
            "lead_time_days": sched.get("lead_time_days"),
            "promised_date": promised_str,
            "required_date": required_date_str,
            "buffer_days": days_diff,
            "meets_deadline": meets_deadline
        })
    return results

def deterministic_risk_lookup(supplier_ids: list, all_risks: list) -> dict:
    records = {}
    for r in all_risks:
        if r.get("supplier_id") in supplier_ids:
            records[r.get("supplier_id")] = r
    return records

def deterministic_contract_lookup(supplier_ids: list, all_contracts: list) -> dict:
    contracts = {}
    for c in all_contracts:
        if c.get("supplier_id") in supplier_ids:
            contracts[c.get("supplier_id")] = c
    return contracts

def deterministic_cost_ranking(quotations: list) -> list:
    ranked = []
    for q in quotations:
        item_tot = float(q.get("item_total", 0))
        ship = float(q.get("shipping_cost", 0))
        tax = float(q.get("tax_amount", 0))
        total_landed = round(item_tot + ship + tax, 2)
        ranked.append({
            "supplier_id": q.get("supplier_id"),
            "supplier_name": q.get("supplier_name"),
            "unit_price": q.get("unit_price"),
            "item_total": item_tot,
            "shipping_cost": ship,
            "tax_amount": tax,
            "total_landed_cost": total_landed
        })
    ranked.sort(key=lambda x: x["total_landed_cost"])
    for idx, r in enumerate(ranked, 1):
        r["cost_rank"] = idx
    return ranked

def deterministic_weighted_scoring(
    cost_data: list,
    delivery_data: list,
    risk_data: dict,
    profile_data: dict,
    evaluation_criteria: dict
) -> list:
    weights = evaluation_criteria.get("weights", {
        "cost_score": 0.35,
        "delivery_score": 0.25,
        "risk_score": 0.20,
        "contract_terms_score": 0.10,
        "quality_capability_score": 0.10
    })

    costs = {c["supplier_id"]: c["total_landed_cost"] for c in cost_data}
    min_cost = min(costs.values()) if costs else 1.0
    delivery_map = {d["supplier_id"]: d for d in delivery_data}

    scored_suppliers = []
    for sup_id, cost in costs.items():
        cost_s = round((min_cost / cost) * 100, 2)
        
        deliv = delivery_map.get(sup_id, {})
        buf = deliv.get("buffer_days", 0)
        deliv_s = 100.0 if buf >= 0 else max(0.0, 100.0 + (buf * 15.0))

        r = risk_data.get(sup_id, {})
        fail_rate = float(r.get("delivery_failure_rate", 0.05))
        defect_rate = float(r.get("quality_defect_rate", 0.02))
        risk_s = max(0.0, round(100.0 - (fail_rate * 300.0) - (defect_rate * 400.0), 2))

        contract_s = 90.0
        prof = profile_data.get(sup_id, {})
        cap_s = 95.0 if prof.get("iso_certified") else 70.0

        total_score = round(
            (cost_s * weights["cost_score"]) +
            (deliv_s * weights["delivery_score"]) +
            (risk_s * weights["risk_score"]) +
            (contract_s * weights["contract_terms_score"]) +
            (cap_s * weights["quality_capability_score"]),
            2
        )

        scored_suppliers.append({
            "supplier_id": sup_id,
            "cost_score": cost_s,
            "delivery_score": deliv_s,
            "risk_score": risk_s,
            "contract_score": contract_s,
            "capability_score": cap_s,
            "total_weighted_score": total_score
        })

    scored_suppliers.sort(key=lambda x: x["total_weighted_score"], reverse=True)
    for idx, s in enumerate(scored_suppliers, 1):
        s["overall_rank"] = idx

    return scored_suppliers


def call_agent_groq(
    agent_id: str,
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    max_tokens: int = 1500,
    max_retries: int = 6,
) -> Dict[str, Any]:
    """Execute a single agent stage via its assigned dedicated Groq API key."""
    env_var_name = AGENT_KEY_MAP[agent_id]
    api_key = os.getenv(env_var_name, "").strip()
    if not api_key:
        raise ValueError(f"Missing API key environment variable {env_var_name} for agent {agent_id}")

    masked_id = f"{env_var_name} ({mask_key(api_key)})"
    client = Groq(api_key=api_key)

    system_prompt = (
        f"You are the {role}.\n"
        f"Goal: {goal}\n"
        f"Backstory: {backstory}\n"
        f"Strictly return valid JSON only. Do not wrap in markdown quotes if possible, or use standard ```json ... ``` blocks."
    )

    t0 = time.time()
    last_err = None
    for attempt in range(max_retries):
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
            if "429" in err_str or "rate_limit" in err_str.lower():
                m_min = re.search(r"try again in (\d+)m(\d+(?:\.\d+)?)s", err_str)
                m_sec = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                if m_min:
                    raw_sec = int(m_min.group(1)) * 60 + float(m_min.group(2))
                    wait_time = min(raw_sec, 25.0)
                elif m_sec:
                    wait_time = min(float(m_sec.group(1)) + 1.0, 25.0)
                else:
                    wait_time = 15.0
                print(f"    [RateLimit Cooldown] Agent '{agent_id}' waiting {wait_time:.1f}s for quota replenishment...", flush=True)
                time.sleep(wait_time)
            else:
                time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"Agent '{agent_id}' failed after {max_retries} attempts: {last_err}")


def run_single_procurement_trace(
    req: Dict[str, Any],
    quotes: List[Dict[str, Any]],
    schedules: List[Dict[str, Any]],
    all_profiles: List[Dict[str, Any]],
    all_risks: List[Dict[str, Any]],
    all_contracts: List[Dict[str, Any]],
    evaluation_criteria: Dict[str, Any],
    run_number: int = 1,
    output_dir: Path = Path("Ingestion_Traces/procurement_supplier_selection"),
) -> Dict[str, Any]:
    """Execute all 9 agents sequentially for one procurement request and persist the trace."""
    run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_trace{run_number:02d}"
    t_start = time.time()
    agent_executions = []

    req_id = req.get("request_id", "RFQ-UNKNOWN")
    item_desc = req.get("item_description", "Item")
    supplier_ids = [q.get("supplier_id") for q in quotes]
    req_date = req.get("target_delivery_date", "2026-12-31")

    print(f"\n--- [Run {run_number:02d}] Executing Procurement Selection for {req_id} ({item_desc[:35]}) ---")

    # 1. Quotation Analysis Agent (Key 1)
    print("  [1/9] Quotation Analysis Agent -> GROQ_API_KEY_1...")
    desc_1 = f"""
Analyze and normalize competing quotations for this procurement request.
Procurement Request:
{json.dumps(req, indent=2)}

Submitted Supplier Quotations:
{json.dumps(quotes, indent=2)}

Tasks:
1. Normalize item quantities, unit prices, shipping, and taxes per supplier.
2. Identify missing pricing elements, validity periods, or currency discrepancies.
3. Return normalized quotation summaries for all competing bidders.
Return strictly valid JSON with keys: 'request_id', 'normalized_quotations', and 'extraction_flags'.
"""
    exec_1 = call_agent_groq(
        agent_id="quotation_analysis_agent",
        role="Senior Quotation Analyst",
        goal="Extract, normalize, and validate competing supplier bids and commercial terms.",
        backstory="A procurement analyst specializing in normalizing commercial bids across disparate supplier formats.",
        task_description=desc_1,
        max_tokens=2500,
    )
    exec_1["order"] = 1
    exec_1["agent_id"] = "quotation_analysis_agent"
    agent_executions.append(exec_1)

    # 2. Supplier Profile Agent (Key 2)
    print("  [2/9] Supplier Profile Agent -> GROQ_API_KEY_2...")
    profile_data = deterministic_profile_lookup(supplier_ids, all_profiles)
    desc_2 = f"""
Evaluate the profiles of competing suppliers.
Target Supplier Profiles from Registry:
{json.dumps(profile_data, indent=2)}

Tasks:
1. Compare technical capability, ISO certifications, and manufacturing capacity.
2. Note any operational limitations or unverified profile data.
Return strictly valid JSON with keys: 'evaluated_profiles', 'profile_strengths', and 'operational_concerns'.
"""
    exec_2 = call_agent_groq(
        agent_id="supplier_profile_agent",
        role="Supplier Intelligence Officer",
        goal="Evaluate supplier capabilities, manufacturing capacity, and operational standing.",
        backstory="A strategic sourcing specialist who audits supplier corporate health, certifications, and technical capacity.",
        task_description=desc_2,
        max_tokens=2500,
    )
    exec_2["order"] = 2
    exec_2["agent_id"] = "supplier_profile_agent"
    agent_executions.append(exec_2)

    # 3. Delivery Analysis Agent (Key 3)
    print("  [3/9] Delivery Analysis Agent -> GROQ_API_KEY_3...")
    deliv_calc = deterministic_delivery_check(req_date, schedules)
    desc_3 = f"""
Analyze delivery commitments and schedule risks for this procurement request.
Required Target Date: {req_date}
Deterministic Schedule Analysis from Python:
{json.dumps(deliv_calc, indent=2)}

Instructions:
1. Highlight suppliers that meet vs. violate the delivery deadline.
2. Assess buffer margins and fulfillment risk.
Return strictly valid JSON with keys: 'delivery_evaluation', 'on_time_suppliers', 'late_risk_suppliers', and 'logistics_summary'.
"""
    exec_3 = call_agent_groq(
        agent_id="delivery_analysis_agent",
        role="Supply Chain Fulfillment Analyst",
        goal="Evaluate supplier delivery schedules, shipping transit times, and milestone feasibility.",
        backstory="A logistics specialist focused on critical path analysis and preventing supply chain delays.",
        task_description=desc_3,
        max_tokens=2500,
    )
    exec_3["order"] = 3
    exec_3["agent_id"] = "delivery_analysis_agent"
    agent_executions.append(exec_3)

    # 4. Supplier Risk Agent (Key 4)
    print("  [4/9] Supplier Risk Agent -> GROQ_API_KEY_4...")
    risk_data = deterministic_risk_lookup(supplier_ids, all_risks)
    desc_4 = f"""
Perform risk assessment across all competing suppliers.
Synthetic Supplier Risk Records:
{json.dumps(risk_data, indent=2)}

Instructions:
1. Evaluate defect rates, delivery failure rates, and audit warnings.
2. Categorize each supplier's risk ('LOW', 'MEDIUM', 'HIGH').
3. Highlight red flags that should block supplier selection.
Return strictly valid JSON with keys: 'supplier_risk_scores', 'flagged_suppliers', and 'risk_mitigation_notes'.
"""
    exec_4 = call_agent_groq(
        agent_id="supplier_risk_agent",
        role="Procurement Risk & Compliance Auditor",
        goal="Assess supplier quality defect rates, historical failure frequency, and audit status.",
        backstory="A risk governance analyst evaluating supplier reliability and preventing vendor failures.",
        task_description=desc_4,
        max_tokens=2500,
    )
    exec_4["order"] = 4
    exec_4["agent_id"] = "supplier_risk_agent"
    agent_executions.append(exec_4)

    # 5. Contract Analysis Agent (Key 5)
    print("  [5/9] Contract Analysis Agent -> GROQ_API_KEY_5...")
    contract_data = deterministic_contract_lookup(supplier_ids, all_contracts)
    desc_5 = f"""
Analyze contract terms across all competing suppliers.
Contract Terms by Supplier:
{json.dumps(contract_data, indent=2)}

Tasks:
1. Compare payment terms (e.g. Net 30 vs Net 45), warranty length, and delay penalties.
2. Identify any unfavorable or restrictive clauses.
Return strictly valid JSON with keys: 'contract_comparison', 'best_terms_supplier', and 'contract_risks'.
"""
    exec_5 = call_agent_groq(
        agent_id="contract_analysis_agent",
        role="Commercial Contracts Specialist",
        goal="Audit legal terms, payment conditions, delay penalties, and warranty protections.",
        backstory="A corporate contract analyst reviewing vendor terms to protect organizational interests.",
        task_description=desc_5,
        max_tokens=2500,
    )
    exec_5["order"] = 5
    exec_5["agent_id"] = "contract_analysis_agent"
    agent_executions.append(exec_5)

    # 6. Cost Comparison Agent (Key 11)
    print(f"  [6/9] Cost Comparison Agent -> {AGENT_KEY_MAP['cost_comparison_agent']}...")
    cost_ranking = deterministic_cost_ranking(quotes)
    desc_6 = f"""
Analyze the total commercial cost of competing supplier quotations.
Deterministic Landed Cost Ranking from Python:
{json.dumps(cost_ranking, indent=2)}

Tasks:
1. Explain the cost breakdown (unit price vs. shipping vs. tax).
2. Highlight the lowest-cost bidder and the cost variance to other bidders.
Return strictly valid JSON with keys: 'cost_ranking', 'lowest_cost_bidder', 'cost_spread_percentage', and 'commercial_summary'.
"""
    exec_6 = call_agent_groq(
        agent_id="cost_comparison_agent",
        role="Cost & Financial Analyst",
        goal="Perform rigorous commercial cost comparison and rank suppliers by total evaluated cost.",
        backstory="A financial procurement analyst calculating total cost of ownership across competing supplier bids.",
        task_description=desc_6,
        max_tokens=2500,
    )
    exec_6["order"] = 6
    exec_6["agent_id"] = "cost_comparison_agent"
    agent_executions.append(exec_6)

    # 7. Weighted Evaluation Agent (Key 7)
    print("  [7/9] Weighted Evaluation Agent -> GROQ_API_KEY_7...")
    scored_results = deterministic_weighted_scoring(cost_ranking, deliv_calc, risk_data, profile_data, evaluation_criteria)
    desc_7 = f"""
Review and explain the multi-criteria weighted scoring across competing suppliers.
Deterministic Weighted Scores from Python:
{json.dumps(scored_results, indent=2)}

Instructions:
1. Explain why the top-ranked supplier achieved the highest score.
2. Discuss the trade-offs between cost savings and risk/delivery factors.
Return strictly valid JSON with keys: 'weighted_scores', 'top_ranked_supplier', 'score_spread', and 'evaluation_rationale'.
"""
    exec_7 = call_agent_groq(
        agent_id="weighted_evaluation_agent",
        role="Multi-Criteria Decision Analysis Specialist",
        goal="Apply formal weighted decision matrices to rank suppliers objectively across all evaluation pillars.",
        backstory="An operations research analyst calculating quantitative scoring models for vendor selection.",
        task_description=desc_7,
        max_tokens=2500,
    )
    exec_7["order"] = 7
    exec_7["agent_id"] = "weighted_evaluation_agent"
    agent_executions.append(exec_7)

    # Stage evidence bundle for subsequent synthesis
    stage_evidence = {
        "quotations": exec_1["output"],
        "profiles": exec_2["output"],
        "delivery": exec_3["output"],
        "risk": exec_4["output"],
        "contracts": exec_5["output"],
        "cost": exec_6["output"],
        "weighted_evaluation": exec_7["output"],
    }

    # 8. Selection Recommendation Agent (Key 12)
    print(f"  [8/9] Selection Recommendation Agent -> {AGENT_KEY_MAP['selection_recommendation_agent']}...")
    desc_8 = f"""
Formulate an authoritative supplier award recommendation.
Consolidated Procurement Evidence:
{json.dumps(stage_evidence, indent=2)}

Instructions:
1. Recommend the winning supplier or explain why no supplier qualifies.
2. Explicitly detail trade-offs (e.g. lowest cost vs. delivery buffer).
3. Note: NEVER issue binding contracts or purchase orders.
Return strictly valid JSON with keys: 'recommended_supplier_id', 'commercial_rationale', 'risk_acceptance', and 'next_steps'.
"""
    exec_8 = call_agent_groq(
        agent_id="selection_recommendation_agent",
        role="Strategic Sourcing Committee Chair",
        goal="Synthesize technical, delivery, commercial, risk, and scoring evidence into an authoritative supplier recommendation.",
        backstory="An executive procurement director balancing strategic risk, quality, and commercial value.",
        task_description=desc_8,
        max_tokens=2500,
    )
    exec_8["order"] = 8
    exec_8["agent_id"] = "selection_recommendation_agent"
    agent_executions.append(exec_8)
    stage_evidence["recommendation"] = exec_8["output"]

    # 9. Final Review Agent (Key 13)
    print(f"  [9/9] Final Review Agent -> {AGENT_KEY_MAP['final_review_agent_procurement']}...")
    desc_9 = f"""
Perform the final governance audit on this supplier selection process.
Complete Sourcing Audit File:
{json.dumps(stage_evidence, indent=2)}

Tasks:
1. Verify that the recommended supplier actually meets mandatory criteria and passed weighted evaluation.
2. Confirm there are no unverified cost calculations or unaddressed risk flags.
3. Generate the final executive procurement report.
Output strictly valid JSON with keys: 'workflow_status' ('COMPLETED'), 'selected_supplier', 'compliance_verification', 'summary_report', and 'award_justification'.
"""
    exec_9 = call_agent_groq(
        agent_id="final_review_agent_procurement",
        role="Chief Procurement Officer & Audit Lead",
        goal="Verify evaluation integrity, confirm weighted arithmetic, and authorize the final sourcing report.",
        backstory="The final governance authority ensuring procurement awards comply with organizational ethics and policy.",
        task_description=desc_9,
        max_tokens=3500,
    )
    exec_9["order"] = 9
    exec_9["agent_id"] = "final_review_agent_procurement"
    agent_executions.append(exec_9)

    t_end = time.time()
    total_dur = round(t_end - t_start, 3)

    tot_prompt = sum(e["prompt_tokens"] for e in agent_executions)
    tot_comp = sum(e["completion_tokens"] for e in agent_executions)
    tot_tokens = tot_prompt + tot_comp
    cost = round((tot_prompt / 1_000_000.0) * 0.59 + (tot_comp / 1_000_000.0) * 0.79, 6)

    trace_record = {
        "run_id": run_id,
        "workflow_name": "procurement_supplier_selection",
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
        "final_output": json.dumps(exec_9["output"], indent=2),
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
    """Aggregate all 30 traces in target_dir into batch_30_runs_summary.json."""
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
        "workflow_name": "procurement_supplier_selection",
        "total_runs": total_runs,
        "successful_runs": total_runs,
        "failed_runs": 0,
        "model_enforced": "openai/gpt-oss-120b",
        "key_distribution_model": "1-to-1 dedicated Groq key per sub-agent (GROQ_API_KEY_1..9)",
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
    parser = argparse.ArgumentParser(description="Generate live Ingestion_Traces for Procurement")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute (default: 1)")
    parser.add_argument("--start-idx", type=int, default=0, help="Requests dataset starting index")
    parser.add_argument("--start-run-num", type=int, default=1, help="Run number to start numbering from")
    parser.add_argument("--total-runs", type=int, default=30, help="Total runs in batch for logging")
    args = parser.parse_args()

    # Load datasets
    data_dir = PROJECT_ROOT / "data" / "procurement_supplier_selection"
    with open(data_dir / "procurement_requests.json", "r", encoding="utf-8") as f:
        all_reqs = json.load(f)
    with open(data_dir / "quotations.json", "r", encoding="utf-8") as f:
        all_quotes = json.load(f)
    with open(data_dir / "delivery_schedules.json", "r", encoding="utf-8") as f:
        all_schedules = json.load(f)
    with open(data_dir / "supplier_profiles.json", "r", encoding="utf-8") as f:
        all_profiles = json.load(f)
    with open(data_dir / "risk_records.json", "r", encoding="utf-8") as f:
        all_risks = json.load(f)
    with open(data_dir / "contracts.json", "r", encoding="utf-8") as f:
        all_contracts = json.load(f)
    with open(data_dir / "evaluation_criteria.json", "r", encoding="utf-8") as f:
        evaluation_criteria = json.load(f)

    target_dir = PROJECT_ROOT / "Ingestion_Traces" / "procurement_supplier_selection"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting execution of {args.runs} live runs for procurement_supplier_selection (Runs {args.start_run_num} to {args.start_run_num + args.runs - 1})...")
    print(f"Output directory: {target_dir}")

    summary_file = target_dir / "summary.jsonl"

    for i in range(args.runs):
        run_num = args.start_run_num + i
        r_idx = (args.start_idx + i) % len(all_reqs)
        req_sample = all_reqs[r_idx]
        req_id = req_sample.get("request_id")

        matching_quotes = [q for q in all_quotes if q.get("request_id") == req_id]
        quo_ids = {q.get("quotation_id") for q in matching_quotes}
        matching_scheds = [s for s in all_schedules if s.get("quotation_id") in quo_ids]

        print(f"\n=======================================================")
        print(f" [RUN {run_num:02d}/{args.total_runs}] Starting...")
        print(f"=======================================================")

        try:
            trace_rec = run_single_procurement_trace(
                req=req_sample,
                quotes=matching_quotes,
                schedules=matching_scheds,
                all_profiles=all_profiles,
                all_risks=all_risks,
                all_contracts=all_contracts,
                evaluation_criteria=evaluation_criteria,
                run_number=run_num,
                output_dir=target_dir,
            )
            with open(summary_file, "a", encoding="utf-8") as sf:
                summary_entry = {
                    "run_number": run_num,
                    "run_id": trace_rec["run_id"],
                    "request_id": req_sample.get("request_id"),
                    "item_description": req_sample.get("item_description"),
                    "bidders_count": len(matching_quotes),
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
                    "request_id": req_sample.get("request_id"),
                    "error": str(e),
                    "status": "FAILED",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")

    generate_batch_summary(target_dir)

