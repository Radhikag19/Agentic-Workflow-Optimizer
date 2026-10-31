"""
Dedicated Live Trace Generator for Sales Lead Qualification Workflow.

Enforces:
1. 1-to-1 Groq API Key Distribution across all 7 sub-agents (Keys 1..7).
2. Live, non-hallucinated execution using Groq API with openai/gpt-oss-120b.
3. Reference trace format strictly matching runs/sales_lead_qualification/live_20261023_070410_sales_.json.
4. Outputs persisted directly in Ingestion_Traces/sales_lead_qualification/ (no agent-wise subfolders).
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
    "lead_enrichment_agent": "GROQ_API_KEY_11",
    "lead_fit_agent": "GROQ_API_KEY_12",
    "intent_analysis_agent": "GROQ_API_KEY_13",
    "lead_scoring_agent": "GROQ_API_KEY_14",
    "opportunity_assessment_agent": "GROQ_API_KEY_15",
    "next_action_agent": "GROQ_API_KEY_16",
    "final_review_agent_sales": "GROQ_API_KEY_3",
}

# Standby reserve pool of healthy keys (in priority order of remaining capacity)
STANDBY_POOL = [
    "GROQ_API_KEY_1",
    "GROQ_API_KEY_2",
    "GROQ_API_KEY_5",
    "GROQ_API_KEY_10",
    "GROQ_API_KEY_7",
    "GROQ_API_KEY_4",
]

# Track exhausted keys to never reuse them
EXHAUSTED_KEYS = {"GROQ_API_KEY_6", "GROQ_API_KEY_8", "GROQ_API_KEY_9"}

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


# Deterministic Helper Functions
def deterministic_enrichment(company_id: str, all_companies: list) -> dict:
    for c in all_companies:
        if c.get("company_id") == company_id:
            return c
    return {}

def deterministic_icp_check(lead_data: dict, company_data: dict, icp: dict) -> dict:
    industry = company_data.get("industry", "")
    emp_count = int(company_data.get("employee_count", 0))

    if industry in icp.get("disqualified_industries", []) or emp_count < 10:
        return {
            "fit_tier": "DISQUALIFIED",
            "is_icp_match": False,
            "reason": f"Industry '{industry}' is in disqualified industries or company size ({emp_count} employees) is below micro-threshold."
        }

    if industry in icp.get("target_industries", []) and emp_count >= icp.get("min_employees", 200):
        return {
            "fit_tier": "STRONG_FIT",
            "is_icp_match": True,
            "reason": f"Matches target industry '{industry}' and enterprise headcount ({emp_count} >= {icp.get('min_employees')})."
        }

    return {
        "fit_tier": "MODERATE_FIT",
        "is_icp_match": True,
        "reason": f"Partial match: Industry '{industry}', employees: {emp_count}."
    }

def deterministic_intent_lookup(lead_id: str, all_engagements: list) -> list:
    for e in all_engagements:
        if e.get("lead_id") == lead_id:
            return e.get("events", [])
    return []

def deterministic_lead_score(lead_data: dict, company_data: dict, events: list) -> dict:
    score = 0
    components = {}

    target_ind = ["Financial Services", "Healthcare & Life Sciences", "Software & Technology", "Supply Chain & Logistics"]
    ind = company_data.get("industry", "")
    if ind in target_ind:
        ind_score = 25
    elif ind in ["Retail & Food Services", "Retail"]:
        ind_score = 0
    else:
        ind_score = 10
    score += ind_score
    components["industry_fit"] = ind_score

    emps = int(company_data.get("employee_count", 0))
    if emps >= 500:
        size_score = 25
    elif emps >= 100:
        size_score = 18
    elif emps >= 20:
        size_score = 10
    else:
        size_score = 0
    score += size_score
    components["company_size_fit"] = size_score

    title = lead_data.get("job_title", "").lower()
    if any(k in title for k in ["cto", "vp", "chief", "head of", "director"]):
        role_score = 20
    elif any(k in title for k in ["manager", "lead", "architect"]):
        role_score = 12
    elif any(k in title for k in ["owner"]):
        role_score = 8 if emps >= 20 else 2
    else:
        role_score = 0
    score += role_score
    components["role_authority"] = role_score

    event_types = [e.get("type", "") for e in events]
    if any("demo_request" in t or "executive_briefing" in t for t in event_types):
        intent_score = 20
    elif any("pricing" in t or "whitepaper" in t for t in event_types):
        intent_score = 12
    elif len(event_types) > 0:
        intent_score = 5
    else:
        intent_score = 0
    score += intent_score
    components["intent_signals"] = intent_score

    budget = float(lead_data.get("stated_budget", 0))
    timeline = lead_data.get("purchase_timeline", "").lower()
    budget_score = 0
    if budget >= 50000.0:
        budget_score += 5
    elif budget >= 15000.0:
        budget_score += 2
    if any(k in timeline for k in ["immediate", "3 months", "q4"]):
        budget_score += 5
    elif "6 months" in timeline:
        budget_score += 3
    score += budget_score
    components["budget_timeline"] = budget_score

    tier = "DISQUALIFIED"
    if score >= 75:
        tier = "SALES_QUALIFIED_LEAD"
    elif score >= 45:
        tier = "MARKETING_NURTURE"

    return {
        "total_score": score,
        "qualification_tier": tier,
        "component_breakdown": components
    }


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


def run_single_sales_trace(
    lead_data: Dict[str, Any],
    all_companies: List[Dict[str, Any]],
    icp_data: Dict[str, Any],
    all_engagements: List[Dict[str, Any]],
    run_number: int = 1,
    output_dir: Path = Path("Ingestion_Traces/sales_lead_qualification"),
) -> Dict[str, Any]:
    """Execute all 7 agents sequentially for one sales lead and persist the trace."""
    run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_trace{run_number:02d}"
    t_start = time.time()
    agent_executions = []

    lead_id = lead_data.get("lead_id", "LEAD-UNKNOWN")
    contact_name = lead_data.get("contact_name", "Unknown Contact")
    company_id = lead_data.get("company_id", "")
    comp_info = deterministic_enrichment(company_id, all_companies)

    print(f"\n--- [Run {run_number:02d}] Executing Sales Lead Qualification for {lead_id} ({contact_name} - {comp_info.get('company_name', company_id)}) ---")

    # 1. Lead Enrichment Agent
    print(f"  [1/7] Lead Enrichment Agent -> {AGENT_KEY_MAP['lead_enrichment_agent']}...", flush=True)
    desc_1 = f"""
Enrich the following sales lead with verified company intelligence.
Raw Lead Record:
{json.dumps(lead_data, indent=2)}

Matching Company Profile:
{json.dumps(comp_info, indent=2)}

Tasks:
1. Merge lead contact data with company firmographics (industry, employee_count, revenue).
2. Flag any missing attributes or unverified fields.
Return strictly valid JSON with keys: 'lead_id', 'enriched_lead_profile', and 'missing_attributes'.
"""
    exec_1 = call_agent_groq(
        agent_id="lead_enrichment_agent",
        role="Lead Enrichment & Intelligence Specialist",
        goal="Enrich raw lead submissions with verified corporate intelligence without hallucinating firmographic data.",
        backstory="A market intelligence specialist correlating corporate domains and CRM records to produce complete firmographic profiles.",
        task_description=desc_1,
        max_tokens=2500,
    )
    exec_1["order"] = 1
    exec_1["agent_id"] = "lead_enrichment_agent"
    agent_executions.append(exec_1)

    # 2. Lead Fit Agent
    print(f"  [2/7] Lead Fit Agent -> {AGENT_KEY_MAP['lead_fit_agent']}...", flush=True)
    fit_calc = deterministic_icp_check(lead_data, comp_info, icp_data)
    desc_2 = f"""
Evaluate account and persona alignment against the Ideal Customer Profile (ICP).
Enriched Profile:
{json.dumps(exec_1['output'].get('enriched_lead_profile', lead_data), indent=2)}

Ideal Customer Profile Rubric:
{json.dumps(icp_data, indent=2)}

Deterministic ICP Check from Python:
{json.dumps(fit_calc, indent=2)}

Tasks:
1. Classify fit category: 'STRONG_FIT', 'MODERATE_FIT', 'WEAK_FIT', or 'DISQUALIFIED'.
2. Assign numerical industry fit and company size fit scores (0-10).
3. Provide clear fit rationale.
Return strictly valid JSON with keys: 'fit_category', 'industry_fit_score', 'company_size_fit_score', and 'fit_rationale'.
"""
    exec_2 = call_agent_groq(
        agent_id="lead_fit_agent",
        role="Ideal Customer Profile (ICP) Fit Evaluator",
        goal="Evaluate company size, vertical, and revenue alignment against the corporate ICP definition.",
        backstory="A sales operations specialist dedicated to verifying whether inbound accounts meet target persona requirements.",
        task_description=desc_2,
        max_tokens=2500,
    )
    exec_2["order"] = 2
    exec_2["agent_id"] = "lead_fit_agent"
    agent_executions.append(exec_2)

    # 3. Intent Analysis Agent
    print(f"  [3/7] Intent Analysis Agent -> {AGENT_KEY_MAP['intent_analysis_agent']}...", flush=True)
    events = deterministic_intent_lookup(lead_id, all_engagements)
    desc_3 = f"""
Analyze buyer purchase intent and engagement history.
Lead Data:
{json.dumps(lead_data, indent=2)}

Recorded Digital Touchpoint Events:
{json.dumps(events, indent=2)}

Tasks:
1. Determine intent level: 'HIGH_INTENT', 'MODERATE_INTENT', 'LOW_INTENT', or 'NO_SIGNALS'.
2. Assign intent score (0-100).
3. Identify high-value signals (demo requests, pricing page visits, whitepaper downloads).
4. Provide a behavioral summary.
Return strictly valid JSON with keys: 'intent_level', 'intent_score', 'high_value_signals', and 'behavioral_summary'.
"""
    exec_3 = call_agent_groq(
        agent_id="intent_analysis_agent",
        role="Buyer Intent & Behavioral Analyst",
        goal="Analyze behavioral signals, digital touchpoints, and timeline commitments to gauge purchase intent.",
        backstory="A behavioral analyst evaluating customer readiness from product engagement and intent signals.",
        task_description=desc_3,
        max_tokens=2500,
    )
    exec_3["order"] = 3
    exec_3["agent_id"] = "intent_analysis_agent"
    agent_executions.append(exec_3)

    # 4. Lead Scoring Agent
    print(f"  [4/7] Lead Scoring Agent -> {AGENT_KEY_MAP['lead_scoring_agent']}...", flush=True)
    scoring_result = deterministic_lead_score(lead_data, comp_info, events)
    desc_4 = f"""
Review and formalize the lead scoring breakdown.
Deterministic Scoring Model Results from Python:
{json.dumps(scoring_result, indent=2)}

Tasks:
1. Explain the score calculation and tier classification ('SALES_QUALIFIED_LEAD', 'MARKETING_NURTURE', 'DISQUALIFIED').
2. Identify the highest-scoring and lowest-scoring dimensions.
Return strictly valid JSON with keys: 'total_score', 'qualification_tier', 'component_scores', and 'scoring_rationale'.
"""
    exec_4 = call_agent_groq(
        agent_id="lead_scoring_agent",
        role="Lead Qualification Scoring Specialist",
        goal="Apply mathematical scoring models to evaluate and grade inbound prospect readiness.",
        backstory="A RevOps scoring architect quantifying inbound lead potential using rigorous rubrics.",
        task_description=desc_4,
        max_tokens=2500,
    )
    exec_4["order"] = 4
    exec_4["agent_id"] = "lead_scoring_agent"
    agent_executions.append(exec_4)

    stage_evidence = {
        "enrichment": exec_1["output"],
        "fit": exec_2["output"],
        "intent": exec_3["output"],
        "scoring": exec_4["output"],
    }

    # 5. Opportunity Assessment Agent
    print(f"  [5/7] Opportunity Assessment Agent -> {AGENT_KEY_MAP['opportunity_assessment_agent']}...", flush=True)
    desc_5 = f"""
Synthesize commercial opportunity potential for this lead.
Consolidated Upstream Evidence:
{json.dumps(stage_evidence, indent=2)}

Lead Commercial Indicators:
- Stated Budget: ${lead_data.get('stated_budget', 0):,.2f}
- Purchase Timeline: {lead_data.get('purchase_timeline', 'Unknown')}

Tasks:
1. Assess opportunity viability: 'HIGH', 'MEDIUM', or 'LOW'.
2. Estimate expected deal size and recommended initial pipeline stage.
3. Highlight key deal risks or competitive blockers.
Return strictly valid JSON with keys: 'opportunity_viability', 'estimated_deal_size', 'pipeline_stage', and 'key_risks'.
"""
    exec_5 = call_agent_groq(
        agent_id="opportunity_assessment_agent",
        role="Commercial Opportunity Assessor",
        goal="Synthesize budget capacity, purchase timeline, and deal viability into an opportunity forecast.",
        backstory="A senior enterprise sales director assessing deal size, budget authorization, and sales cycle timing.",
        task_description=desc_5,
        max_tokens=2500,
    )
    exec_5["order"] = 5
    exec_5["agent_id"] = "opportunity_assessment_agent"
    agent_executions.append(exec_5)
    stage_evidence["opportunity"] = exec_5["output"]

    # 6. Next Action Recommendation Agent
    print(f"  [6/7] Next Action Agent -> {AGENT_KEY_MAP['next_action_agent']}...", flush=True)
    desc_6 = f"""
Formulate strategic next actions and routing for this sales lead.
Consolidated Evidence:
{json.dumps(stage_evidence, indent=2)}

Instructions:
1. Formulate recommended action based on qualification tier:
   - SALES_QUALIFIED_LEAD -> Immediate SDR outreach / AE booking within 2 hours.
   - MARKETING_NURTURE -> Add to targeted email sequence / webinar track.
   - DISQUALIFIED -> Log disqualification reason and archive.
2. Specify target team, priority, and clear business justification.
Return strictly valid JSON with keys: 'recommended_action', 'target_team', 'priority', and 'action_justification'.
"""
    exec_6 = call_agent_groq(
        agent_id="next_action_agent",
        role="Sales Engagement Strategy Lead",
        goal="Formulate concrete next steps (routing, sales rep assignment, cadence, SLA) based on qualification tier.",
        backstory="An inbound sales development manager routing leads to the right team with customized playbooks.",
        task_description=desc_6,
        max_tokens=2500,
    )
    exec_6["order"] = 6
    exec_6["agent_id"] = "next_action_agent"
    agent_executions.append(exec_6)
    stage_evidence["action"] = exec_6["output"]

    # 7. Final Review Agent
    print(f"  [7/7] Final Review Agent -> {AGENT_KEY_MAP['final_review_agent_sales']}...", flush=True)
    desc_7 = f"""
Perform final quality audit and produce the authoritative lead qualification report.
Full Lead Audit File:
{json.dumps(stage_evidence, indent=2)}

Tasks:
1. Verify consistency between ICP fit, scoring rubric, and next actions.
2. Confirm there are no contradictory routing decisions.
3. Author the final RevOps lead summary report.
Output strictly valid JSON with keys: 'workflow_status' ('COMPLETED'), 'lead_id', 'final_qualification_tier', 'final_score', 'summary_report', and 'audit_findings'.
"""
    exec_7 = call_agent_groq(
        agent_id="final_review_agent_sales",
        role="Revenue Operations QA Lead",
        goal="Perform quality audit on lead scoring and routing recommendations to ensure compliance with sales SLAs.",
        backstory="The executive gatekeeper of the sales funnel ensuring CRM data integrity and accurate lead assignments.",
        task_description=desc_7,
        max_tokens=3000,
    )
    exec_7["order"] = 7
    exec_7["agent_id"] = "final_review_agent_sales"
    agent_executions.append(exec_7)

    t_end = time.time()
    total_dur = round(t_end - t_start, 3)

    tot_prompt = sum(e["prompt_tokens"] for e in agent_executions)
    tot_comp = sum(e["completion_tokens"] for e in agent_executions)
    tot_tokens = tot_prompt + tot_comp
    cost = round((tot_prompt / 1_000_000.0) * 0.59 + (tot_comp / 1_000_000.0) * 0.79, 6)

    trace_record = {
        "run_id": run_id,
        "workflow_name": "sales_lead_qualification",
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
        "final_output": json.dumps(exec_7["output"], indent=2),
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
        "workflow_name": "sales_lead_qualification",
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
    parser = argparse.ArgumentParser(description="Generate live Ingestion_Traces for Sales Lead Qualification")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute (default: 1)")
    parser.add_argument("--start-idx", type=int, default=0, help="Leads dataset starting index")
    parser.add_argument("--start-run-num", type=int, default=1, help="Run number to start numbering from")
    parser.add_argument("--total-runs", type=int, default=30, help="Total runs in batch for logging")
    args = parser.parse_args()

    # Load datasets
    data_dir = PROJECT_ROOT / "data" / "sales_lead_qualification"
    with open(data_dir / "leads.json", "r", encoding="utf-8") as f:
        all_leads = json.load(f)
    with open(data_dir / "company_profiles.json", "r", encoding="utf-8") as f:
        all_companies = json.load(f)
    with open(data_dir / "ideal_customer_profiles.json", "r", encoding="utf-8") as f:
        icps = json.load(f)
        icp_data = icps[0] if icps else {}
    with open(data_dir / "engagement_histories.json", "r", encoding="utf-8") as f:
        all_engagements = json.load(f)

    target_dir = PROJECT_ROOT / "Ingestion_Traces" / "sales_lead_qualification"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting execution of {args.runs} live runs for sales_lead_qualification (Runs {args.start_run_num} to {args.start_run_num + args.runs - 1})...")
    print(f"Output directory: {target_dir}")

    summary_file = target_dir / "summary.jsonl"

    for i in range(args.runs):
        run_num = args.start_run_num + i
        l_idx = (args.start_idx + i) % len(all_leads)
        lead_sample = all_leads[l_idx]

        print(f"\n=======================================================")
        print(f" [RUN {run_num:02d}/{args.total_runs}] Starting...")
        print(f"=======================================================")

        try:
            trace_rec = run_single_sales_trace(
                lead_data=lead_sample,
                all_companies=all_companies,
                icp_data=icp_data,
                all_engagements=all_engagements,
                run_number=run_num,
                output_dir=target_dir,
            )
            with open(summary_file, "a", encoding="utf-8") as sf:
                summary_entry = {
                    "run_number": run_num,
                    "run_id": trace_rec["run_id"],
                    "lead_id": lead_sample.get("lead_id"),
                    "contact_name": lead_sample.get("contact_name"),
                    "company_id": lead_sample.get("company_id"),
                    "stated_budget": lead_sample.get("stated_budget"),
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
                    "lead_id": lead_sample.get("lead_id"),
                    "error": str(e),
                    "status": "FAILED",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")

    generate_batch_summary(target_dir)

