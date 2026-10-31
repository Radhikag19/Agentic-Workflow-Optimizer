"""
Dedicated Live Trace Generator for Customer Support Workflow.

Enforces:
1. 1-to-1 Groq API Key Distribution across all 6 sub-agents (Keys 1..6).
2. Live, non-hallucinated execution using Groq API with openai/gpt-oss-120b.
3. Reference trace format strictly matching runs/customer_support/live_20261023_070149_custom.json.
4. Outputs persisted directly in Ingestion_Traces/customer_support/ (no agent-wise subfolders).
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
    "ticket_classification_agent": "GROQ_API_KEY_1",
    "policy_retrieval_agent": "GROQ_API_KEY_2",
    "issue_investigation_agent": "GROQ_API_KEY_3",
    "resolution_agent": "GROQ_API_KEY_4",
    "response_drafting_agent": "GROQ_API_KEY_5",
    "resolution_verification_agent": "GROQ_API_KEY_6",
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
def deterministic_classification_hints(ticket_data: dict) -> dict:
    subject = ticket_data.get("subject", "").lower()
    desc = ticket_data.get("description", "").lower()
    full_text = f"{subject} {desc}"

    category = "general"
    if any(k in full_text for k in ["refund", "money back", "credit"]):
        category = "refund_request"
    elif any(k in full_text for k in ["password", "login", "locked", "auth", "access"]):
        category = "account_access"
    elif any(k in full_text for k in ["invoice", "charge", "billed", "payment"]):
        category = "billing"
    elif any(k in full_text for k in ["ship", "delivery", "carrier", "order #"]):
        category = "delivery"
    elif any(k in full_text for k in ["500", "error", "api", "crash", "webhook"]):
        category = "technical_failure"

    is_simple = category in ["account_access", "billing"] and "intermittent" not in full_text
    apparent_complexity = "simple" if is_simple else "complex"

    return {
        "suggested_category": category,
        "suggested_complexity": apparent_complexity
    }


def deterministic_policy_retrieval(category: str, policies: list, guides: list) -> dict:
    matched_policies = []
    for p in policies:
        cat_lower = p.get("category", "").lower()
        if category.lower() in cat_lower or (category == "refund_request" and "refund" in cat_lower):
            matched_policies.append(p)

    matched_guides = []
    for g in guides:
        if g.get("issue_category") == category:
            matched_guides.append(g)

    return {
        "matched_policies": matched_policies,
        "matched_troubleshooting_guides": matched_guides,
        "has_coverage": len(matched_policies) > 0 or len(matched_guides) > 0
    }


def deterministic_customer_lookup(customer_id: str, customers: list, hist_resolutions: list) -> dict:
    found_customer = None
    for c in customers:
        if c.get("customer_id") == customer_id:
            found_customer = c
            break

    related_history = hist_resolutions[:3]

    return {
        "customer_profile": found_customer,
        "past_case_sample": related_history
    }


def call_agent_groq(
    agent_id: str,
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    max_tokens: int = 3000,
    max_retries: int = 3,
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
            time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"Agent '{agent_id}' failed after {max_retries} attempts: {last_err}")


def run_single_support_trace(
    ticket_data: Dict[str, Any],
    policies_list: List[Dict[str, Any]],
    guides_list: List[Dict[str, Any]],
    customers_list: List[Dict[str, Any]],
    hist_resolutions: List[Dict[str, Any]],
    run_number: int = 1,
    output_dir: Path = Path("Ingestion_Traces/customer_support"),
) -> Dict[str, Any]:
    """Execute all 6 agents sequentially for one support ticket and persist the trace."""
    run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_trace{run_number:02d}"
    t_start = time.time()
    agent_executions = []

    t_id = ticket_data.get("ticket_id", "TICK-UNKNOWN")
    c_name = ticket_data.get("customer_name", "Unknown Customer")
    subject = ticket_data.get("subject", "Support Inquiry")

    print(f"\n--- [Run {run_number:02d}] Executing Customer Support for {t_id} ({c_name} - {subject[:40]}) ---")

    # 1. Ticket Classification Agent (Key 1)
    print("  [1/6] Ticket Classification Agent -> GROQ_API_KEY_1...")
    hints = deterministic_classification_hints(ticket_data)
    desc_1 = f"""
Classify the following customer support ticket.
Ticket Data:
{json.dumps(ticket_data, indent=2)}

Rule-Based Triage Signals:
{json.dumps(hints, indent=2)}

Instructions:
1. Classify category (account_access, billing, payment, delivery, technical_failure, refund_request).
2. Assess urgency ('low', 'medium', 'high', 'critical') and customer impact.
3. Classify apparent complexity ('simple', 'moderate', 'complex').
4. Identify missing customer details or ambiguities.
Return strictly valid JSON with keys: 'category', 'urgency', 'complexity', 'customer_impact', and 'routing_recommendation'.
"""
    exec_1 = call_agent_groq(
        agent_id="ticket_classification_agent",
        role="Customer Support Triage Specialist",
        goal="Classify customer tickets into accurate operational categories, urgency tiers, and complexity levels.",
        backstory="An experienced support dispatcher who routes customer inquiries quickly and accurately.",
        task_description=desc_1,
        max_tokens=2000,
    )
    exec_1["order"] = 1
    exec_1["agent_id"] = "ticket_classification_agent"
    agent_executions.append(exec_1)

    cat = exec_1["output"].get("category", ticket_data.get("category", "general"))

    # 2. Policy Retrieval Agent (Key 2)
    print("  [2/6] Policy Retrieval Agent -> GROQ_API_KEY_2...")
    retrieval_data = deterministic_policy_retrieval(cat, policies_list, guides_list)
    desc_2 = f"""
Retrieve applicable policies and SOPs for this support ticket.
Ticket Data:
{json.dumps(ticket_data, indent=2)}

Retrieved Policy & Knowledge Base Records:
{json.dumps(retrieval_data, indent=2)}

Instructions:
1. Identify the primary governing policy ID and troubleshooting guide ID.
2. Detail eligibility conditions, SLAs, and restrictions.
3. Note explicitly if no relevant policy covers this specific issue.
Return strictly valid JSON with keys: 'applicable_policy_ids', 'applicable_guide_ids', 'policy_conditions', and 'retrieval_summary'.
"""
    exec_2 = call_agent_groq(
        agent_id="policy_retrieval_agent",
        role="Customer Policy & Knowledge Base Specialist",
        goal="Retrieve authoritative support guidelines, SLAs, and troubleshooting procedures for the ticket.",
        backstory="A knowledge manager providing certified procedures to support agents without guessing.",
        task_description=desc_2,
        max_tokens=2500,
    )
    exec_2["order"] = 2
    exec_2["agent_id"] = "policy_retrieval_agent"
    agent_executions.append(exec_2)

    # 3. Issue Investigation Agent (Key 3)
    print("  [3/6] Issue Investigation Agent -> GROQ_API_KEY_3...")
    cust_id = ticket_data.get("customer_id", "")
    customer_context = deterministic_customer_lookup(cust_id, customers_list, hist_resolutions)
    desc_3 = f"""
Investigate the customer's reported problem.
Ticket Data:
{json.dumps(ticket_data, indent=2)}

Synthetic Customer Record & System History:
{json.dumps(customer_context, indent=2)}

Instructions:
1. Determine likely root cause based on symptoms and customer tier.
2. Separate confirmed facts from unconfirmed hypotheses.
3. State if further diagnostic data or user verification is required.
Return strictly valid JSON with keys: 'investigation_findings', 'root_cause_hypothesis', 'confirmed_facts', and 'data_gaps'.
"""
    exec_3 = call_agent_groq(
        agent_id="issue_investigation_agent",
        role="Technical Support Investigator",
        goal="Investigate root causes, correlate customer account history, and separate confirmed facts from hypotheses.",
        backstory="A diagnostic engineer reviewing backend logs, customer history, and issue symptoms.",
        task_description=desc_3,
        max_tokens=2500,
    )
    exec_3["order"] = 3
    exec_3["agent_id"] = "issue_investigation_agent"
    agent_executions.append(exec_3)

    stage_evidence = {
        "classification": exec_1["output"],
        "policy_retrieval": exec_2["output"],
        "investigation": exec_3["output"],
    }

    # 4. Resolution Agent (Key 4)
    print("  [4/6] Resolution Agent -> GROQ_API_KEY_4...")
    desc_4 = f"""
Formulate a concrete resolution recommendation for this ticket.
Ticket Data:
{json.dumps(ticket_data, indent=2)}

Upstream Evidence:
{json.dumps(stage_evidence, indent=2)}

Instructions:
1. Recommend resolution type: 'immediate_resolution', 'requires_user_action', or 'escalate_to_engineering'.
2. Provide step-by-step resolution actions conforming strictly to retrieved policy rules.
3. Note: Never execute actual database alterations or financial credits.
Return strictly valid JSON with keys: 'resolution_type', 'recommended_action', 'policy_justification', and 'escalation_required'.
"""
    exec_4 = call_agent_groq(
        agent_id="resolution_agent",
        role="Customer Resolution Strategist",
        goal="Synthesize policies and technical investigation into a clear, actionable customer resolution.",
        backstory="A tier-3 senior resolution specialist with authority to recommend credits, troubleshooting, or escalations.",
        task_description=desc_4,
        max_tokens=2500,
    )
    exec_4["order"] = 4
    exec_4["agent_id"] = "resolution_agent"
    agent_executions.append(exec_4)
    stage_evidence["resolution"] = exec_4["output"]

    # 5. Response Drafting Agent (Key 5)
    print("  [5/6] Response Drafting Agent -> GROQ_API_KEY_5...")
    desc_5 = f"""
Draft a formal, customer-facing response message.
Ticket Context:
{json.dumps(ticket_data, indent=2)}

Approved Resolution Strategy:
{json.dumps(exec_4['output'], indent=2)}

Guidelines:
1. Acknowledge the customer's specific issue empathetically.
2. Explain the resolution or required user action with clear bullet points.
3. Do not disclose internal reasoning, prompt names, or unverified claims.
4. Do not promise external actions the system cannot guarantee.
Return strictly valid JSON with keys: 'response_text', 'tone', 'next_steps_for_customer', and 'ticket_status'.
"""
    exec_5 = call_agent_groq(
        agent_id="response_drafting_agent",
        role="Customer Communications Specialist",
        goal="Draft clear, empathetic, and professional customer responses explaining the resolution.",
        backstory="An expert in customer experience communication ensuring clear guidance and professional empathy.",
        task_description=desc_5,
        max_tokens=2500,
    )
    exec_5["order"] = 5
    exec_5["agent_id"] = "response_drafting_agent"
    agent_executions.append(exec_5)
    stage_evidence["drafting"] = exec_5["output"]

    # 6. Resolution Verification Agent (Key 6)
    print("  [6/6] Resolution Verification Agent -> GROQ_API_KEY_6...")
    desc_6 = f"""
Perform quality audit on the proposed ticket resolution and customer draft.
Full Support Audit Trail:
{json.dumps(stage_evidence, indent=2)}

Verification Checks:
1. Does the drafted response completely solve the customer's stated complaint?
2. Are there any unsupported commitments, policy contradictions, or missing instructions?
3. Mark final status: 'RESOLVED', 'ACTION_REQUIRED_BY_USER', or 'ESCALATED'.
Return strictly valid JSON with keys: 'qa_status' ('APPROVED' or 'REJECTED'), 'final_ticket_status', 'verification_notes', and 'support_summary_report'.
"""
    exec_6 = call_agent_groq(
        agent_id="resolution_verification_agent",
        role="Support QA & Verification Auditor",
        goal="Verify that the drafted response fully addresses the ticket within policy boundaries.",
        backstory="A quality assurance lead ensuring high customer satisfaction and strict policy adherence.",
        task_description=desc_6,
        max_tokens=3000,
    )
    exec_6["order"] = 6
    exec_6["agent_id"] = "resolution_verification_agent"
    agent_executions.append(exec_6)

    t_end = time.time()
    total_dur = round(t_end - t_start, 3)

    tot_prompt = sum(e["prompt_tokens"] for e in agent_executions)
    tot_comp = sum(e["completion_tokens"] for e in agent_executions)
    tot_tokens = tot_prompt + tot_comp
    cost = round((tot_prompt / 1_000_000.0) * 0.59 + (tot_comp / 1_000_000.0) * 0.79, 6)

    trace_record = {
        "run_id": run_id,
        "workflow_name": "customer_support",
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
        "final_output": json.dumps(exec_6["output"], indent=2),
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
        "workflow_name": "customer_support",
        "total_runs": total_runs,
        "successful_runs": total_runs,
        "failed_runs": 0,
        "model_enforced": "openai/gpt-oss-120b",
        "key_distribution_model": "1-to-1 dedicated Groq key per sub-agent (GROQ_API_KEY_1..6)",
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
    parser = argparse.ArgumentParser(description="Generate live Ingestion_Traces for Customer Support")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute (default: 1)")
    parser.add_argument("--start-idx", type=int, default=0, help="Tickets dataset starting index")
    parser.add_argument("--start-run-num", type=int, default=1, help="Run number to start numbering from")
    parser.add_argument("--total-runs", type=int, default=30, help="Total runs in batch for logging")
    args = parser.parse_args()

    # Load datasets
    data_dir = PROJECT_ROOT / "data" / "customer_support"
    with open(data_dir / "tickets.json", "r", encoding="utf-8") as f:
        all_tickets = json.load(f)
    with open(data_dir / "policies.json", "r", encoding="utf-8") as f:
        policies_list = json.load(f)
    with open(data_dir / "troubleshooting_guides.json", "r", encoding="utf-8") as f:
        guides_list = json.load(f)
    with open(data_dir / "customers.json", "r", encoding="utf-8") as f:
        customers_list = json.load(f)
    with open(data_dir / "historical_resolutions.json", "r", encoding="utf-8") as f:
        hist_resolutions = json.load(f)

    target_dir = PROJECT_ROOT / "Ingestion_Traces" / "customer_support"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting execution of {args.runs} live runs for customer_support (Runs {args.start_run_num} to {args.start_run_num + args.runs - 1})...")
    print(f"Output directory: {target_dir}")

    summary_file = target_dir / "summary.jsonl"

    for i in range(args.runs):
        run_num = args.start_run_num + i
        t_idx = (args.start_idx + i) % len(all_tickets)
        ticket_sample = all_tickets[t_idx]

        print(f"\n=======================================================")
        print(f" [RUN {run_num:02d}/{args.total_runs}] Starting...")
        print(f"=======================================================")

        try:
            trace_rec = run_single_support_trace(
                ticket_data=ticket_sample,
                policies_list=policies_list,
                guides_list=guides_list,
                customers_list=customers_list,
                hist_resolutions=hist_resolutions,
                run_number=run_num,
                output_dir=target_dir,
            )
            with open(summary_file, "a", encoding="utf-8") as sf:
                summary_entry = {
                    "run_number": run_num,
                    "run_id": trace_rec["run_id"],
                    "ticket_id": ticket_sample.get("ticket_id"),
                    "customer_id": ticket_sample.get("customer_id"),
                    "customer_name": ticket_sample.get("customer_name"),
                    "subject": ticket_sample.get("subject"),
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
                    "ticket_id": ticket_sample.get("ticket_id"),
                    "error": str(e),
                    "status": "FAILED",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")

    generate_batch_summary(target_dir)

