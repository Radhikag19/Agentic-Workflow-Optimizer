"""
Dedicated Live Trace Generator for Expense Reimbursement Workflow.

Enforces:
1. 1-to-1 Groq API Key Distribution across all 8 sub-agents with dynamic auto-failover.
2. Live, non-hallucinated execution using Groq API with openai/gpt-oss-120b.
3. Reference trace format strictly matching runs/expense_reimbursement/live_20261023_070709_expens.json.
4. Outputs persisted directly in Ingestion_Traces/expense_reimbursement/ (no agent-wise subfolders).
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

# 1-to-1 Sub-Agent to Groq API Key Mapping
AGENT_KEY_MAP = {
    "receipt_agent": "GROQ_API_KEY_20",
    "normalize_agent": "GROQ_API_KEY_18",
    "category_agent": "GROQ_API_KEY_17",
    "amount_agent": "GROQ_API_KEY_14",
    "policy_agent": "GROQ_API_KEY_19",
    "duplicate_agent": "GROQ_API_KEY_7",
    "exception_agent": "GROQ_API_KEY_10",
    "expense_review_agent": "GROQ_API_KEY_13",
}

# Standby reserve pool of healthy keys (in priority order of remaining capacity)
STANDBY_POOL = [
    "GROQ_API_KEY_16",
    "GROQ_API_KEY_4",
    "GROQ_API_KEY_11",
    "GROQ_API_KEY_3",
    "GROQ_API_KEY_9",
    "GROQ_API_KEY_1",
]

# Track exhausted keys to never reuse them
EXHAUSTED_KEYS = {
    "GROQ_API_KEY_12",
    "GROQ_API_KEY_8",
    "GROQ_API_KEY_2",
    "GROQ_API_KEY_5",
    "GROQ_API_KEY_15",
    "GROQ_API_KEY_6",
}

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
def calculate_deterministic_arithmetic(subtotal: float, tax: float, total: float) -> Dict[str, Any]:
    expected_total = round(subtotal + tax, 2)
    difference = round(abs(expected_total - round(total, 2)), 2)
    is_valid = difference <= 0.02

    return {
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "declared_total": round(total, 2),
        "computed_total": expected_total,
        "discrepancy": difference,
        "is_arithmetically_valid": is_valid,
        "has_negative_values": any(v < 0 for v in [subtotal, tax, total]),
    }


def query_policies_for_category(category: str, policies_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matches = [p for p in policies_list if p.get("category", "").lower() in category.lower() or category.lower() in p.get("category", "").lower()]
    return matches if matches else policies_list[:2]


def query_candidate_duplicates(employee_id: str, merchant: str, amount: float, historical_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = []
    for sub in historical_list:
        same_emp = sub.get("employee_id") == employee_id
        same_merchant = sub.get("merchant", "").lower() == merchant.lower()
        same_amount = abs(float(sub.get("amount", 0.0)) - amount) < 0.05

        if (same_emp and same_amount) or (same_merchant and same_amount):
            candidates.append(sub)

    return candidates[:3]


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

        # If key is missing or marked exhausted, failover immediately
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
                "raw_text": raw_content,
                "notes": "",
            }
        except Exception as exc:
            last_err = exc
            err_str = str(exc)
            
            # Check if daily quota exhausted (TPD limit)
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
                continue
            elif "429" in err_str or "rate_limit" in err_str.lower():
                m_sec = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                wait_time = min(float(m_sec.group(1)) + 1.0, 10.0) if m_sec else 4.0
                print(f"    [Transient RateLimit] Agent '{agent_id}' waiting {wait_time:.1f}s...", flush=True)
                time.sleep(wait_time)
            else:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Agent '{agent_id}' failed after {max_retries} attempts: {last_err}")


def run_single_expense_trace(
    claim_input: Dict[str, Any],
    all_policies: List[Dict[str, Any]],
    historical_submissions: List[Dict[str, Any]],
    run_number: int = 1,
    output_dir: Path = Path("Ingestion_Traces/expense_reimbursement"),
) -> Dict[str, Any]:
    """Execute all 8 agents sequentially for one expense claim and persist trace."""
    run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_trace{run_number:02d}"
    t_start = time.time()
    agent_executions = []

    raw_receipt = claim_input.get("receipt_text", "")
    claim_id = claim_input.get("claim_id", f"CLM-{2000 + run_number}")
    employee_id = claim_input.get("employee_id", "EMP-000")
    merchant = claim_input.get("merchant", "Vendor")
    subtotal = float(claim_input.get("subtotal", 0.0))
    tax = float(claim_input.get("tax", 0.0))
    total = float(claim_input.get("total", 0.0))

    print(f"\n--- [Run {run_number:02d}] Executing Expense Reimbursement for {claim_id} ({merchant}) - '${total:.2f}' ---", flush=True)

    # 1. Receipt Agent
    print(f"  [1/8] Receipt Agent -> {AGENT_KEY_MAP['receipt_agent']}...", flush=True)
    desc_1 = f"""
Read and parse this synthetic receipt text:
{raw_receipt if raw_receipt else '[NO RECEIPT TEXT PROVIDED IN CLAIM]'}

Extract:
- merchant
- transaction_date
- currency
- subtotal
- tax
- total
- line_items: list of items with description, quantity, price
- missing_fields: list of any missing or illegible fields

Return strictly valid JSON matching:
{{
  "merchant": "...",
  "transaction_date": "YYYY-MM-DD",
  "currency": "USD",
  "subtotal": 0.0,
  "tax": 0.0,
  "total": 0.0,
  "line_items": [
    {{"description": "...", "quantity": 1, "price": 0.0}}
  ],
  "missing_fields": []
}}
"""
    exec_1 = call_agent_groq(
        agent_id="receipt_agent",
        role="Receipt Extraction Specialist",
        goal="Read and parse synthetic receipt text. Extract merchant, transaction_date, currency, subtotal, tax, total, and line items. Explicitly note missing fields.",
        backstory="An automated OCR and Receipt Parsing Specialist extracting structured financial facts from scanned or plaintext receipt text.",
        task_description=desc_1,
        max_tokens=2000,
    )
    exec_1["order"] = 1
    exec_1["agent_id"] = "receipt_agent"
    agent_executions.append(exec_1)
    receipt_res = exec_1["output"]
    receipt_text = json.dumps(receipt_res, indent=2)

    # 2. Normalize Agent
    print(f"  [2/8] Normalize Agent -> {AGENT_KEY_MAP['normalize_agent']}...", flush=True)
    desc_2 = f"""
Standardize and validate fields between claim and receipt:
Claim Payload:
{json.dumps(claim_input, indent=2)}

Receipt Output:
{receipt_text}

Format dates to YYYY-MM-DD, currency to ISO code (e.g., USD), and numeric amounts to 2 decimals.

Return strictly valid JSON matching:
{{
  "employee_id": "{employee_id}",
  "merchant": "...",
  "date": "YYYY-MM-DD",
  "currency": "USD",
  "subtotal": 0.0,
  "tax": 0.0,
  "total": 0.0
}}
"""
    exec_2 = call_agent_groq(
        agent_id="normalize_agent",
        role="Expense Normalization Specialist",
        goal="Standardize and validate fields between claim and receipt: format dates to YYYY-MM-DD, currency to ISO code, and numeric amounts to 2 decimals.",
        backstory="An Expense Normalization Engineer harmonizing diverse receipt formats into standardized corporate accounting records.",
        task_description=desc_2,
        max_tokens=2000,
    )
    exec_2["order"] = 2
    exec_2["agent_id"] = "normalize_agent"
    agent_executions.append(exec_2)
    normalize_res = exec_2["output"]
    normalize_text = json.dumps(normalize_res, indent=2)

    # 3. Category Agent
    print(f"  [3/8] Category Agent -> {AGENT_KEY_MAP['category_agent']}...", flush=True)
    desc_3 = f"""
Classify this expense into an official corporate category:
Allowed Categories: Travel, Meals & Entertainment, Accommodation, Transport, Office Supplies, Software & Cloud.

Normalized Context:
{normalize_text}

Original Claim Context:
{json.dumps({k: v for k, v in claim_input.items() if k != 'receipt_text'}, indent=2)}

Return strictly valid JSON matching:
{{
  "category": "Travel",
  "classification_confidence": 0.95,
  "reasoning": "Detailed justification citing merchant and line items."
}}
"""
    exec_3 = call_agent_groq(
        agent_id="category_agent",
        role="Expense Categorization Specialist",
        goal="Classify expense into official category (Travel, Meals & Entertainment, Accommodation, Transport, Office Supplies, Software & Cloud) and provide confidence and reasoning.",
        backstory="A Corporate Tax & Categorization Specialist who accurately assigns accounting ledger categories to business expenses.",
        task_description=desc_3,
        max_tokens=1800,
    )
    exec_3["order"] = 3
    exec_3["agent_id"] = "category_agent"
    agent_executions.append(exec_3)
    category_res = exec_3["output"]
    category_text = json.dumps(category_res, indent=2)

    # Extract assigned category for policy lookup
    assigned_category = category_res.get("category", claim_input.get("category", "Meals & Entertainment"))

    # 4. Amount Agent (Deterministic arithmetic + LLM verification)
    print(f"  [4/8] Amount Agent -> {AGENT_KEY_MAP['amount_agent']}...", flush=True)
    arithmetic_check = calculate_deterministic_arithmetic(subtotal, tax, total)
    desc_4 = f"""
Deterministic Arithmetic Verification:
{json.dumps(arithmetic_check, indent=2)}

Verify if subtotal (${subtotal:.2f}) + tax (${tax:.2f}) equals total (${total:.2f}).
Confirm whether any negative amounts or rounding anomalies exist.

Return strictly valid JSON matching:
{{
  "is_arithmetic_valid": {str(arithmetic_check['is_arithmetically_valid']).lower()},
  "discrepancy_usd": {arithmetic_check['discrepancy']},
  "verification_notes": "Verification details regarding subtotal, tax, total, and rounding."
}}
"""
    exec_4 = call_agent_groq(
        agent_id="amount_agent",
        role="Expense Arithmetic Specialist",
        goal="Verify all arithmetic calculations on the claim: subtotal + tax = total. Flag rounding errors, negative sums, or declared total mismatches.",
        backstory="A Forensic Financial Auditor who checks numerical calculations to the exact cent, ensuring books reconcile with mathematical precision.",
        task_description=desc_4,
        max_tokens=1800,
    )
    exec_4["order"] = 4
    exec_4["agent_id"] = "amount_agent"
    agent_executions.append(exec_4)
    amount_res = exec_4["output"]
    amount_text = json.dumps(amount_res, indent=2)

    # 5. Policy Agent (Deterministic lookup + LLM compliance check)
    print(f"  [5/8] Policy Agent -> {AGENT_KEY_MAP['policy_agent']}...", flush=True)
    applicable_policies = query_policies_for_category(assigned_category, all_policies)
    desc_5 = f"""
Evaluate this claim against corporate policy limits:
Applicable Policies:
{json.dumps(applicable_policies, indent=2)}

Claim Details:
Total: ${total:.2f}
Category: {assigned_category}
Has Receipt: {bool(raw_receipt)}
Business Purpose: {claim_input.get('business_purpose', '')}

Return which policy IDs apply, whether any limit is breached, and compliance status.

Return strictly valid JSON matching:
{{
  "policy_compliance_status": "Compliant" | "Non-compliant",
  "cited_policy_ids": ["POL-EXP-01"],
  "violation_reasons": []
}}
"""
    exec_5 = call_agent_groq(
        agent_id="policy_agent",
        role="Reimbursement Policy Compliance Specialist",
        goal="Apply corporate reimbursement policies against claim details. Check caps, receipt requirements, and advance approval rules. Return policy IDs cited.",
        backstory="A Corporate Compliance Officer with encyclopedic knowledge of company travel and expense policies, spending limits, and audit standards.",
        task_description=desc_5,
        max_tokens=2200,
    )
    exec_5["order"] = 5
    exec_5["agent_id"] = "policy_agent"
    agent_executions.append(exec_5)
    policy_res = exec_5["output"]
    policy_text = json.dumps(policy_res, indent=2)

    # 6. Duplicate Agent (Deterministic candidate lookup + LLM comparison)
    print(f"  [6/8] Duplicate Agent -> {AGENT_KEY_MAP['duplicate_agent']}...", flush=True)
    duplicate_candidates = query_candidate_duplicates(employee_id, merchant, total, historical_submissions)
    desc_6 = f"""
Check for duplicate submissions against historical database:
Candidate Submissions:
{json.dumps(duplicate_candidates, indent=2)}

Current Claim:
Employee: {employee_id}
Merchant: {merchant}
Total: ${total:.2f}
Date: {claim_input.get('submission_date', '')}

Determine if this is a "Confirmed Duplicate", "Possible Coincidence", or "No Duplicate Found".

Return strictly valid JSON matching:
{{
  "duplicate_status": "Confirmed Duplicate" | "Possible Coincidence" | "No Duplicate Found",
  "matching_claim_ids": ["HIST-CLM-2001"],
  "duplicate_risk_score": 0.95
}}
"""
    exec_6 = call_agent_groq(
        agent_id="duplicate_agent",
        role="Duplicate Claim Detection Specialist",
        goal="Cross-reference active claim against historical submissions. Distinguish between confirmed duplicate submissions and coincidentally similar expenses.",
        backstory="A Fraud Prevention Analyst who scans transaction histories to prevent double-reimbursements and duplicate receipt submissions across accounting periods.",
        task_description=desc_6,
        max_tokens=2000,
    )
    exec_6["order"] = 6
    exec_6["agent_id"] = "duplicate_agent"
    agent_executions.append(exec_6)
    duplicate_res = exec_6["output"]
    duplicate_text = json.dumps(duplicate_res, indent=2)

    # 7. Exception Agent (Independent Red-Team Audit)
    print(f"  [7/8] Exception Agent -> {AGENT_KEY_MAP['exception_agent']}...", flush=True)
    desc_7 = f"""
Conduct an independent exception audit across all findings:
Receipt Info: {receipt_text}
Arithmetic: {amount_text}
Policy Check: {policy_text}
Duplicate Check: {duplicate_text}

Identify all anomalies, unitemized receipts, policy breaches, or red flags.

Return strictly valid JSON matching:
{{
  "exceptions_found": [
    {{
      "type": "Excessive Amount" | "Policy Violation" | "Duplicate Claim" | "Unitemized Receipt",
      "detail": "Description of exception",
      "policy_id": "POL-EXP-02",
      "severity": "Critical" | "High" | "Medium" | "Low"
    }}
  ],
  "severity_levels": {{
    "Critical": 0,
    "High": 0,
    "Medium": 0,
    "Low": 0
  }},
  "audit_warnings": [
    "Warning message 1"
  ]
}}
"""
    exec_7 = call_agent_groq(
        agent_id="exception_agent",
        role="Expense Exception & Audit Specialist",
        goal="Conduct an independent exception audit across all findings. Identify all anomalies, unitemized receipts, policy breaches, or red flags.",
        backstory="An Independent Internal Auditor specializing in fraud detection, risk exposure assessment, and compliance discrepancies.",
        task_description=desc_7,
        max_tokens=2500,
    )
    exec_7["order"] = 7
    exec_7["agent_id"] = "exception_agent"
    agent_executions.append(exec_7)
    exception_res = exec_7["output"]
    exception_text = json.dumps(exception_res, indent=2)

    # 8. Review Agent (Final Decision Synthesis)
    print(f"  [8/8] Review Agent -> {AGENT_KEY_MAP['expense_review_agent']}...", flush=True)
    desc_8 = f"""
Synthesize all preceding 7 stages into a final reimbursement decision:
1. Receipt: {receipt_text}
2. Normalization: {normalize_text}
3. Category: {category_text}
4. Amount: {amount_text}
5. Policy: {policy_text}
6. Duplicate: {duplicate_text}
7. Exceptions: {exception_text}

Output final decision: ELIGIBLE, INELIGIBLE, or NEEDS_REVIEW. Include clear rationale.

Return strictly valid JSON matching:
{{
  "claim_id": "{claim_id}",
  "employee_id": "{employee_id}",
  "merchant": "{merchant}",
  "date": "{claim_input.get('submission_date', '')}",
  "currency": "USD",
  "total_amount": {total},
  "final_decision": "ELIGIBLE" | "INELIGIBLE" | "NEEDS_REVIEW",
  "rationale": [
    "Rationale statement 1",
    "Rationale statement 2"
  ],
  "cited_stages": {{
    "receipt_extraction": "Summary of stage 1 findings",
    "normalization": "Summary of stage 2 findings",
    "categorization": "Summary of stage 3 findings",
    "amount_check": "Summary of stage 4 findings",
    "policy_verification": "Summary of stage 5 findings",
    "duplicate_check": "Summary of stage 6 findings",
    "exception_audit": "Summary of stage 7 findings"
  }},
  "recommendation": "Final actionable recommendation for finance operations."
}}
"""
    exec_8 = call_agent_groq(
        agent_id="expense_review_agent",
        role="Expense Review & Adjudication Specialist",
        goal="Synthesize all preceding 7 stages into a final reimbursement decision: ELIGIBLE, INELIGIBLE, or NEEDS_REVIEW. Include clear rationale.",
        backstory="The Senior Finance Director with final signing authority on employee expense reimbursements.",
        task_description=desc_8,
        max_tokens=3000,
    )
    exec_8["order"] = 8
    exec_8["agent_id"] = "expense_review_agent"
    agent_executions.append(exec_8)
    review_res = exec_8["output"]

    t_end = time.time()
    total_dur = round(t_end - t_start, 3)

    tot_prompt = sum(e["prompt_tokens"] for e in agent_executions)
    tot_comp = sum(e["completion_tokens"] for e in agent_executions)
    tot_tokens = tot_prompt + tot_comp
    cost = round((tot_prompt / 1_000_000.0) * 0.59 + (tot_comp / 1_000_000.0) * 0.79, 6)

    trace_record = {
        "run_id": run_id,
        "workflow_name": "expense_reimbursement",
        "model": "openai/gpt-oss-120b",
        "status": "SUCCESS",
        "duration_seconds": total_dur,
        "agent_execution_count": len(agent_executions),
        "agent_executions": [
            {
                "order": e["order"],
                "agent_id": e["agent_id"],
                "assigned_key_identifier": e["assigned_key_identifier"],
                "duration_seconds": e["duration_seconds"],
                "status": e["status"],
                "tokens": e["tokens"],
                "prompt_tokens": e["prompt_tokens"],
                "completion_tokens": e["completion_tokens"],
                "output": e["output"],
                "notes": e["notes"],
            }
            for e in agent_executions
        ],
        "total_tokens": tot_tokens,
        "prompt_tokens": tot_prompt,
        "completion_tokens": tot_comp,
        "estimated_cost_usd": cost,
        "error_message": None,
        "final_output": json.dumps(review_res, indent=2),
        "output": {
            "workflow": "expense_reimbursement",
            "run_id": run_id,
            "input": claim_input,
            "stages": {
                "receipt": json.dumps(receipt_res, indent=2),
                "normalize": json.dumps(normalize_res, indent=2),
                "category": json.dumps(category_res, indent=2),
                "amount": json.dumps(amount_res, indent=2),
                "policy": json.dumps(policy_res, indent=2),
                "duplicate": json.dumps(duplicate_res, indent=2),
                "exceptions": json.dumps(exception_res, indent=2),
                "final_review": json.dumps(review_res, indent=2),
            },
            "synthetic": True,
        },
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
    """Aggregate all traces in target_dir into batch_30_runs_summary.json."""
    trace_files = sorted(list(target_dir.glob("live_*_trace*.json")))
    if not trace_files:
        return

    runs_summary = []
    total_tokens_all = 0
    total_cost_all = 0.0
    total_duration_all = 0.0
    key_counter = {}

    for tf in trace_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            run_item = {
                "run_id": data.get("run_id"),
                "status": data.get("status"),
                "duration_seconds": data.get("duration_seconds"),
                "agent_execution_count": data.get("agent_execution_count"),
                "total_tokens": data.get("total_tokens"),
                "prompt_tokens": data.get("prompt_tokens"),
                "completion_tokens": data.get("completion_tokens"),
                "estimated_cost_usd": data.get("estimated_cost_usd"),
                "timestamp": data.get("timestamp"),
            }
            runs_summary.append(run_item)
            total_tokens_all += data.get("total_tokens", 0)
            total_cost_all += data.get("estimated_cost_usd", 0.0)
            total_duration_all += data.get("duration_seconds", 0.0)

            for ag in data.get("agent_executions", []):
                k = ag.get("assigned_key_identifier", "Unknown")
                key_counter[k] = key_counter.get(k, 0) + 1
        except Exception as e:
            print(f"Error reading {tf}: {e}")

    summary_payload = {
        "workflow_name": "expense_reimbursement",
        "total_runs": len(runs_summary),
        "successful_runs": sum(1 for r in runs_summary if r["status"] == "SUCCESS"),
        "failed_runs": sum(1 for r in runs_summary if r["status"] != "SUCCESS"),
        "total_tokens_consumed": total_tokens_all,
        "total_estimated_cost_usd": round(total_cost_all, 6),
        "total_duration_seconds": round(total_duration_all, 3),
        "average_duration_per_run_seconds": round(total_duration_all / len(runs_summary), 3) if runs_summary else 0,
        "key_distribution_call_counts": key_counter,
        "runs": runs_summary,
    }

    summary_file = target_dir / "batch_30_runs_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"\n--> Successfully compiled batch summary: {summary_file}")

    # Append to summary.jsonl
    jsonl_file = target_dir / "summary.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in runs_summary:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run live traces for expense reimbursement workflow")
    parser.add_argument("--runs", type=int, default=30, help="Number of runs to execute in this batch")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index in expense_claims dataset")
    parser.add_argument("--start-run-num", type=int, default=1, help="Starting run number for filenames")
    parser.add_argument("--total-runs", type=int, default=30, help="Total runs expected in the workflow")
    args = parser.parse_args()

    # Load claims
    claims_path = PROJECT_ROOT / "data" / "expense_reimbursement" / "expense_claims.json"
    with open(claims_path, "r", encoding="utf-8") as f:
        claims_data = json.load(f)
        all_claims = claims_data.get("claims", [])

    # Load policies
    policies_path = PROJECT_ROOT / "data" / "expense_reimbursement" / "policies.json"
    with open(policies_path, "r", encoding="utf-8") as f:
        policies_data = json.load(f)
        all_policies = policies_data.get("policies", [])

    # Load historical submissions
    hist_path = PROJECT_ROOT / "data" / "expense_reimbursement" / "historical_submissions.json"
    with open(hist_path, "r", encoding="utf-8") as f:
        hist_data = json.load(f)
        historical_submissions = hist_data.get("historical_submissions", [])

    output_dir = PROJECT_ROOT / "Ingestion_Traces" / "expense_reimbursement"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting execution of {args.runs} live runs for expense_reimbursement (Runs {args.start_run_num} to {args.start_run_num + args.runs - 1})...")
    print(f"Output directory: {output_dir}\n")

    for i in range(args.runs):
        claim_idx = (args.start_idx + i) % len(all_claims)
        current_claim = all_claims[claim_idx]
        run_num = args.start_run_num + i

        print(f"=======================================================")
        print(f" [RUN {run_num:02d}/{args.total_runs}] Starting...")
        print(f"=======================================================")

        run_single_expense_trace(
            claim_input=current_claim,
            all_policies=all_policies,
            historical_submissions=historical_submissions,
            run_number=run_num,
            output_dir=output_dir,
        )

        generate_batch_summary(output_dir)
        time.sleep(1.0)


if __name__ == "__main__":
    main()

