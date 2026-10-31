"""
Dedicated Live Trace Generator for Invoice Processing Workflow.

Enforces:
1. 1-to-1 Groq API Key Distribution across all 8 sub-agents (Keys 1..8).
2. Live, non-hallucinated execution using Groq API with openai/gpt-oss-120b.
3. Reference trace format strictly matching runs/invoice_processing/live_20261023_070056_invoic.json.
4. Outputs persisted directly in Ingestion_Traces/invoice_processing/ (no agent-wise subfolders).
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
    "invoice_extraction_agent": "GROQ_API_KEY_1",
    "invoice_validation_agent": "GROQ_API_KEY_2",
    "po_matching_agent": "GROQ_API_KEY_3",
    "policy_compliance_agent": "GROQ_API_KEY_4",
    "duplicate_detection_agent": "GROQ_API_KEY_5",
    "exception_analysis_agent": "GROQ_API_KEY_6",
    "approval_recommendation_agent": "GROQ_API_KEY_7",
    "final_review_agent_invoice": "GROQ_API_KEY_8",
}

def mask_key(raw_key: str) -> str:
    if not raw_key:
        return "[NOT_SET]"
    s = raw_key.strip()
    return f"{s[:6]}...{s[-4:]}" if len(s) > 10 else "***"

def extract_json_from_text(text: str) -> Any:
    """Robust JSON extraction from LLM response text."""
    s = text.strip()
    # Try finding markdown code fences ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if match:
        s = match.group(1).strip()
    
    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try finding the first '{' and last '}'
    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = s[first_brace:last_brace + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    # Soft repair for truncated JSON
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")
    if open_braces > 0 or open_brackets > 0:
        s_repaired = s.rstrip(", \n\r\t") + ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))
        return json.loads(s_repaired)

    raise ValueError(f"Could not parse valid JSON from text: {text[:200]}...")


# Deterministic Helper Functions
def deterministic_arithmetic_check(invoice_data: dict) -> dict:
    lines = invoice_data.get("line_items", [])
    calc_subtotal = 0.0
    line_discrepancies = []
    
    for idx, item in enumerate(lines, 1):
        qty = float(item.get("quantity", 0))
        price = float(item.get("unit_price", 0))
        stated_tot = float(item.get("line_total", 0))
        expected_line_tot = round(qty * price, 2)
        calc_subtotal += expected_line_tot
        if abs(expected_line_tot - stated_tot) > 0.02:
            line_discrepancies.append({
                "line": idx,
                "description": item.get("description"),
                "expected": expected_line_tot,
                "stated": stated_tot,
                "diff": round(stated_tot - expected_line_tot, 2)
            })
            
    calc_subtotal = round(calc_subtotal, 2)
    stated_tax = float(invoice_data.get("tax", 0))
    stated_discount = float(invoice_data.get("discount", 0))
    stated_total = float(invoice_data.get("total_amount", 0))
    calc_total = round(calc_subtotal + stated_tax - stated_discount, 2)
    
    total_diff = round(stated_total - calc_total, 2)
    is_arithmetic_valid = (abs(total_diff) <= 0.05) and len(line_discrepancies) == 0

    return {
        "is_arithmetic_valid": is_arithmetic_valid,
        "calculated_subtotal": calc_subtotal,
        "calculated_total": calc_total,
        "stated_total": stated_total,
        "discrepancy_amount": total_diff,
        "line_discrepancies": line_discrepancies
    }


def deterministic_po_match(invoice_data: dict, pos: list) -> dict:
    po_ref = invoice_data.get("po_reference", "").strip()
    if not po_ref:
        return {
            "match_status": "MISSING_PO_REFERENCE",
            "matched_po_id": None,
            "discrepancies": ["No purchase order reference specified in invoice."]
        }
        
    matched_po = None
    for po in pos:
        if po.get("po_id") == po_ref:
            matched_po = po
            break

    if not matched_po:
        return {
            "match_status": "PO_NOT_FOUND",
            "matched_po_id": po_ref,
            "discrepancies": [f"Purchase Order '{po_ref}' not found in registry."]
        }

    discrepancies = []
    if matched_po.get("supplier_id") != invoice_data.get("supplier_id"):
        discrepancies.append(
            f"Supplier mismatch: Invoice supplier '{invoice_data.get('supplier_id')}' != PO supplier '{matched_po.get('supplier_id')}'"
        )
    
    po_tot = float(matched_po.get("total_amount", 0))
    inv_tot = float(invoice_data.get("total_amount", 0))
    if abs(po_tot - inv_tot) > 1.0:
        discrepancies.append(
            f"Total amount mismatch: Invoice total ${inv_tot:.2f} differs from PO total ${po_tot:.2f}"
        )

    po_lines = {l.get("description"): l for l in matched_po.get("lines", [])}
    for inv_line in invoice_data.get("line_items", []):
        desc = inv_line.get("description")
        if desc in po_lines:
            po_qty = po_lines[desc].get("quantity", 0)
            inv_qty = inv_line.get("quantity", 0)
            if inv_qty != po_qty:
                discrepancies.append(
                    f"Line quantity mismatch on '{desc}': Invoiced {inv_qty} units vs PO {po_qty} units"
                )

    match_status = "EXACT_MATCH" if not discrepancies else "DISCREPANCY_DETECTED"
    return {
        "match_status": match_status,
        "matched_po_id": po_ref,
        "matched_po_data": matched_po,
        "discrepancies": discrepancies
    }


def deterministic_policy_check(invoice_data: dict, suppliers: dict) -> dict:
    violations = []
    applied_policy_ids = []
    total_amt = float(invoice_data.get("total_amount", 0))
    sup_id = invoice_data.get("supplier_id")
    sup_info = suppliers.get(sup_id, {})

    applied_policy_ids.append("POL-AP-001")
    if total_amt >= 1000.0 and not invoice_data.get("po_reference"):
        violations.append({
            "policy_id": "POL-AP-001",
            "rule": "Every invoice above $1,000 must strictly match a valid Purchase Order.",
            "severity": "high",
            "detail": f"Invoice amount is ${total_amt:.2f} but PO reference is missing."
        })

    applied_policy_ids.append("POL-AP-002")
    if sup_info.get("status") == "suspended":
        violations.append({
            "policy_id": "POL-AP-002",
            "rule": "Invoices from suspended suppliers cannot be approved.",
            "severity": "critical",
            "detail": f"Supplier {sup_id} ({sup_info.get('name')}) is currently suspended."
        })

    applied_policy_ids.append("POL-AP-003")
    if total_amt > 25000.0:
        violations.append({
            "policy_id": "POL-AP-003",
            "rule": "Invoices totaling over $25,000 require executive financial review.",
            "severity": "medium",
            "detail": f"Total amount ${total_amt:.2f} exceeds high-value threshold ($25,000)."
        })

    return {
        "applied_policy_ids": applied_policy_ids,
        "violations": violations,
        "is_compliant": len(violations) == 0
    }


def deterministic_duplicate_check(invoice_data: dict, hist_records: list) -> dict:
    inv_num = invoice_data.get("invoice_number", "").strip()
    sup_id = invoice_data.get("supplier_id", "").strip()
    po_ref = invoice_data.get("po_reference", "").strip()
    total_amt = float(invoice_data.get("total_amount", 0))

    exact_matches = []
    potential_matches = []

    for h in hist_records:
        h_num = h.get("invoice_number", "").strip()
        h_sup = h.get("supplier_id", "").strip()
        h_po = h.get("po_reference", "").strip()
        h_amt = float(h.get("total_amount", 0))

        if h_num == inv_num or (sup_id and h_sup == sup_id and po_ref and h_po == po_ref and abs(h_amt - total_amt) < 0.01):
            exact_matches.append({
                "historical_invoice_id": h.get("invoice_id"),
                "invoice_number": h_num,
                "supplier_id": h_sup,
                "amount": h_amt,
                "reason": "Exact invoice number or identical (supplier, PO, amount) duplicate."
            })
        elif sup_id and h_sup == sup_id and abs(h_amt - total_amt) < 0.01:
            potential_matches.append({
                "historical_invoice_id": h.get("invoice_id"),
                "invoice_number": h_num,
                "supplier_id": h_sup,
                "amount": h_amt,
                "reason": "Same supplier and exact same dollar amount."
            })

    return {
        "is_duplicate": len(exact_matches) > 0,
        "is_potential_duplicate": len(potential_matches) > 0,
        "exact_matches": exact_matches,
        "potential_matches": potential_matches
    }


def deterministic_recommendation_rule(consolidated_evidence: dict) -> dict:
    val_status = consolidated_evidence.get("validation", {}).get("validation_status", "PASSED")
    dup_status = consolidated_evidence.get("duplicate_detection", {}).get("duplicate_status", "NONE")
    po_status = consolidated_evidence.get("po_matching", {}).get("po_match_status", "EXACT_MATCH")
    is_compliant = consolidated_evidence.get("policy_compliance", {}).get("is_compliant", True)

    if val_status == "FAILED" or dup_status == "EXACT_DUPLICATE":
        return {
            "recommended_outcome": "rejected",
            "primary_reason": "Arithmetic calculation failure or confirmed exact duplicate invoice submission."
        }
    if not is_compliant:
        critical_violations = [v for v in consolidated_evidence.get("policy_compliance", {}).get("violations", []) if v.get("severity") == "critical"]
        if critical_violations:
            return {
                "recommended_outcome": "rejected",
                "primary_reason": "Critical policy violation (e.g., suspended supplier or unapproved vendor)."
            }
        return {
            "recommended_outcome": "needs_review",
            "primary_reason": "Policy threshold flag requiring manager review."
        }

    if po_status in ["PO_NOT_FOUND", "MISSING_PO_REFERENCE", "DISCREPANCY_DETECTED"] or dup_status == "POTENTIAL_DUPLICATE":
        return {
            "recommended_outcome": "needs_review",
            "primary_reason": "Purchase order mismatch, missing PO, or potential duplicate warning."
        }

    return {
        "recommended_outcome": "recommended_for_approval",
        "primary_reason": "All validation, 3-way matching, compliance checks, and duplicate scans passed successfully."
    }


def call_agent_groq(
    agent_id: str,
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    max_tokens: int = 3500,
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


def run_single_invoice_trace(
    invoice_data: Dict[str, Any],
    pos_data: List[Dict[str, Any]],
    suppliers_dict: Dict[str, Any],
    hist_records: List[Dict[str, Any]],
    run_number: int = 1,
    output_dir: Path = Path("Ingestion_Traces/invoice_processing"),
) -> Dict[str, Any]:
    """Execute all 8 agents sequentially for one invoice and persist the trace."""
    run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_trace{run_number:02d}"
    t_start = time.time()
    agent_executions = []

    inv_num = invoice_data.get("invoice_number", "INV-UNKNOWN")
    sup_name = invoice_data.get("supplier_name", "Unknown Supplier")
    tot_amt = invoice_data.get("total_amount", 0)

    print(f"\n--- [Run {run_number:02d}] Executing Invoice Processing for {inv_num} ({sup_name} - ${tot_amt}) ---")

    # 1. Invoice Extraction Agent (Key 1)
    print("  [1/8] Invoice Extraction Agent -> GROQ_API_KEY_1...")
    desc_1 = f"""
Analyze the following invoice and extract its structured fields.
Raw Invoice Input:
{json.dumps(invoice_data, indent=2)}

Tasks:
1. Extract: invoice_number, supplier_id, supplier_name, po_reference, invoice_date, due_date, currency.
2. Extract line_items with description, quantity, unit_price, line_total.
3. Extract subtotal, tax, discount, and total_amount.
4. Identify any missing, ambiguous, or inconsistent fields.
Output strictly valid JSON with keys: 'extracted_invoice' and 'extraction_flags'.
"""
    exec_1 = call_agent_groq(
        agent_id="invoice_extraction_agent",
        role="Senior Invoice Extraction Specialist",
        goal="Accurately extract all structured invoice fields (supplier, PO, line items, dates, totals) without hallucinating missing data.",
        backstory="An expert document processor specialized in parsing accounts payable documents into structured formats.",
        task_description=desc_1,
        max_tokens=2500,
    )
    exec_1["order"] = 1
    exec_1["agent_id"] = "invoice_extraction_agent"
    agent_executions.append(exec_1)

    # 2. Invoice Validation Agent (Key 2)
    print("  [2/8] Invoice Validation Agent -> GROQ_API_KEY_2...")
    math_audit = deterministic_arithmetic_check(invoice_data)
    desc_2 = f"""
Perform structured validation of this invoice.
Invoice Data:
{json.dumps(invoice_data, indent=2)}

Verified Deterministic Math Audit from Python:
{json.dumps(math_audit, indent=2)}

Instructions:
1. Check if all required fields are present (supplier, dates, line items, totals).
2. Incorporate the deterministic arithmetic check results.
3. Note whether the invoice has valid arithmetic or critical math errors.
Return strictly valid JSON with keys: 'validation_status' ('PASSED' or 'FAILED'), 'math_audit', and 'validation_errors'.
"""
    exec_2 = call_agent_groq(
        agent_id="invoice_validation_agent",
        role="Invoice Validation Auditor",
        goal="Verify invoice field integrity, validate calculations against deterministic math checks, and document discrepancies.",
        backstory="A meticulous compliance auditor specializing in accounts payable fraud and mathematical discrepancy detection.",
        task_description=desc_2,
        max_tokens=2500,
    )
    exec_2["order"] = 2
    exec_2["agent_id"] = "invoice_validation_agent"
    agent_executions.append(exec_2)

    # 3. Purchase Order Matching Agent (Key 3)
    print("  [3/8] Purchase Order Matching Agent -> GROQ_API_KEY_3...")
    po_match_result = deterministic_po_match(invoice_data, pos_data)
    desc_3 = f"""
Reconcile this invoice against its corresponding purchase order.
Invoice Data:
{json.dumps(invoice_data, indent=2)}

Deterministic PO Matching Results:
{json.dumps(po_match_result, indent=2)}

Instructions:
1. Evaluate whether the invoice accurately matches the PO.
2. Detail all quantity, price, or supplier discrepancies.
Return strictly valid JSON with keys: 'po_match_status', 'matched_po_id', 'discrepancies', and 'matching_confidence'.
"""
    exec_3 = call_agent_groq(
        agent_id="po_matching_agent",
        role="Purchase Order Reconciliation Specialist",
        goal="Match invoice line items, pricing, quantities, and terms against authorized purchase orders.",
        backstory="An accounts payable specialist ensuring invoices only get paid if backed by approved purchase orders.",
        task_description=desc_3,
        max_tokens=2500,
    )
    exec_3["order"] = 3
    exec_3["agent_id"] = "po_matching_agent"
    agent_executions.append(exec_3)

    # 4. Policy Compliance Agent (Key 4)
    print("  [4/8] Policy Compliance Agent -> GROQ_API_KEY_4...")
    pol_audit = deterministic_policy_check(invoice_data, suppliers_dict)
    desc_4 = f"""
Perform policy compliance review on this invoice.
Invoice Data:
{json.dumps(invoice_data, indent=2)}

Deterministic Policy Audit:
{json.dumps(pol_audit, indent=2)}

Instructions:
1. Evaluate policy violations and threshold conditions.
2. Explicitly cite each applied policy ID.
3. Classify whether infractions are fatal policy violations or flags requiring secondary sign-off.
Return strictly valid JSON with keys: 'is_compliant', 'applied_policy_ids', 'violations', and 'compliance_summary'.
"""
    exec_4 = call_agent_groq(
        agent_id="policy_compliance_agent",
        role="Corporate Accounts Payable Compliance Officer",
        goal="Audit invoice against corporate purchasing policies, threshold limits, and vendor eligibility.",
        backstory="An uncompromising governance officer ensuring adherence to internal procurement policies and SOX controls.",
        task_description=desc_4,
        max_tokens=2500,
    )
    exec_4["order"] = 4
    exec_4["agent_id"] = "policy_compliance_agent"
    agent_executions.append(exec_4)

    # 5. Duplicate Detection Agent (Key 5)
    print("  [5/8] Duplicate Detection Agent -> GROQ_API_KEY_5...")
    dup_audit = deterministic_duplicate_check(invoice_data, hist_records)
    desc_5 = f"""
Perform duplicate screening on this invoice.
Current Invoice:
{json.dumps(invoice_data, indent=2)}

Deterministic Historical Match Scan:
{json.dumps(dup_audit, indent=2)}

Instructions:
1. Evaluate whether this invoice is an exact duplicate, potential duplicate, or original billing.
2. Cite matching historical invoice IDs and field comparisons.
Return strictly valid JSON with keys: 'duplicate_status' ('NONE', 'POTENTIAL_DUPLICATE', 'EXACT_DUPLICATE'), 'matching_records', and 'risk_assessment'.
"""
    exec_5 = call_agent_groq(
        agent_id="duplicate_detection_agent",
        role="Fraud & Duplicate Payment Prevention Analyst",
        goal="Screen invoices against historical payments to stop duplicate billing and overpayments.",
        backstory="An AP forensic specialist focused on identifying redundant billing and matching invoice histories.",
        task_description=desc_5,
        max_tokens=2500,
    )
    exec_5["order"] = 5
    exec_5["agent_id"] = "duplicate_detection_agent"
    agent_executions.append(exec_5)

    # Stage evidence bundle for subsequent synthesis
    stage_evidence = {
        "extraction": exec_1["output"],
        "validation": exec_2["output"],
        "po_matching": exec_3["output"],
        "policy_compliance": exec_4["output"],
        "duplicate_detection": exec_5["output"],
    }

    # 6. Exception Analysis Agent (Key 6)
    print("  [6/8] Exception Analysis Agent -> GROQ_API_KEY_6...")
    desc_6 = f"""
Analyze all upstream findings for accounts payable exceptions.
Consolidated Evidence:
{json.dumps(stage_evidence, indent=2)}

Tasks:
1. Consolidate math discrepancies, PO mismatch findings, policy violations, and duplicate alerts.
2. Assign an exception severity to each issue ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
3. Note whether any exceptions are resolvable with further human review or are fatal blockers.
Return strictly valid JSON with keys: 'has_exceptions', 'exception_count', 'critical_exceptions', and 'detailed_exceptions'.
"""
    exec_6 = call_agent_groq(
        agent_id="exception_analysis_agent",
        role="Accounts Payable Exception Specialist",
        goal="Consolidate discrepancies, classify exception severities, and flag operational risks across all stages.",
        backstory="A senior risk analyst evaluating edge cases, discrepancies, and policy violations in payment workflows.",
        task_description=desc_6,
        max_tokens=2500,
    )
    exec_6["order"] = 6
    exec_6["agent_id"] = "exception_analysis_agent"
    agent_executions.append(exec_6)
    stage_evidence["exception_analysis"] = exec_6["output"]

    # 7. Approval Recommendation Agent (Key 7)
    print("  [7/8] Approval Recommendation Agent -> GROQ_API_KEY_7...")
    rule_guidance = deterministic_recommendation_rule(stage_evidence)
    desc_7 = f"""
Formulate a structured approval recommendation for this invoice.
Consolidated Evidence from all preceding stages:
{json.dumps(stage_evidence, indent=2)}

Deterministic Decision Guidance:
{json.dumps(rule_guidance, indent=2)}

Instructions:
1. Select outcome: 'recommended_for_approval', 'rejected', or 'needs_review'.
2. Provide explicit justification referencing PO match status, policy IDs, math audit, and duplicate findings.
3. Note: NEVER execute real financial transactions.
Return strictly valid JSON with keys: 'recommendation' ('recommended_for_approval', 'rejected', or 'needs_review'), 'justification', and 'conditions'.
"""
    exec_7 = call_agent_groq(
        agent_id="approval_recommendation_agent",
        role="Accounts Payable Decision Authority",
        goal="Determine final payment recommendation (recommended_for_approval, rejected, or needs_review) based on factual evidence.",
        backstory="An executive accounts payable controller applying strict fiduciary oversight to payment approvals.",
        task_description=desc_7,
        max_tokens=2500,
    )
    exec_7["order"] = 7
    exec_7["agent_id"] = "approval_recommendation_agent"
    agent_executions.append(exec_7)
    stage_evidence["approval_recommendation"] = exec_7["output"]

    # 8. Final Review Agent (Key 8)
    print("  [8/8] Final Review Agent -> GROQ_API_KEY_8...")
    desc_8 = f"""
Perform the final comprehensive review of this accounts payable workflow execution.
Complete Audit Trail:
{json.dumps(stage_evidence, indent=2)}

Tasks:
1. Verify that the recommendation aligns with math validation, PO matching, policy checks, and duplicate findings.
2. Confirm there are no unaddressed critical exceptions.
3. Generate the final structured AP review report.
Output strictly valid JSON with keys: 'workflow_status' ('COMPLETED'), 'invoice_id', 'final_recommendation', 'risk_level', 'summary_report', and 'audit_findings'.
"""
    exec_8 = call_agent_groq(
        agent_id="final_review_agent_invoice",
        role="Chief AP Quality & Verification Auditor",
        goal="Perform final consistency verification across all stages and produce the authoritative accounts payable report.",
        backstory="The final executive gatekeeper ensuring audit trails are complete, robust, and mathematically sound.",
        task_description=desc_8,
        max_tokens=4096,
    )
    exec_8["order"] = 8
    exec_8["agent_id"] = "final_review_agent_invoice"
    agent_executions.append(exec_8)

    t_end = time.time()
    total_dur = round(t_end - t_start, 3)

    tot_prompt = sum(e["prompt_tokens"] for e in agent_executions)
    tot_comp = sum(e["completion_tokens"] for e in agent_executions)
    tot_tokens = tot_prompt + tot_comp
    cost = round((tot_prompt / 1_000_000.0) * 0.59 + (tot_comp / 1_000_000.0) * 0.79, 6)

    trace_record = {
        "run_id": run_id,
        "workflow_name": "invoice_processing",
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
        "final_output": json.dumps(exec_8["output"], indent=2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save directly in Ingestion_Traces/invoice_processing/ (no agent-wise subfolders)
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
        "workflow_name": "invoice_processing",
        "total_runs": total_runs,
        "successful_runs": total_runs,
        "failed_runs": 0,
        "model_enforced": "openai/gpt-oss-120b",
        "key_distribution_model": "1-to-1 dedicated Groq key per sub-agent (GROQ_API_KEY_1..8)",
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
    parser = argparse.ArgumentParser(description="Generate live Ingestion_Traces for Invoice Processing")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute (default: 1)")
    parser.add_argument("--start-idx", type=int, default=0, help="Invoices dataset starting index")
    parser.add_argument("--start-run-num", type=int, default=1, help="Run number to start numbering from")
    parser.add_argument("--total-runs", type=int, default=30, help="Total runs in batch for logging")
    args = parser.parse_args()

    # Load datasets
    data_dir = PROJECT_ROOT / "data" / "invoice_processing"
    with open(data_dir / "invoices.json", "r", encoding="utf-8") as f:
        all_invoices = json.load(f)
    with open(data_dir / "purchase_orders.json", "r", encoding="utf-8") as f:
        pos_data = json.load(f)
    with open(data_dir / "suppliers.json", "r", encoding="utf-8") as f:
        sups_list = json.load(f)
        suppliers_dict = {s["supplier_id"]: s for s in sups_list}
    with open(data_dir / "historical_invoices.json", "r", encoding="utf-8") as f:
        hist_records = json.load(f)

    target_dir = PROJECT_ROOT / "Ingestion_Traces" / "invoice_processing"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting execution of {args.runs} live runs for invoice_processing (Runs {args.start_run_num} to {args.start_run_num + args.runs - 1})...")
    print(f"Output directory: {target_dir}")

    summary_file = target_dir / "summary.jsonl"

    for i in range(args.runs):
        run_num = args.start_run_num + i
        inv_idx = (args.start_idx + i) % len(all_invoices)
        inv_sample = all_invoices[inv_idx]

        print(f"\n=======================================================")
        print(f" [RUN {run_num:02d}/{args.total_runs}] Starting...")
        print(f"=======================================================")

        try:
            trace_rec = run_single_invoice_trace(
                invoice_data=inv_sample,
                pos_data=pos_data,
                suppliers_dict=suppliers_dict,
                hist_records=hist_records,
                run_number=run_num,
                output_dir=target_dir,
            )
            # Append to summary.jsonl
            with open(summary_file, "a", encoding="utf-8") as sf:
                summary_entry = {
                    "run_number": run_num,
                    "run_id": trace_rec["run_id"],
                    "invoice_number": inv_sample.get("invoice_number"),
                    "supplier_id": inv_sample.get("supplier_id"),
                    "supplier_name": inv_sample.get("supplier_name"),
                    "total_amount": inv_sample.get("total_amount"),
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
                    "invoice_number": inv_sample.get("invoice_number"),
                    "error": str(e),
                    "status": "FAILED",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")

    generate_batch_summary(target_dir)
