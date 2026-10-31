"""Employee Expense Reimbursement Baseline Multi-Agent Workflow.

Workflow Two: Exactly 8 agents executing sequentially:
1. Receipt Agent
2. Normalization Agent
3. Category Agent
4. Amount Agent
5. Policy Agent
6. Duplicate Agent
7. Exception Agent
8. Review Agent
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from crewai import Crew, Process, Task

from Agents.expense_reimbursement.amount_agent import calculate_deterministic_arithmetic, create_amount_agent
from Agents.expense_reimbursement.category_agent import create_category_agent
from Agents.expense_reimbursement.duplicate_agent import create_duplicate_agent, query_candidate_duplicates
from Agents.expense_reimbursement.exception_agent import create_exception_agent
from Agents.expense_reimbursement.normalize_agent import create_normalize_agent
from Agents.expense_reimbursement.policy_agent import create_policy_agent, query_policies_for_category
from Agents.expense_reimbursement.receipt_agent import create_receipt_agent
from Agents.expense_reimbursement.review_agent import create_expense_review_agent
from common.agent_runner import ACTIVE_TRACKER, WorkflowExecutionResult, generate_run_id, save_run_result
from common.tracing import agent_span, workflow_span


def run_expense_workflow(
    claim_input: Dict[str, Any],
    run_id: Optional[str] = None
) -> WorkflowExecutionResult:
    """Execute the 8-agent Expense Reimbursement workflow."""
    ACTIVE_TRACKER.reset()
    run_id = run_id or generate_run_id("expense")
    start_total = time.perf_counter()
    agent_durations: Dict[str, float] = {}
    order = [
        "receipt_agent",
        "normalize_agent",
        "category_agent",
        "amount_agent",
        "policy_agent",
        "duplicate_agent",
        "exception_agent",
        "expense_review_agent",
    ]

    with workflow_span("expense_reimbursement", run_id, claim_input):
        # 1. Receipt Agent
        t0 = time.perf_counter()
        raw_receipt = claim_input.get("receipt_text", "")
        with agent_span("receipt_agent", run_id, {"has_receipt": bool(raw_receipt)}):
            receipt_agent = create_receipt_agent()
            receipt_task = Task(
                description=(
                    f"Read and parse this synthetic receipt text:\n"
                    f"{raw_receipt if raw_receipt else '[NO RECEIPT TEXT PROVIDED IN CLAIM]'}\n"
                    f"Extract merchant, transaction_date, currency, subtotal, tax, total, and line items. "
                    f"Explicitly note if receipt is missing or illegible."
                ),
                expected_output="JSON containing extracted merchant, date, currency, subtotal, tax, total, and line items.",
                agent=receipt_agent,
            )
            crew1 = Crew(agents=[receipt_agent], tasks=[receipt_task], process=Process.sequential, verbose=False)
            receipt_output = crew1.kickoff()
        agent_durations["receipt_agent"] = time.perf_counter() - t0
        receipt_text = str(receipt_output.raw if hasattr(receipt_output, "raw") else receipt_output)

        # 2. Normalize Agent
        t0 = time.perf_counter()
        with agent_span("normalize_agent", run_id, {"raw_claim_date": claim_input.get("submission_date")}):
            normalize_agent = create_normalize_agent()
            normalize_task = Task(
                description=(
                    f"Standardize and validate fields between claim and receipt:\n"
                    f"Claim Payload: {json.dumps(claim_input, indent=2)}\n"
                    f"Receipt Output: {receipt_text}\n"
                    f"Format dates to YYYY-MM-DD, currency to ISO code (e.g., USD), and numeric amounts to 2 decimals."
                ),
                expected_output="JSON containing normalized employee_id, merchant, date, currency, subtotal, tax, and total.",
                agent=normalize_agent,
            )
            crew2 = Crew(agents=[normalize_agent], tasks=[normalize_task], process=Process.sequential, verbose=False)
            normalize_output = crew2.kickoff()
        agent_durations["normalize_agent"] = time.perf_counter() - t0
        normalize_text = str(normalize_output.raw if hasattr(normalize_output, "raw") else normalize_output)

        # 3. Category Agent
        t0 = time.perf_counter()
        with agent_span("category_agent", run_id, {}):
            category_agent = create_category_agent()
            category_task = Task(
                description=(
                    f"Classify this expense into an official category:\n"
                    f"Allowed: Travel, Meals & Entertainment, Accommodation, Transport, Office Supplies, Software & Cloud.\n"
                    f"Normalized Context: {normalize_text}\n"
                    f"State the assigned category and provide evidence for this classification."
                ),
                expected_output="JSON containing category, classification_confidence, and reasoning.",
                agent=category_agent,
            )
            crew3 = Crew(agents=[category_agent], tasks=[category_task], process=Process.sequential, verbose=False)
            category_output = crew3.kickoff()
        agent_durations["category_agent"] = time.perf_counter() - t0
        category_text = str(category_output.raw if hasattr(category_output, "raw") else category_output)

        # Extract assigned category for policy lookup
        assigned_category = claim_input.get("category", "Meals & Entertainment")
        for cat in ["Travel", "Meals & Entertainment", "Accommodation", "Transport", "Office Supplies", "Software & Cloud"]:
            if cat.lower() in category_text.lower():
                assigned_category = cat
                break

        # 4. Amount Agent (Deterministic arithmetic + LLM verification)
        t0 = time.perf_counter()
        sub = float(claim_input.get("subtotal", 0.0))
        tx = float(claim_input.get("tax", 0.0))
        tot = float(claim_input.get("total", 0.0))
        arithmetic_check = calculate_deterministic_arithmetic(sub, tx, tot)

        with agent_span("amount_agent", run_id, arithmetic_check):
            amount_agent = create_amount_agent()
            amount_task = Task(
                description=(
                    f"Deterministic Arithmetic Verification:\n{json.dumps(arithmetic_check, indent=2)}\n"
                    f"Verify if subtotal (${sub:.2f}) + tax (${tx:.2f}) equals total (${tot:.2f}). "
                    f"Confirm whether any negative amounts or rounding anomalies exist."
                ),
                expected_output="JSON with is_arithmetic_valid, discrepancy_usd, and verification_notes.",
                agent=amount_agent,
            )
            crew4 = Crew(agents=[amount_agent], tasks=[amount_task], process=Process.sequential, verbose=False)
            amount_output = crew4.kickoff()
        agent_durations["amount_agent"] = time.perf_counter() - t0
        amount_text = str(amount_output.raw if hasattr(amount_output, "raw") else amount_output)

        # 5. Policy Agent (Deterministic lookup + LLM compliance check)
        t0 = time.perf_counter()
        policies = query_policies_for_category(assigned_category)
        with agent_span("policy_agent", run_id, {"policies_evaluated_count": len(policies)}):
            policy_agent = create_policy_agent()
            policy_task = Task(
                description=(
                    f"Evaluate this claim against corporate policy limits:\n"
                    f"Applicable Policies:\n{json.dumps(policies, indent=2)}\n"
                    f"Claim Details: Total=${tot:.2f}, Category={assigned_category}, HasReceipt={bool(raw_receipt)}\n"
                    f"Return which policy IDs apply, whether any limit is breached, and compliance status."
                ),
                expected_output="JSON with policy_compliance_status, cited_policy_ids, and violation_reasons.",
                agent=policy_agent,
            )
            crew5 = Crew(agents=[policy_agent], tasks=[policy_task], process=Process.sequential, verbose=False)
            policy_output = crew5.kickoff()
        agent_durations["policy_agent"] = time.perf_counter() - t0
        policy_text = str(policy_output.raw if hasattr(policy_output, "raw") else policy_output)

        # 6. Duplicate Agent (Deterministic candidate lookup + LLM comparison)
        t0 = time.perf_counter()
        emp_id = claim_input.get("employee_id", "")
        merchant_name = claim_input.get("merchant", "")
        duplicate_candidates = query_candidate_duplicates(emp_id, merchant_name, tot)

        with agent_span("duplicate_agent", run_id, {"candidate_matches_count": len(duplicate_candidates)}):
            duplicate_agent = create_duplicate_agent()
            duplicate_task = Task(
                description=(
                    f"Check for duplicate submissions against historical database:\n"
                    f"Candidate Submissions:\n{json.dumps(duplicate_candidates, indent=2)}\n"
                    f"Current Claim: Employee={emp_id}, Merchant={merchant_name}, Total=${tot:.2f}, Date={claim_input.get('submission_date')}\n"
                    f"Determine if this is a Confirmed Duplicate, Possible Coincidence, or No Duplicate Found."
                ),
                expected_output="JSON with duplicate_status, matching_claim_ids, and duplicate_risk_score.",
                agent=duplicate_agent,
            )
            crew6 = Crew(agents=[duplicate_agent], tasks=[duplicate_task], process=Process.sequential, verbose=False)
            duplicate_output = crew6.kickoff()
        agent_durations["duplicate_agent"] = time.perf_counter() - t0
        duplicate_text = str(duplicate_output.raw if hasattr(duplicate_output, "raw") else duplicate_output)

        # 7. Exception Agent (Independent Red-Team Audit)
        t0 = time.perf_counter()
        with agent_span("exception_agent", run_id, {}):
            exception_agent = create_exception_agent()
            exception_task = Task(
                description=(
                    f"Conduct an independent exception audit across all findings:\n"
                    f"Receipt Info: {receipt_text}\n"
                    f"Arithmetic: {amount_text}\n"
                    f"Policy Check: {policy_text}\n"
                    f"Duplicate Check: {duplicate_text}\n"
                    f"Identify all anomalies, unitemized receipts, policy breaches, or red flags."
                ),
                expected_output="JSON with list of exceptions_found, severity_levels, and audit_warnings.",
                agent=exception_agent,
            )
            crew7 = Crew(agents=[exception_agent], tasks=[exception_task], process=Process.sequential, verbose=False)
            exception_output = crew7.kickoff()
        agent_durations["exception_agent"] = time.perf_counter() - t0
        exception_text = str(exception_output.raw if hasattr(exception_output, "raw") else exception_output)

        # 8. Review Agent (Final Decision Synthesis)
        t0 = time.perf_counter()
        with agent_span("expense_review_agent", run_id, {}):
            review_agent = create_expense_review_agent()
            review_task = Task(
                description=(
                    f"Synthesize all preceding 7 stages into a final reimbursement decision:\n"
                    f"1. Receipt: {receipt_text}\n"
                    f"2. Normalization: {normalize_text}\n"
                    f"3. Category: {category_text}\n"
                    f"4. Amount: {amount_text}\n"
                    f"5. Policy: {policy_text}\n"
                    f"6. Duplicate: {duplicate_text}\n"
                    f"7. Exceptions: {exception_text}\n"
                    f"Output final decision: ELIGIBLE, INELIGIBLE, or NEEDS_REVIEW. Include clear rationale."
                ),
                expected_output="Final Expense Adjudication JSON report.",
                agent=review_agent,
            )
            crew8 = Crew(agents=[review_agent], tasks=[review_task], process=Process.sequential, verbose=False)
            review_output = crew8.kickoff()
        agent_durations["expense_review_agent"] = time.perf_counter() - t0
        review_text = str(review_output.raw if hasattr(review_output, "raw") else review_output)

    total_duration = time.perf_counter() - start_total

    final_payload = {
        "workflow": "expense_reimbursement",
        "run_id": run_id,
        "input": claim_input,
        "stages": {
            "receipt": receipt_text,
            "normalize": normalize_text,
            "category": category_text,
            "amount": amount_text,
            "policy": policy_text,
            "duplicate": duplicate_text,
            "exceptions": exception_text,
            "final_review": review_text,
        },
        "synthetic": True,
    }

    tracker_data = ACTIVE_TRACKER.snapshot()
    result = WorkflowExecutionResult(
        workflow_name="expense_reimbursement",
        run_id=run_id,
        status="SUCCESS",
        execution_order=order,
        agent_durations=agent_durations,
        total_duration_seconds=total_duration,
        agent_count=len(order),
        llm_call_count=tracker_data["llm_call_count"],
        token_usage=tracker_data["token_usage"],
        estimated_cost_usd=tracker_data["estimated_cost_usd"],
        cost_estimation_label=tracker_data["cost_estimation_label"],
        output=final_payload,
    )
    save_run_result(result)
    return result
