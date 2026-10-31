"""
Workflow 1: Invoice Processing and Accounts Payable Orchestrator.
Sequential 8-Agent Baseline System.
"""

import time
import json
from typing import Dict, Any, Optional
from crewai import Crew, Process

from common.tracing import WorkflowRunTracker, init_phoenix, extract_token_usage
from common.api_key_manager import get_key_for_agent

from Agents.invoice_processing.invoice_extraction_agent import (
    create_agent as create_extraction_agent,
    create_task as create_extraction_task,
    AGENT_ID as AGENT_EXTRACTION_ID
)
from Agents.invoice_processing.invoice_validation_agent import (
    create_agent as create_validation_agent,
    create_task as create_validation_task,
    deterministic_arithmetic_check,
    AGENT_ID as AGENT_VALIDATION_ID
)
from Agents.invoice_processing.po_matching_agent import (
    create_agent as create_po_agent,
    create_task as create_po_task,
    deterministic_po_match,
    AGENT_ID as AGENT_PO_ID
)
from Agents.invoice_processing.policy_compliance_agent import (
    create_agent as create_compliance_agent,
    create_task as create_compliance_task,
    deterministic_policy_check,
    AGENT_ID as AGENT_COMPLIANCE_ID
)
from Agents.invoice_processing.duplicate_detection_agent import (
    create_agent as create_duplicate_agent,
    create_task as create_duplicate_task,
    deterministic_duplicate_check,
    AGENT_ID as AGENT_DUPLICATE_ID
)
from Agents.invoice_processing.exception_analysis_agent import (
    create_agent as create_exception_agent,
    create_task as create_exception_task,
    AGENT_ID as AGENT_EXCEPTION_ID
)
from Agents.invoice_processing.approval_recommendation_agent import (
    create_agent as create_approval_agent,
    create_task as create_approval_task,
    deterministic_recommendation_rule,
    AGENT_ID as AGENT_APPROVAL_ID
)
from Agents.invoice_processing.final_review_agent import (
    create_agent as create_review_agent,
    create_task as create_review_task,
    AGENT_ID as AGENT_REVIEW_ID
)

WORKFLOW_NAME = "invoice_processing"

def run_invoice_processing_workflow(
    invoice_data: Dict[str, Any],
    run_id: Optional[str] = None,
    mock_mode: bool = False
) -> Dict[str, Any]:
    """
    Executes the 8-agent sequential Invoice Processing baseline workflow.
    """
    init_phoenix()
    tracker = WorkflowRunTracker(WORKFLOW_NAME, run_id).start()

    print(f"\n========================================================")
    print(f"Starting Workflow: {WORKFLOW_NAME} (Run ID: {tracker.run_id})")
    print(f"Target Invoice: {invoice_data.get('invoice_number', 'UNKNOWN')} | Supplier: {invoice_data.get('supplier_id', 'UNKNOWN')}")
    print(f"========================================================")

    stage_evidence = {}

    try:
        # Agent 1: Extraction
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_EXTRACTION_ID)
        agent1 = create_extraction_agent()
        task1 = create_extraction_task(agent1, invoice_data)
        crew1 = Crew(agents=[agent1], tasks=[task1], verbose=False)
        res1 = crew1.kickoff()
        out1 = res1.raw if hasattr(res1, "raw") else str(res1)
        tot1, p1, c1 = extract_token_usage(res1, crew1)
        dur1 = time.time() - t0
        tracker.log_agent_execution(AGENT_EXTRACTION_ID, key_id, dur1, status="SUCCESS", tokens=tot1, prompt_tokens=p1, completion_tokens=c1, output=out1)
        stage_evidence["extraction"] = out1
        print(f" -> [Agent 1/8] {AGENT_EXTRACTION_ID} completed in {dur1:.2f}s ({tot1} tokens)")

        # Agent 2: Validation
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_VALIDATION_ID)
        math_check = deterministic_arithmetic_check(invoice_data)
        agent2 = create_validation_agent()
        task2 = create_validation_task(agent2, invoice_data)
        crew2 = Crew(agents=[agent2], tasks=[task2], verbose=False)
        res2 = crew2.kickoff()
        out2 = res2.raw if hasattr(res2, "raw") else str(res2)
        tot2, p2, c2 = extract_token_usage(res2, crew2)
        dur2 = time.time() - t0
        tracker.log_agent_execution(AGENT_VALIDATION_ID, key_id, dur2, status="SUCCESS", tokens=tot2, prompt_tokens=p2, completion_tokens=c2, output=out2)
        stage_evidence["validation"] = {"llm_output": out2, "deterministic_audit": math_check}
        print(f" -> [Agent 2/8] {AGENT_VALIDATION_ID} completed in {dur2:.2f}s ({tot2} tokens)")

        # Agent 3: PO Matching
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_PO_ID)
        po_match = deterministic_po_match(invoice_data)
        agent3 = create_po_agent()
        task3 = create_po_task(agent3, invoice_data)
        crew3 = Crew(agents=[agent3], tasks=[task3], verbose=False)
        res3 = crew3.kickoff()
        out3 = res3.raw if hasattr(res3, "raw") else str(res3)
        tot3, p3, c3 = extract_token_usage(res3, crew3)
        dur3 = time.time() - t0
        tracker.log_agent_execution(AGENT_PO_ID, key_id, dur3, status="SUCCESS", tokens=tot3, prompt_tokens=p3, completion_tokens=c3, output=out3)
        stage_evidence["po_matching"] = {"llm_output": out3, "deterministic_match": po_match}
        print(f" -> [Agent 3/8] {AGENT_PO_ID} completed in {dur3:.2f}s ({tot3} tokens)")

        # Agent 4: Policy Compliance
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_COMPLIANCE_ID)
        pol_check = deterministic_policy_check(invoice_data)
        agent4 = create_compliance_agent()
        task4 = create_compliance_task(agent4, invoice_data)
        crew4 = Crew(agents=[agent4], tasks=[task4], verbose=False)
        res4 = crew4.kickoff()
        out4 = res4.raw if hasattr(res4, "raw") else str(res4)
        tot4, p4, c4 = extract_token_usage(res4, crew4)
        dur4 = time.time() - t0
        tracker.log_agent_execution(AGENT_COMPLIANCE_ID, key_id, dur4, status="SUCCESS", tokens=tot4, prompt_tokens=p4, completion_tokens=c4, output=out4)
        stage_evidence["policy_compliance"] = {"llm_output": out4, "deterministic_compliance": pol_check}
        print(f" -> [Agent 4/8] {AGENT_COMPLIANCE_ID} completed in {dur4:.2f}s ({tot4} tokens)")

        # Agent 5: Duplicate Detection
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_DUPLICATE_ID)
        dup_check = deterministic_duplicate_check(invoice_data)
        agent5 = create_duplicate_agent()
        task5 = create_duplicate_task(agent5, invoice_data)
        crew5 = Crew(agents=[agent5], tasks=[task5], verbose=False)
        res5 = crew5.kickoff()
        out5 = res5.raw if hasattr(res5, "raw") else str(res5)
        tot5, p5, c5 = extract_token_usage(res5, crew5)
        dur5 = time.time() - t0
        tracker.log_agent_execution(AGENT_DUPLICATE_ID, key_id, dur5, status="SUCCESS", tokens=tot5, prompt_tokens=p5, completion_tokens=c5, output=out5)
        stage_evidence["duplicate_detection"] = {"llm_output": out5, "deterministic_duplicates": dup_check}
        print(f" -> [Agent 5/8] {AGENT_DUPLICATE_ID} completed in {dur5:.2f}s ({tot5} tokens)")

        # Agent 6: Exception Analysis
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_EXCEPTION_ID)
        agent6 = create_exception_agent()
        task6 = create_exception_task(agent6, stage_evidence)
        crew6 = Crew(agents=[agent6], tasks=[task6], verbose=False)
        res6 = crew6.kickoff()
        out6 = res6.raw if hasattr(res6, "raw") else str(res6)
        tot6, p5, c6 = extract_token_usage(res6, crew6)
        dur6 = time.time() - t0
        tracker.log_agent_execution(AGENT_EXCEPTION_ID, key_id, dur6, status="SUCCESS", tokens=tot6, prompt_tokens=p5, completion_tokens=c6, output=out6)
        stage_evidence["exception_analysis"] = out6
        print(f" -> [Agent 6/8] {AGENT_EXCEPTION_ID} completed in {dur6:.2f}s ({tot6} tokens)")

        # Agent 7: Approval Recommendation
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_APPROVAL_ID)
        agent7 = create_approval_agent()
        task7 = create_approval_task(agent7, stage_evidence)
        crew7 = Crew(agents=[agent7], tasks=[task7], verbose=False)
        res7 = crew7.kickoff()
        out7 = res7.raw if hasattr(res7, "raw") else str(res7)
        tot7, p7, c7 = extract_token_usage(res7, crew7)
        dur7 = time.time() - t0
        tracker.log_agent_execution(AGENT_APPROVAL_ID, key_id, dur7, status="SUCCESS", tokens=tot7, prompt_tokens=p7, completion_tokens=c7, output=out7)
        stage_evidence["approval_recommendation"] = out7
        print(f" -> [Agent 7/8] {AGENT_APPROVAL_ID} completed in {dur7:.2f}s ({tot7} tokens)")

        # Agent 8: Final Review
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_REVIEW_ID)
        agent8 = create_review_agent()
        task8 = create_review_task(agent8, stage_evidence)
        crew8 = Crew(agents=[agent8], tasks=[task8], verbose=False)
        res8 = crew8.kickoff()
        final_output = res8.raw if hasattr(res8, "raw") else str(res8)
        tot8, p8, c8 = extract_token_usage(res8, crew8)
        dur8 = time.time() - t0
        tracker.log_agent_execution(AGENT_REVIEW_ID, key_id, dur8, status="SUCCESS", tokens=tot8, prompt_tokens=p8, completion_tokens=c8, output=final_output)
        print(f" -> [Agent 8/8] {AGENT_REVIEW_ID} completed in {dur8:.2f}s ({tot8} tokens)")

        tracker.complete(final_output)
        print(f"Workflow '{WORKFLOW_NAME}' finished successfully in {tracker.duration_seconds}s (Total Tokens: {tracker.total_tokens}).")
        return {
            "run_id": tracker.run_id,
            "status": "SUCCESS",
            "duration_seconds": tracker.duration_seconds,
            "total_tokens": tracker.total_tokens,
            "prompt_tokens": tracker.prompt_tokens,
            "completion_tokens": tracker.completion_tokens,
            "estimated_cost_usd": tracker.estimated_cost_usd,
            "agent_executions": tracker.agent_executions,
            "final_output": final_output
        }

    except Exception as exc:
        tracker.fail(exc)
        print(f"Workflow '{WORKFLOW_NAME}' FAILED: {exc}")
        raise
