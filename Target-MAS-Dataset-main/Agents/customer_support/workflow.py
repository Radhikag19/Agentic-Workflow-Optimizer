"""
Workflow 2: Customer Support Ticket Resolution Orchestrator.
Sequential 6-Agent Baseline System.
"""

import time
import json
from typing import Dict, Any, Optional
from crewai import Crew

from common.tracing import WorkflowRunTracker, init_phoenix, extract_token_usage
from common.api_key_manager import get_key_for_agent

from Agents.customer_support.ticket_classification_agent import (
    create_agent as create_classification_agent,
    create_task as create_classification_task,
    deterministic_classification_hints,
    AGENT_ID as AGENT_CLASSIFICATION_ID
)
from Agents.customer_support.policy_retrieval_agent import (
    create_agent as create_retrieval_agent,
    create_task as create_retrieval_task,
    deterministic_policy_retrieval,
    AGENT_ID as AGENT_RETRIEVAL_ID
)
from Agents.customer_support.issue_investigation_agent import (
    create_agent as create_investigation_agent,
    create_task as create_investigation_task,
    deterministic_customer_lookup,
    AGENT_ID as AGENT_INVESTIGATION_ID
)
from Agents.customer_support.resolution_agent import (
    create_agent as create_resolution_agent,
    create_task as create_resolution_task,
    AGENT_ID as AGENT_RESOLUTION_ID
)
from Agents.customer_support.response_drafting_agent import (
    create_agent as create_drafting_agent,
    create_task as create_drafting_task,
    AGENT_ID as AGENT_DRAFTING_ID
)
from Agents.customer_support.resolution_verification_agent import (
    create_agent as create_verification_agent,
    create_task as create_verification_task,
    AGENT_ID as AGENT_VERIFICATION_ID
)

WORKFLOW_NAME = "customer_support"

def run_customer_support_workflow(
    ticket_data: Dict[str, Any],
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the 6-agent sequential Customer Support baseline workflow.
    """
    init_phoenix()
    tracker = WorkflowRunTracker(WORKFLOW_NAME, run_id).start()

    print(f"\n========================================================")
    print(f"Starting Workflow: {WORKFLOW_NAME} (Run ID: {tracker.run_id})")
    print(f"Target Ticket: {ticket_data.get('ticket_id', 'UNKNOWN')} | Category: {ticket_data.get('category', 'UNKNOWN')}")
    print(f"========================================================")

    stage_evidence = {}

    try:
        # Agent 1: Classification
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_CLASSIFICATION_ID)
        hints = deterministic_classification_hints(ticket_data)
        agent1 = create_classification_agent()
        task1 = create_classification_task(agent1, ticket_data)
        crew1 = Crew(agents=[agent1], tasks=[task1], verbose=False)
        res1 = crew1.kickoff()
        out1 = res1.raw if hasattr(res1, "raw") else str(res1)
        tot1, p1, c1 = extract_token_usage(res1, crew1)
        dur1 = time.time() - t0
        tracker.log_agent_execution(AGENT_CLASSIFICATION_ID, key_id, dur1, status="SUCCESS", tokens=tot1, prompt_tokens=p1, completion_tokens=c1, output=out1)
        stage_evidence["classification"] = {"llm_output": out1, "hints": hints}
        print(f" -> [Agent 1/6] {AGENT_CLASSIFICATION_ID} completed in {dur1:.2f}s ({tot1} tokens)")

        # Agent 2: Policy Retrieval
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_RETRIEVAL_ID)
        cat = ticket_data.get("category", hints.get("suggested_category", "general"))
        retrieval_data = deterministic_policy_retrieval(cat)
        agent2 = create_retrieval_agent()
        task2 = create_retrieval_task(agent2, ticket_data, {"category": cat})
        crew2 = Crew(agents=[agent2], tasks=[task2], verbose=False)
        res2 = crew2.kickoff()
        out2 = res2.raw if hasattr(res2, "raw") else str(res2)
        tot2, p2, c2 = extract_token_usage(res2, crew2)
        dur2 = time.time() - t0
        tracker.log_agent_execution(AGENT_RETRIEVAL_ID, key_id, dur2, status="SUCCESS", tokens=tot2, prompt_tokens=p2, completion_tokens=c2, output=out2)
        stage_evidence["policy_retrieval"] = {"llm_output": out2, "retrieved": retrieval_data}
        print(f" -> [Agent 2/6] {AGENT_RETRIEVAL_ID} completed in {dur2:.2f}s ({tot2} tokens)")

        # Agent 3: Issue Investigation
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_INVESTIGATION_ID)
        agent3 = create_investigation_agent()
        task3 = create_investigation_task(agent3, ticket_data)
        crew3 = Crew(agents=[agent3], tasks=[task3], verbose=False)
        res3 = crew3.kickoff()
        out3 = res3.raw if hasattr(res3, "raw") else str(res3)
        tot3, p3, c3 = extract_token_usage(res3, crew3)
        dur3 = time.time() - t0
        tracker.log_agent_execution(AGENT_INVESTIGATION_ID, key_id, dur3, status="SUCCESS", tokens=tot3, prompt_tokens=p3, completion_tokens=c3, output=out3)
        stage_evidence["investigation"] = out3
        print(f" -> [Agent 3/6] {AGENT_INVESTIGATION_ID} completed in {dur3:.2f}s ({tot3} tokens)")

        # Agent 4: Resolution Recommendation
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_RESOLUTION_ID)
        agent4 = create_resolution_agent()
        task4 = create_resolution_task(agent4, ticket_data, stage_evidence)
        crew4 = Crew(agents=[agent4], tasks=[task4], verbose=False)
        res4 = crew4.kickoff()
        out4 = res4.raw if hasattr(res4, "raw") else str(res4)
        tot4, p4, c4 = extract_token_usage(res4, crew4)
        dur4 = time.time() - t0
        tracker.log_agent_execution(AGENT_RESOLUTION_ID, key_id, dur4, status="SUCCESS", tokens=tot4, prompt_tokens=p4, completion_tokens=c4, output=out4)
        stage_evidence["resolution"] = out4
        print(f" -> [Agent 4/6] {AGENT_RESOLUTION_ID} completed in {dur4:.2f}s ({tot4} tokens)")

        # Agent 5: Response Drafting
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_DRAFTING_ID)
        agent5 = create_drafting_agent()
        task5 = create_drafting_task(agent5, ticket_data, {"resolution": out4})
        crew5 = Crew(agents=[agent5], tasks=[task5], verbose=False)
        res5 = crew5.kickoff()
        out5 = res5.raw if hasattr(res5, "raw") else str(res5)
        tot5, p5, c5 = extract_token_usage(res5, crew5)
        dur5 = time.time() - t0
        tracker.log_agent_execution(AGENT_DRAFTING_ID, key_id, dur5, status="SUCCESS", tokens=tot5, prompt_tokens=p5, completion_tokens=c5, output=out5)
        stage_evidence["drafting"] = out5
        print(f" -> [Agent 5/6] {AGENT_DRAFTING_ID} completed in {dur5:.2f}s ({tot5} tokens)")

        # Agent 6: Resolution Verification
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_VERIFICATION_ID)
        agent6 = create_verification_agent()
        task6 = create_verification_task(agent6, stage_evidence)
        crew6 = Crew(agents=[agent6], tasks=[task6], verbose=False)
        res6 = crew6.kickoff()
        final_output = res6.raw if hasattr(res6, "raw") else str(res6)
        tot6, p6, c6 = extract_token_usage(res6, crew6)
        dur6 = time.time() - t0
        tracker.log_agent_execution(AGENT_VERIFICATION_ID, key_id, dur6, status="SUCCESS", tokens=tot6, prompt_tokens=p6, completion_tokens=c6, output=final_output)
        print(f" -> [Agent 6/6] {AGENT_VERIFICATION_ID} completed in {dur6:.2f}s ({tot6} tokens)")

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
