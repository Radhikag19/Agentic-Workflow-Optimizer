"""
Workflow 3: Procurement and Supplier Selection Orchestrator.
Sequential 9-Agent Baseline System.
"""

import time
import json
from typing import Dict, Any, Optional
from crewai import Crew

from common.tracing import WorkflowRunTracker, init_phoenix, extract_token_usage
from common.api_key_manager import get_key_for_agent

from Agents.procurement_supplier_selection.quotation_analysis_agent import (
    create_agent as create_quotation_agent,
    create_task as create_quotation_task,
    AGENT_ID as AGENT_QUOTATION_ID
)
from Agents.procurement_supplier_selection.supplier_profile_agent import (
    create_agent as create_profile_agent,
    create_task as create_profile_task,
    deterministic_profile_lookup,
    AGENT_ID as AGENT_PROFILE_ID
)
from Agents.procurement_supplier_selection.delivery_analysis_agent import (
    create_agent as create_delivery_agent,
    create_task as create_delivery_task,
    deterministic_delivery_check,
    AGENT_ID as AGENT_DELIVERY_ID
)
from Agents.procurement_supplier_selection.supplier_risk_agent import (
    create_agent as create_risk_agent,
    create_task as create_risk_task,
    deterministic_risk_lookup,
    AGENT_ID as AGENT_RISK_ID
)
from Agents.procurement_supplier_selection.contract_analysis_agent import (
    create_agent as create_contract_agent,
    create_task as create_contract_task,
    deterministic_contract_lookup,
    AGENT_ID as AGENT_CONTRACT_ID
)
from Agents.procurement_supplier_selection.cost_comparison_agent import (
    create_agent as create_cost_agent,
    create_task as create_cost_task,
    deterministic_cost_ranking,
    AGENT_ID as AGENT_COST_ID
)
from Agents.procurement_supplier_selection.weighted_evaluation_agent import (
    create_agent as create_weighted_agent,
    create_task as create_weighted_task,
    deterministic_weighted_scoring,
    AGENT_ID as AGENT_WEIGHTED_ID
)
from Agents.procurement_supplier_selection.selection_recommendation_agent import (
    create_agent as create_recommendation_agent,
    create_task as create_recommendation_task,
    AGENT_ID as AGENT_RECOMMENDATION_ID
)
from Agents.procurement_supplier_selection.final_review_agent import (
    create_agent as create_review_agent,
    create_task as create_review_task,
    AGENT_ID as AGENT_REVIEW_ID
)

WORKFLOW_NAME = "procurement_supplier_selection"

def run_procurement_workflow(
    procurement_case: Dict[str, Any],
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the 9-agent sequential Procurement and Supplier Selection baseline workflow.
    """
    init_phoenix()
    tracker = WorkflowRunTracker(WORKFLOW_NAME, run_id).start()

    req = procurement_case.get("procurement_request", {})
    quotes = procurement_case.get("quotations", [])
    schedules = procurement_case.get("delivery_schedules", [])
    supplier_ids = [q.get("supplier_id") for q in quotes]
    req_date = req.get("target_delivery_date", "2026-12-31")

    print(f"\n========================================================")
    print(f"Starting Workflow: {WORKFLOW_NAME} (Run ID: {tracker.run_id})")
    print(f"Request: {req.get('request_id')} | Item: {req.get('item_description')} | Bidders: {supplier_ids}")
    print(f"========================================================")

    stage_evidence = {}

    try:
        # Agent 1: Quotation Analysis
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_QUOTATION_ID)
        agent1 = create_quotation_agent()
        task1 = create_quotation_task(agent1, req, quotes)
        crew1 = Crew(agents=[agent1], tasks=[task1], verbose=False)
        res1 = crew1.kickoff()
        out1 = res1.raw if hasattr(res1, "raw") else str(res1)
        tot1, p1, c1 = extract_token_usage(res1, crew1)
        dur1 = time.time() - t0
        tracker.log_agent_execution(AGENT_QUOTATION_ID, key_id, dur1, status="SUCCESS", tokens=tot1, prompt_tokens=p1, completion_tokens=c1, output=out1)
        stage_evidence["quotations"] = out1
        print(f" -> [Agent 1/9] {AGENT_QUOTATION_ID} completed in {dur1:.2f}s ({tot1} tokens)")

        # Agent 2: Supplier Profile
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_PROFILE_ID)
        profile_data = deterministic_profile_lookup(supplier_ids)
        agent2 = create_profile_agent()
        task2 = create_profile_task(agent2, supplier_ids)
        crew2 = Crew(agents=[agent2], tasks=[task2], verbose=False)
        res2 = crew2.kickoff()
        out2 = res2.raw if hasattr(res2, "raw") else str(res2)
        tot2, p2, c2 = extract_token_usage(res2, crew2)
        dur2 = time.time() - t0
        tracker.log_agent_execution(AGENT_PROFILE_ID, key_id, dur2, status="SUCCESS", tokens=tot2, prompt_tokens=p2, completion_tokens=c2, output=out2)
        stage_evidence["profiles"] = {"llm_output": out2, "profiles": profile_data}
        print(f" -> [Agent 2/9] {AGENT_PROFILE_ID} completed in {dur2:.2f}s ({tot2} tokens)")

        # Agent 3: Delivery Analysis
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_DELIVERY_ID)
        deliv_calc = deterministic_delivery_check(req_date, schedules)
        agent3 = create_delivery_agent()
        task3 = create_delivery_task(agent3, req_date, schedules)
        crew3 = Crew(agents=[agent3], tasks=[task3], verbose=False)
        res3 = crew3.kickoff()
        out3 = res3.raw if hasattr(res3, "raw") else str(res3)
        tot3, p3, c3 = extract_token_usage(res3, crew3)
        dur3 = time.time() - t0
        tracker.log_agent_execution(AGENT_DELIVERY_ID, key_id, dur3, status="SUCCESS", tokens=tot3, prompt_tokens=p3, completion_tokens=c3, output=out3)
        stage_evidence["delivery"] = {"llm_output": out3, "schedule_calc": deliv_calc}
        print(f" -> [Agent 3/9] {AGENT_DELIVERY_ID} completed in {dur2:.2f}s ({tot3} tokens)")

        # Agent 4: Supplier Risk
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_RISK_ID)
        risk_data = deterministic_risk_lookup(supplier_ids)
        agent4 = create_risk_agent()
        task4 = create_risk_task(agent4, supplier_ids)
        crew4 = Crew(agents=[agent4], tasks=[task4], verbose=False)
        res4 = crew4.kickoff()
        out4 = res4.raw if hasattr(res4, "raw") else str(res4)
        tot4, p4, c4 = extract_token_usage(res4, crew4)
        dur4 = time.time() - t0
        tracker.log_agent_execution(AGENT_RISK_ID, key_id, dur4, status="SUCCESS", tokens=tot4, prompt_tokens=p4, completion_tokens=c4, output=out4)
        stage_evidence["risk"] = {"llm_output": out4, "risk_data": risk_data}
        print(f" -> [Agent 4/9] {AGENT_RISK_ID} completed in {dur4:.2f}s ({tot4} tokens)")

        # Agent 5: Contract Analysis
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_CONTRACT_ID)
        contract_data = deterministic_contract_lookup(supplier_ids)
        agent5 = create_contract_agent()
        task5 = create_contract_task(agent5, supplier_ids)
        crew5 = Crew(agents=[agent5], tasks=[task5], verbose=False)
        res5 = crew5.kickoff()
        out5 = res5.raw if hasattr(res5, "raw") else str(res5)
        tot5, p5, c5 = extract_token_usage(res5, crew5)
        dur5 = time.time() - t0
        tracker.log_agent_execution(AGENT_CONTRACT_ID, key_id, dur5, status="SUCCESS", tokens=tot5, prompt_tokens=p5, completion_tokens=c5, output=out5)
        stage_evidence["contracts"] = {"llm_output": out5, "contracts": contract_data}
        print(f" -> [Agent 5/9] {AGENT_CONTRACT_ID} completed in {dur5:.2f}s ({tot5} tokens)")

        # Agent 6: Cost Comparison
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_COST_ID)
        cost_ranking = deterministic_cost_ranking(quotes)
        agent6 = create_cost_agent()
        task6 = create_cost_task(agent6, quotes)
        crew6 = Crew(agents=[agent6], tasks=[task6], verbose=False)
        res6 = crew6.kickoff()
        out6 = res6.raw if hasattr(res6, "raw") else str(res6)
        tot6, p6, c6 = extract_token_usage(res6, crew6)
        dur6 = time.time() - t0
        tracker.log_agent_execution(AGENT_COST_ID, key_id, dur6, status="SUCCESS", tokens=tot6, prompt_tokens=p6, completion_tokens=c6, output=out6)
        stage_evidence["cost"] = {"llm_output": out6, "ranking": cost_ranking}
        print(f" -> [Agent 6/9] {AGENT_COST_ID} completed in {dur6:.2f}s ({tot6} tokens)")

        # Agent 7: Weighted Evaluation
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_WEIGHTED_ID)
        scored_results = deterministic_weighted_scoring(cost_ranking, deliv_calc, risk_data, profile_data)
        agent7 = create_weighted_agent()
        task7 = create_weighted_task(agent7, scored_results)
        crew7 = Crew(agents=[agent7], tasks=[task7], verbose=False)
        res7 = crew7.kickoff()
        out7 = res7.raw if hasattr(res7, "raw") else str(res7)
        tot7, p7, c7 = extract_token_usage(res7, crew7)
        dur7 = time.time() - t0
        tracker.log_agent_execution(AGENT_WEIGHTED_ID, key_id, dur7, status="SUCCESS", tokens=tot7, prompt_tokens=p7, completion_tokens=c7, output=out7)
        stage_evidence["weighted_evaluation"] = {"llm_output": out7, "scores": scored_results}
        print(f" -> [Agent 7/9] {AGENT_WEIGHTED_ID} completed in {dur7:.2f}s ({tot7} tokens)")

        # Agent 8: Selection Recommendation
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_RECOMMENDATION_ID)
        agent8 = create_recommendation_agent()
        task8 = create_recommendation_task(agent8, stage_evidence)
        crew8 = Crew(agents=[agent8], tasks=[task8], verbose=False)
        res8 = crew8.kickoff()
        out8 = res8.raw if hasattr(res8, "raw") else str(res8)
        tot8, p8, c8 = extract_token_usage(res8, crew8)
        dur8 = time.time() - t0
        tracker.log_agent_execution(AGENT_RECOMMENDATION_ID, key_id, dur8, status="SUCCESS", tokens=tot8, prompt_tokens=p8, completion_tokens=c8, output=out8)
        stage_evidence["recommendation"] = out8
        print(f" -> [Agent 8/9] {AGENT_RECOMMENDATION_ID} completed in {dur8:.2f}s ({tot8} tokens)")

        # Agent 9: Final Review
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_REVIEW_ID)
        agent9 = create_review_agent()
        task9 = create_review_task(agent9, stage_evidence)
        crew9 = Crew(agents=[agent9], tasks=[task9], verbose=False)
        res9 = crew9.kickoff()
        final_output = res9.raw if hasattr(res9, "raw") else str(res9)
        tot9, p9, c9 = extract_token_usage(res9, crew9)
        dur9 = time.time() - t0
        tracker.log_agent_execution(AGENT_REVIEW_ID, key_id, dur9, status="SUCCESS", tokens=tot9, prompt_tokens=p9, completion_tokens=c9, output=final_output)
        print(f" -> [Agent 9/9] {AGENT_REVIEW_ID} completed in {dur9:.2f}s ({tot9} tokens)")

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
