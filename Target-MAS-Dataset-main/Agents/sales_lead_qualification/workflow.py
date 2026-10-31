"""
Workflow 4: Sales Lead Qualification Orchestrator.
Sequential 7-Agent Baseline System.
"""

import time
import json
from typing import Dict, Any, Optional
from crewai import Crew

from common.tracing import WorkflowRunTracker, init_phoenix, extract_token_usage
from common.api_key_manager import get_key_for_agent

from Agents.sales_lead_qualification.lead_enrichment_agent import (
    create_agent as create_enrichment_agent,
    create_task as create_enrichment_task,
    deterministic_enrichment,
    AGENT_ID as AGENT_ENRICHMENT_ID
)
from Agents.sales_lead_qualification.lead_fit_agent import (
    create_agent as create_fit_agent,
    create_task as create_fit_task,
    deterministic_icp_check,
    AGENT_ID as AGENT_FIT_ID
)
from Agents.sales_lead_qualification.intent_analysis_agent import (
    create_agent as create_intent_agent,
    create_task as create_intent_task,
    deterministic_intent_lookup,
    AGENT_ID as AGENT_INTENT_ID
)
from Agents.sales_lead_qualification.lead_scoring_agent import (
    create_agent as create_scoring_agent,
    create_task as create_scoring_task,
    deterministic_lead_score,
    AGENT_ID as AGENT_SCORING_ID
)
from Agents.sales_lead_qualification.opportunity_assessment_agent import (
    create_agent as create_opportunity_agent,
    create_task as create_opportunity_task,
    AGENT_ID as AGENT_OPPORTUNITY_ID
)
from Agents.sales_lead_qualification.next_action_agent import (
    create_agent as create_action_agent,
    create_task as create_action_task,
    deterministic_action_rule,
    AGENT_ID as AGENT_ACTION_ID
)
from Agents.sales_lead_qualification.final_review_agent import (
    create_agent as create_review_agent,
    create_task as create_review_task,
    AGENT_ID as AGENT_REVIEW_ID
)

WORKFLOW_NAME = "sales_lead_qualification"

def run_sales_lead_workflow(
    lead_data: Dict[str, Any],
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the 7-agent sequential Sales Lead Qualification baseline workflow.
    """
    init_phoenix()
    tracker = WorkflowRunTracker(WORKFLOW_NAME, run_id).start()

    print(f"\n========================================================")
    print(f"Starting Workflow: {WORKFLOW_NAME} (Run ID: {tracker.run_id})")
    print(f"Target Lead: {lead_data.get('lead_id', 'UNKNOWN')} | Contact: {lead_data.get('contact_name', 'UNKNOWN')}")
    print(f"========================================================")

    stage_evidence = {}

    try:
        # Agent 1: Lead Enrichment
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_ENRICHMENT_ID)
        comp_info = deterministic_enrichment(lead_data.get("company_id", ""))
        agent1 = create_enrichment_agent()
        task1 = create_enrichment_task(agent1, lead_data)
        crew1 = Crew(agents=[agent1], tasks=[task1], verbose=False)
        res1 = crew1.kickoff()
        out1 = res1.raw if hasattr(res1, "raw") else str(res1)
        tot1, p1, c1 = extract_token_usage(res1, crew1)
        dur1 = time.time() - t0
        tracker.log_agent_execution(AGENT_ENRICHMENT_ID, key_id, dur1, status="SUCCESS", tokens=tot1, prompt_tokens=p1, completion_tokens=c1, output=out1)
        stage_evidence["enrichment"] = {"llm_output": out1, "company_profile": comp_info}
        print(f" -> [Agent 1/7] {AGENT_ENRICHMENT_ID} completed in {dur1:.2f}s ({tot1} tokens)")

        # Agent 2: Lead Fit
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_FIT_ID)
        fit_calc = deterministic_icp_check(lead_data, comp_info)
        agent2 = create_fit_agent()
        task2 = create_fit_task(agent2, lead_data, comp_info)
        crew2 = Crew(agents=[agent2], tasks=[task2], verbose=False)
        res2 = crew2.kickoff()
        out2 = res2.raw if hasattr(res2, "raw") else str(res2)
        tot2, p2, c2 = extract_token_usage(res2, crew2)
        dur2 = time.time() - t0
        tracker.log_agent_execution(AGENT_FIT_ID, key_id, dur2, status="SUCCESS", tokens=tot2, prompt_tokens=p2, completion_tokens=c2, output=out2)
        stage_evidence["fit"] = {"llm_output": out2, "fit_eval": fit_calc}
        print(f" -> [Agent 2/7] {AGENT_FIT_ID} completed in {dur2:.2f}s ({tot2} tokens)")

        # Agent 3: Intent Analysis
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_INTENT_ID)
        events = deterministic_intent_lookup(lead_data.get("lead_id", ""))
        agent3 = create_intent_agent()
        task3 = create_intent_task(agent3, lead_data)
        crew3 = Crew(agents=[agent3], tasks=[task3], verbose=False)
        res3 = crew3.kickoff()
        out3 = res3.raw if hasattr(res3, "raw") else str(res3)
        tot3, p3, c3 = extract_token_usage(res3, crew3)
        dur3 = time.time() - t0
        tracker.log_agent_execution(AGENT_INTENT_ID, key_id, dur3, status="SUCCESS", tokens=tot3, prompt_tokens=p3, completion_tokens=c3, output=out3)
        stage_evidence["intent"] = {"llm_output": out3, "events": events}
        print(f" -> [Agent 3/7] {AGENT_INTENT_ID} completed in {dur3:.2f}s ({tot3} tokens)")

        # Agent 4: Lead Scoring
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_SCORING_ID)
        score_calc = deterministic_lead_score(lead_data, comp_info, events)
        agent4 = create_scoring_agent()
        task4 = create_scoring_task(agent4, lead_data, comp_info, events)
        crew4 = Crew(agents=[agent4], tasks=[task4], verbose=False)
        res4 = crew4.kickoff()
        out4 = res4.raw if hasattr(res4, "raw") else str(res4)
        tot4, p4, c4 = extract_token_usage(res4, crew4)
        dur4 = time.time() - t0
        tracker.log_agent_execution(AGENT_SCORING_ID, key_id, dur4, status="SUCCESS", tokens=tot4, prompt_tokens=p4, completion_tokens=c4, output=out4)
        stage_evidence["scoring"] = {"llm_output": out4, "deterministic_score": score_calc}
        print(f" -> [Agent 4/7] {AGENT_SCORING_ID} completed in {dur4:.2f}s ({tot4} tokens)")

        # Agent 5: Opportunity Assessment
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_OPPORTUNITY_ID)
        agent5 = create_opportunity_agent()
        task5 = create_opportunity_task(agent5, stage_evidence)
        crew5 = Crew(agents=[agent5], tasks=[task5], verbose=False)
        res5 = crew5.kickoff()
        out5 = res5.raw if hasattr(res5, "raw") else str(res5)
        tot5, p5, c5 = extract_token_usage(res5, crew5)
        dur5 = time.time() - t0
        tracker.log_agent_execution(AGENT_OPPORTUNITY_ID, key_id, dur5, status="SUCCESS", tokens=tot5, prompt_tokens=p5, completion_tokens=c5, output=out5)
        stage_evidence["opportunity"] = out5
        print(f" -> [Agent 5/7] {AGENT_OPPORTUNITY_ID} completed in {dur5:.2f}s ({tot5} tokens)")

        # Agent 6: Next Action Recommendation
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_ACTION_ID)
        agent6 = create_action_agent()
        task6 = create_action_task(agent6, stage_evidence)
        crew6 = Crew(agents=[agent6], tasks=[task6], verbose=False)
        res6 = crew6.kickoff()
        out6 = res6.raw if hasattr(res6, "raw") else str(res6)
        tot6, p6, c6 = extract_token_usage(res6, crew6)
        dur6 = time.time() - t0
        tracker.log_agent_execution(AGENT_ACTION_ID, key_id, dur6, status="SUCCESS", tokens=tot6, prompt_tokens=p6, completion_tokens=c6, output=out6)
        stage_evidence["action"] = out6
        print(f" -> [Agent 6/7] {AGENT_ACTION_ID} completed in {dur6:.2f}s ({tot6} tokens)")

        # Agent 7: Final Review
        t0 = time.time()
        _, _, key_id = get_key_for_agent(AGENT_REVIEW_ID)
        agent7 = create_review_agent()
        task7 = create_review_task(agent7, stage_evidence)
        crew7 = Crew(agents=[agent7], tasks=[task7], verbose=False)
        res7 = crew7.kickoff()
        final_output = res7.raw if hasattr(res7, "raw") else str(res7)
        tot7, p7, c7 = extract_token_usage(res7, crew7)
        dur7 = time.time() - t0
        tracker.log_agent_execution(AGENT_REVIEW_ID, key_id, dur7, status="SUCCESS", tokens=tot7, prompt_tokens=p7, completion_tokens=c7, output=final_output)
        print(f" -> [Agent 7/7] {AGENT_REVIEW_ID} completed in {dur7:.2f}s ({tot7} tokens)")

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
