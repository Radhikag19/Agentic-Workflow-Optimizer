"""Market and Competitor Research Baseline Multi-Agent Workflow.

Workflow Three: Exactly 9 agents executing sequentially:
1. Planning Agent
2. Profile Agent
3. Product Agent
4. Pricing Agent
5. Feedback Agent
6. Evidence Agent
7. Comparison Agent
8. Synthesis Agent
9. Fact-Check Agent
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from crewai import Crew, Process, Task

from Agents.market_research.comparison_agent import create_comparison_agent
from Agents.market_research.evidence_agent import create_evidence_agent
from Agents.market_research.fact_check_agent import create_fact_check_agent
from Agents.market_research.feedback_agent import create_feedback_agent, query_feedback_documents
from Agents.market_research.planning_agent import create_planning_agent
from Agents.market_research.pricing_agent import create_pricing_agent, query_pricing_documents
from Agents.market_research.product_agent import create_product_agent, query_product_documents
from Agents.market_research.profile_agent import create_profile_agent, query_company_profiles
from Agents.market_research.synthesis_agent import create_synthesis_agent
from common.agent_runner import ACTIVE_TRACKER, WorkflowExecutionResult, generate_run_id, save_run_result
from common.tracing import agent_span, workflow_span


def run_market_research_workflow(
    research_query: Dict[str, Any],
    run_id: Optional[str] = None
) -> WorkflowExecutionResult:
    """Execute the 9-agent Market and Competitor Research workflow."""
    ACTIVE_TRACKER.reset()
    run_id = run_id or generate_run_id("market")
    start_total = time.perf_counter()
    agent_durations: Dict[str, float] = {}
    order = [
        "planning_agent",
        "profile_agent",
        "product_agent",
        "pricing_agent",
        "feedback_agent",
        "evidence_agent",
        "comparison_agent",
        "synthesis_agent",
        "fact_check_agent",
    ]

    with workflow_span("market_research", run_id, research_query):
        # 1. Planning Agent
        t0 = time.perf_counter()
        q_text = research_query.get("query", "Comprehensive evaluation of enterprise agent platforms")
        with agent_span("planning_agent", run_id, {"query": q_text}):
            planning_agent = create_planning_agent()
            planning_task = Task(
                description=(
                    f"Develop a structured research plan for this market inquiry:\n"
                    f"Inquiry: {q_text}\n"
                    f"Identify key comparison dimensions (architecture, throughput, pricing, SLA, customer sentiment) "
                    f"for competitors: CloudScale AI, DataPulse Systems, NexusOps Technologies, and Veloce Security."
                ),
                expected_output="JSON containing target_competitors, evaluation_dimensions, and evidence_gathering_plan.",
                agent=planning_agent,
            )
            crew1 = Crew(agents=[planning_agent], tasks=[planning_task], process=Process.sequential, verbose=False)
            plan_out = crew1.kickoff()
        agent_durations["planning_agent"] = time.perf_counter() - t0
        plan_text = str(plan_out.raw if hasattr(plan_out, "raw") else plan_out)

        # 2. Profile Agent (Query profiles from dataset)
        t0 = time.perf_counter()
        profiles_data = query_company_profiles()
        with agent_span("profile_agent", run_id, {"competitors_count": len(profiles_data.get("competitors", []))}):
            profile_agent = create_profile_agent()
            profile_task = Task(
                description=(
                    f"Using this corporate profiles dataset:\n{json.dumps(profiles_data, indent=2)}\n"
                    f"Extract headquarters, founding year, headcount, funding, and positioning for each competitor. "
                    f"Cite all source document IDs (e.g., DOC-CS-01, DOC-DP-01)."
                ),
                expected_output="JSON structured competitor profiles with cited doc_ids.",
                agent=profile_agent,
            )
            crew2 = Crew(agents=[profile_agent], tasks=[profile_task], process=Process.sequential, verbose=False)
            prof_out = crew2.kickoff()
        agent_durations["profile_agent"] = time.perf_counter() - t0
        prof_text = str(prof_out.raw if hasattr(prof_out, "raw") else prof_out)

        # 3. Product Agent (Query product documents)
        t0 = time.perf_counter()
        product_docs = query_product_documents()
        with agent_span("product_agent", run_id, {"product_docs_count": len(product_docs)}):
            product_agent = create_product_agent()
            product_task = Task(
                description=(
                    f"Analyze technical capabilities from these source documents:\n{json.dumps(product_docs, indent=2)}\n"
                    f"Extract features, latency/throughput metrics, SDKs, and architectural limitations for each competitor. "
                    f"Cite specific source doc_ids for each claim."
                ),
                expected_output="JSON structured product capabilities by competitor with doc_id citations.",
                agent=product_agent,
            )
            crew3 = Crew(agents=[product_agent], tasks=[product_task], process=Process.sequential, verbose=False)
            prod_out = crew3.kickoff()
        agent_durations["product_agent"] = time.perf_counter() - t0
        prod_text = str(prod_out.raw if hasattr(prod_out, "raw") else prod_out)

        # 4. Pricing Agent (Query pricing documents)
        t0 = time.perf_counter()
        pricing_docs = query_pricing_documents()
        with agent_span("pricing_agent", run_id, {"pricing_docs_count": len(pricing_docs)}):
            pricing_agent = create_pricing_agent()
            pricing_task = Task(
                description=(
                    f"Extract commercial pricing and licensing from these documents:\n{json.dumps(pricing_docs, indent=2)}\n"
                    f"Detail monthly subscription tiers, overage clauses, and contract minimums. "
                    f"Preserve exact currency units and mark unknown values. Cite source doc_ids."
                ),
                expected_output="JSON structured pricing breakdown with doc_id citations.",
                agent=pricing_agent,
            )
            crew4 = Crew(agents=[pricing_agent], tasks=[pricing_task], process=Process.sequential, verbose=False)
            price_out = crew4.kickoff()
        agent_durations["pricing_agent"] = time.perf_counter() - t0
        price_text = str(price_out.raw if hasattr(price_out, "raw") else price_out)

        # 5. Feedback Agent (Query feedback documents)
        t0 = time.perf_counter()
        feedback_docs = query_feedback_documents()
        with agent_span("feedback_agent", run_id, {"feedback_docs_count": len(feedback_docs)}):
            feedback_agent = create_feedback_agent()
            feedback_task = Task(
                description=(
                    f"Analyze customer reviews and reported field experiences:\n{json.dumps(feedback_docs, indent=2)}\n"
                    f"Summarize user feedback, reported outages, praise, and usability friction. "
                    f"Separate recorded quotes from subjective conclusions and cite doc_ids."
                ),
                expected_output="JSON customer feedback analysis citing specific review doc_ids.",
                agent=feedback_agent,
            )
            crew5 = Crew(agents=[feedback_agent], tasks=[feedback_task], process=Process.sequential, verbose=False)
            feed_out = crew5.kickoff()
        agent_durations["feedback_agent"] = time.perf_counter() - t0
        feed_text = str(feed_out.raw if hasattr(feed_out, "raw") else feed_out)

        # 6. Evidence Agent (Cross-Examination and Conflict Detection)
        t0 = time.perf_counter()
        with agent_span("evidence_agent", run_id, {}):
            evidence_agent = create_evidence_agent()
            evidence_task = Task(
                description=(
                    f"Cross-examine the evidence collected across all 4 domains:\n"
                    f"Profiles: {prof_text}\n"
                    f"Product: {prod_text}\n"
                    f"Pricing: {price_text}\n"
                    f"Feedback: {feed_text}\n"
                    f"Identify deliberate conflicts (e.g., uptime SLA vs customer reported outages, flat fees vs hidden overages). "
                    f"Flag any unsupported claims and preserve all source doc_ids."
                ),
                expected_output="JSON containing triangulated evidence, verified contradictions, and citation gaps.",
                agent=evidence_agent,
            )
            crew6 = Crew(agents=[evidence_agent], tasks=[evidence_task], process=Process.sequential, verbose=False)
            evid_out = crew6.kickoff()
        agent_durations["evidence_agent"] = time.perf_counter() - t0
        evid_text = str(evid_out.raw if hasattr(evid_out, "raw") else evid_out)

        # 7. Comparison Agent (Structured Matrix Generation)
        t0 = time.perf_counter()
        with agent_span("comparison_agent", run_id, {}):
            comparison_agent = create_comparison_agent()
            comparison_task = Task(
                description=(
                    f"Construct a side-by-side competitor matrix for CloudScale AI, DataPulse Systems, "
                    f"NexusOps Technologies, and Veloce Security across dimensions:\n"
                    f"Focus, Architecture, Key Strengths, Known Weaknesses, Starting Price, and SLA.\n"
                    f"Correlated Evidence: {evid_text}\n"
                    f"Clearly distinguish factual specs from analytical opinions. Include doc_ids."
                ),
                expected_output="Structured competitor comparison matrix with citations.",
                agent=comparison_agent,
            )
            crew7 = Crew(agents=[comparison_agent], tasks=[comparison_task], process=Process.sequential, verbose=False)
            comp_out = crew7.kickoff()
        agent_durations["comparison_agent"] = time.perf_counter() - t0
        comp_text = str(comp_out.raw if hasattr(comp_out, "raw") else comp_out)

        # 8. Synthesis Agent (Executive Research Narrative)
        t0 = time.perf_counter()
        with agent_span("synthesis_agent", run_id, {}):
            synthesis_agent = create_synthesis_agent()
            synthesis_task = Task(
                description=(
                    f"Author an executive competitive research report synthesized from:\n"
                    f"Comparison Matrix: {comp_text}\n"
                    f"Evidence & Contradictions: {evid_text}\n"
                    f"Structure into: Executive Summary, Competitive Landscape, Strategic Trade-Offs, "
                    f"and Key Recommendations. Avoid repetitive prose."
                ),
                expected_output="Incisive executive research narrative with integrated source citations.",
                agent=synthesis_agent,
            )
            crew8 = Crew(agents=[synthesis_agent], tasks=[synthesis_task], process=Process.sequential, verbose=False)
            synth_out = crew8.kickoff()
        agent_durations["synthesis_agent"] = time.perf_counter() - t0
        synth_text = str(synth_out.raw if hasattr(synth_out, "raw") else synth_out)

        # 9. Fact-Check Agent (Audit Citations against Documents)
        t0 = time.perf_counter()
        with agent_span("fact_check_agent", run_id, {}):
            fact_check_agent = create_fact_check_agent()
            fact_check_task = Task(
                description=(
                    f"Perform an audit on this final synthesis report:\n"
                    f"{synth_text}\n"
                    f"Verify that key claims reference valid doc_ids (DOC-CS-*, DOC-DP-*, DOC-NO-*, DOC-VS-*). "
                    f"Flag any unsupported statements. Confirm the report is marked as synthetic data."
                ),
                expected_output="Final checked report with fact-check audit summary and verified citations.",
                agent=fact_check_agent,
            )
            crew9 = Crew(agents=[fact_check_agent], tasks=[fact_check_task], process=Process.sequential, verbose=False)
            fact_out = crew9.kickoff()
        agent_durations["fact_check_agent"] = time.perf_counter() - t0
        fact_text = str(fact_out.raw if hasattr(fact_out, "raw") else fact_out)

    total_duration = time.perf_counter() - start_total

    final_payload = {
        "workflow": "market_research",
        "run_id": run_id,
        "input": research_query,
        "stages": {
            "planning": plan_text,
            "profiles": prof_text,
            "product": prod_text,
            "pricing": price_text,
            "feedback": feed_text,
            "evidence": evid_text,
            "comparison": comp_text,
            "synthesis": synth_text,
            "fact_checked_report": fact_text,
        },
        "synthetic": True,
    }

    tracker_data = ACTIVE_TRACKER.snapshot()
    result = WorkflowExecutionResult(
        workflow_name="market_research",
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
