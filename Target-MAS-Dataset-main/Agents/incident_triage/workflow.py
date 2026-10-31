"""IT Incident Triage Baseline Multi-Agent Workflow.

Workflow One: Exactly 7 agents executing sequentially:
1. Intake Agent
2. Classification Agent
3. Runbook Agent
4. Symptom Agent
5. Severity Agent
6. Remediation Agent
7. Review Agent
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from crewai import Crew, Process, Task

from Agents.incident_triage.classification_agent import create_classification_agent
from Agents.incident_triage.intake_agent import create_intake_agent
from Agents.incident_triage.remediation_agent import create_remediation_agent
from Agents.incident_triage.review_agent import create_review_agent
from Agents.incident_triage.runbook_agent import create_runbook_agent, query_candidate_runbooks
from Agents.incident_triage.severity_agent import calculate_deterministic_severity, create_severity_agent
from Agents.incident_triage.symptom_agent import create_symptom_agent, query_historical_evidence
from common.agent_runner import ACTIVE_TRACKER, WorkflowExecutionResult, generate_run_id, save_run_result
from common.tracing import agent_span, workflow_span


def run_incident_triage_workflow(
    incident_input: Dict[str, Any],
    run_id: Optional[str] = None
) -> WorkflowExecutionResult:
    """Execute the 7-agent IT Incident Triage workflow."""
    ACTIVE_TRACKER.reset()
    run_id = run_id or generate_run_id("incident")
    start_total = time.perf_counter()
    agent_durations: Dict[str, float] = {}
    order = [
        "intake_agent",
        "classification_agent",
        "runbook_agent",
        "symptom_agent",
        "severity_agent",
        "remediation_agent",
        "incident_review_agent",
    ]

    with workflow_span("incident_triage", run_id, incident_input):
        # 1. Intake Agent
        t0 = time.perf_counter()
        with agent_span("intake_agent", run_id, incident_input):
            intake_agent = create_intake_agent()
            intake_task = Task(
                description=(
                    f"Normalize this raw incident ticket and extract structured fields:\n"
                    f"Raw Ticket: {json.dumps(incident_input, indent=2)}\n"
                    f"Return JSON with: normalized_title, affected_service, symptoms, timestamps, "
                    f"reported_impact_users, outage_minutes, and missing_fields."
                ),
                expected_output="Valid JSON string containing normalized incident telemetry and missing fields.",
                agent=intake_agent,
            )
            crew1 = Crew(agents=[intake_agent], tasks=[intake_task], process=Process.sequential, verbose=False)
            intake_output = crew1.kickoff()
        agent_durations["intake_agent"] = time.perf_counter() - t0

        # Parse or wrap intake output
        intake_text = str(intake_output.raw if hasattr(intake_output, "raw") else intake_output)

        # 2. Classification Agent
        t0 = time.perf_counter()
        with agent_span("classification_agent", run_id, {"intake_data": intake_text}):
            classification_agent = create_classification_agent()
            classification_task = Task(
                description=(
                    f"Based on the intake findings below, classify the primary failure category "
                    f"(network_failure, application_error, authentication_issue, database_problem, storage_exhaustion) "
                    f"and apparent complexity (Simple, Moderate, Complex):\n"
                    f"Intake Findings: {intake_text}"
                ),
                expected_output="JSON containing category, apparent_complexity, and classification_reasoning.",
                agent=classification_agent,
            )
            crew2 = Crew(agents=[classification_agent], tasks=[classification_task], process=Process.sequential, verbose=False)
            classification_output = crew2.kickoff()
        agent_durations["classification_agent"] = time.perf_counter() - t0

        classification_text = str(classification_output.raw if hasattr(classification_output, "raw") else classification_output)

        # Deterministic extraction of category for local runbook & historical search
        detected_category = "database_problem"
        for cat in ["network_failure", "application_error", "authentication_issue", "database_problem", "storage_exhaustion"]:
            if cat in classification_text.lower():
                detected_category = cat
                break

        # 3. Runbook Agent (Deterministic query + LLM reasoning)
        t0 = time.perf_counter()
        candidate_runbooks = query_candidate_runbooks(detected_category, limit=3)
        with agent_span("runbook_agent", run_id, {"candidate_runbooks_count": len(candidate_runbooks)}):
            runbook_agent = create_runbook_agent()
            runbook_task = Task(
                description=(
                    f"Evaluate these candidate operational runbooks for category '{detected_category}':\n"
                    f"{json.dumps(candidate_runbooks, indent=2)}\n"
                    f"Incident Telemetry: {intake_text}\n"
                    f"Select the best matching runbook_id, outline why it applies, and state its safety notes."
                ),
                expected_output="JSON with selected_runbook_id, match_confidence, prerequisites, and safety_notes.",
                agent=runbook_agent,
            )
            crew3 = Crew(agents=[runbook_agent], tasks=[runbook_task], process=Process.sequential, verbose=False)
            runbook_output = crew3.kickoff()
        agent_durations["runbook_agent"] = time.perf_counter() - t0

        runbook_text = str(runbook_output.raw if hasattr(runbook_output, "raw") else runbook_output)

        # 4. Symptom Agent (Correlate with historical data & logs)
        t0 = time.perf_counter()
        target_service = incident_input.get("service", "unknown-service")
        evidence_data = query_historical_evidence(target_service, detected_category, limit=3)
        with agent_span("symptom_agent", run_id, {"historical_evidence_count": len(evidence_data["incidents"])}):
            symptom_agent = create_symptom_agent()
            symptom_task = Task(
                description=(
                    f"Correlate current symptoms with historical incidents and log records:\n"
                    f"Historical Evidence: {json.dumps(evidence_data, indent=2)}\n"
                    f"Current Symptoms: {intake_text}\n"
                    f"Return matching past incident IDs, recurring error signatures, and diagnosis confidence."
                ),
                expected_output="JSON with correlated_incident_ids, recurring_patterns, and diagnostic_stage_assessment.",
                agent=symptom_agent,
            )
            crew4 = Crew(agents=[symptom_agent], tasks=[symptom_task], process=Process.sequential, verbose=False)
            symptom_output = crew4.kickoff()
        agent_durations["symptom_agent"] = time.perf_counter() - t0

        symptom_text = str(symptom_output.raw if hasattr(symptom_output, "raw") else symptom_output)

        # 5. Severity Agent (Deterministic rules + LLM justification)
        t0 = time.perf_counter()
        users_impacted = incident_input.get("impact_users", 250)
        srv_tier = incident_input.get("service_tier", "Tier-2")
        outage_dur = incident_input.get("outage_minutes", 30)
        deterministic_sev = calculate_deterministic_severity(users_impacted, srv_tier, outage_dur)

        with agent_span("severity_agent", run_id, deterministic_sev):
            severity_agent = create_severity_agent()
            severity_task = Task(
                description=(
                    f"Deterministic Severity Rule Result:\n{json.dumps(deterministic_sev, indent=2)}\n"
                    f"Telemetry: Users={users_impacted}, ServiceTier={srv_tier}, OutageDuration={outage_dur}m.\n"
                    f"Correlated Context: {symptom_text}\n"
                    f"Confirm or refine the final severity level (SEV-1..SEV-4) with clear SLA justification."
                ),
                expected_output="JSON with severity_level, governing_sla_rule, and justification.",
                agent=severity_agent,
            )
            crew5 = Crew(agents=[severity_agent], tasks=[severity_task], process=Process.sequential, verbose=False)
            severity_output = crew5.kickoff()
        agent_durations["severity_agent"] = time.perf_counter() - t0

        severity_text = str(severity_output.raw if hasattr(severity_output, "raw") else severity_output)

        # 6. Remediation Agent
        t0 = time.perf_counter()
        with agent_span("remediation_agent", run_id, {"runbook": runbook_text}):
            remediation_agent = create_remediation_agent()
            remediation_task = Task(
                description=(
                    f"Based on the selected runbook and diagnosis:\n"
                    f"Runbook: {runbook_text}\n"
                    f"Diagnosis & Symptoms: {symptom_text}\n"
                    f"Recommend step-by-step troubleshooting and remediation actions. "
                    f"Must be safe, documented, and fully reversible. Do not execute real infrastructure changes."
                ),
                expected_output="JSON with ordered remediation_steps, rollback_plan, and safety_verification.",
                agent=remediation_agent,
            )
            crew6 = Crew(agents=[remediation_agent], tasks=[remediation_task], process=Process.sequential, verbose=False)
            remediation_output = crew6.kickoff()
        agent_durations["remediation_agent"] = time.perf_counter() - t0

        remediation_text = str(remediation_output.raw if hasattr(remediation_output, "raw") else remediation_output)

        # 7. Review Agent
        t0 = time.perf_counter()
        with agent_span("incident_review_agent", run_id, {}):
            review_agent = create_review_agent()
            review_task = Task(
                description=(
                    f"Audit all 6 preceding stages for agreement and missing info:\n"
                    f"Intake: {intake_text}\n"
                    f"Classification: {classification_text}\n"
                    f"Runbook: {runbook_text}\n"
                    f"Symptoms: {symptom_text}\n"
                    f"Severity: {severity_text}\n"
                    f"Remediation: {remediation_text}\n"
                    f"Produce the final structured incident triage report in clean JSON."
                ),
                expected_output="Complete final incident triage JSON report.",
                agent=review_agent,
            )
            crew7 = Crew(agents=[review_agent], tasks=[review_task], process=Process.sequential, verbose=False)
            review_output = crew7.kickoff()
        agent_durations["incident_review_agent"] = time.perf_counter() - t0

        review_text = str(review_output.raw if hasattr(review_output, "raw") else review_output)

    total_duration = time.perf_counter() - start_total

    final_payload = {
        "workflow": "incident_triage",
        "run_id": run_id,
        "input": incident_input,
        "stages": {
            "intake": intake_text,
            "classification": classification_text,
            "runbook": runbook_text,
            "symptoms": symptom_text,
            "severity": severity_text,
            "remediation": remediation_text,
            "final_review": review_text,
        },
        "synthetic": True,
    }

    tracker_data = ACTIVE_TRACKER.snapshot()
    result = WorkflowExecutionResult(
        workflow_name="incident_triage",
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
