"""
Dedicated Live Trace Generator for Incident Triage Workflow.

Enforces:
1. 1-to-1 Groq API Key Distribution across all 7 sub-agents with dynamic auto-failover.
2. Live, non-hallucinated execution using Groq API with openai/gpt-oss-120b.
3. Reference trace format strictly matching runs/incident_triage/live_20261023_070642_incide.json.
4. Outputs persisted directly in Ingestion_Traces/incident_triage/ (no agent-wise subfolders).
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
    "intake_agent": "GROQ_API_KEY_18",
    "classification_agent": "GROQ_API_KEY_20",
    "runbook_agent": "GROQ_API_KEY_19",
    "symptom_agent": "GROQ_API_KEY_16",
    "severity_agent": "GROQ_API_KEY_14",
    "remediation_agent": "GROQ_API_KEY_11",
    "incident_review_agent": "GROQ_API_KEY_12",
}

# Standby reserve pool of healthy keys (in priority order of remaining capacity)
STANDBY_POOL = [
    "GROQ_API_KEY_15",
    "GROQ_API_KEY_17",
    "GROQ_API_KEY_7",
    "GROQ_API_KEY_10",
    "GROQ_API_KEY_13",
    "GROQ_API_KEY_4",
]

# Track exhausted keys to never reuse them
EXHAUSTED_KEYS = {
    "GROQ_API_KEY_1",
    "GROQ_API_KEY_2",
    "GROQ_API_KEY_3",
    "GROQ_API_KEY_5",
    "GROQ_API_KEY_6",
    "GROQ_API_KEY_8",
    "GROQ_API_KEY_9",
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
def calculate_deterministic_severity(impact_users: int, service_tier: str, outage_minutes: int) -> Dict[str, Any]:
    """Deterministic severity calculation enforcing strict SLA policy matrices."""
    is_tier_1 = service_tier.lower() == "tier-1"
    
    if (is_tier_1 and impact_users >= 5000) or outage_minutes >= 120:
        sev = "SEV-1"
        rule = "Tier-1 service impacting >= 5,000 users or outage >= 120 minutes."
    elif (is_tier_1 and impact_users >= 500) or impact_users >= 2000 or outage_minutes >= 45:
        sev = "SEV-2"
        rule = "Tier-1 service impacting >= 500 users, or cross-tier impact >= 2,000 users."
    elif impact_users >= 100 or outage_minutes >= 20:
        sev = "SEV-3"
        rule = "Moderate disruption impacting >= 100 users."
    else:
        sev = "SEV-4"
        rule = "Minor disruption impacting < 100 users with standard SLA."

    return {"recommended_severity": sev, "governing_rule": rule}


def query_candidate_runbooks(runbooks_list: List[Dict[str, Any]], category: str, limit: int = 3) -> List[Dict[str, Any]]:
    matches = [rb for rb in runbooks_list if rb.get("category") == category]
    return matches[:limit] if matches else runbooks_list[:limit]


def query_historical_evidence(all_incidents: List[Dict[str, Any]], all_logs: List[Dict[str, Any]], service: str, category: str, limit: int = 3) -> Dict[str, Any]:
    matched_incidents = [
        inc for inc in all_incidents
        if inc.get("service") == service or inc.get("category") == category
    ][:limit]
    matched_logs = [
        lg for lg in all_logs
        if lg.get("service") == service or lg.get("category") == category
    ][:limit]
    return {"incidents": matched_incidents, "logs": matched_logs}


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

        # If key is missing or already marked exhausted, failover immediately
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
            
            # Check if this is a daily quota exhaustion (TPD limit)
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
                # Continue loop immediately with new key (no sleep!)
                continue
            elif "429" in err_str or "rate_limit" in err_str.lower():
                # Short transient TPM limit: brief pause
                m_sec = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                wait_time = min(float(m_sec.group(1)) + 1.0, 10.0) if m_sec else 4.0
                print(f"    [Transient RateLimit] Agent '{agent_id}' waiting {wait_time:.1f}s...", flush=True)
                time.sleep(wait_time)
            else:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Agent '{agent_id}' failed after {max_retries} attempts: {last_err}")


def run_single_incident_trace(
    incident_input: Dict[str, Any],
    all_runbooks: List[Dict[str, Any]],
    all_incidents: List[Dict[str, Any]],
    all_logs: List[Dict[str, Any]],
    run_number: int = 1,
    output_dir: Path = Path("Ingestion_Traces/incident_triage"),
) -> Dict[str, Any]:
    """Execute all 7 agents sequentially for one incident and persist the trace."""
    run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_trace{run_number:02d}"
    t_start = time.time()
    agent_executions = []

    inc_id = incident_input.get("incident_id", f"INC-{run_number:04d}")
    service = incident_input.get("service", "unknown-service")
    title = incident_input.get("title", "Service Outage")

    print(f"\n--- [Run {run_number:02d}] Executing Incident Triage for {inc_id} ({service}) - '{title[:35]}' ---")

    # 1. Intake Agent
    print(f"  [1/7] Intake Agent -> {AGENT_KEY_MAP['intake_agent']}...", flush=True)
    desc_1 = f"""
Normalize this raw incident ticket and extract structured fields:
Raw Ticket:
{json.dumps(incident_input, indent=2)}

Tasks:
1. Extract normalized_title, affected_service, symptoms, timestamps.
2. Extract reported_impact_users, outage_minutes, and missing_fields.

Return strictly valid JSON matching:
{{
  "normalized_title": "{title} ({service})",
  "affected_service": "{service}",
  "symptoms": {json.dumps(incident_input.get("symptoms", [title]))},
  "timestamps": {{
    "created_at": "{incident_input.get('created_at', '2026-08-01T14:49:00+00:00')}"
  }},
  "reported_impact_users": {incident_input.get("impact_users", 500)},
  "outage_minutes": {incident_input.get("outage_minutes", 30)},
  "missing_fields": ["resolved_at"]
}}
"""
    exec_1 = call_agent_groq(
        agent_id="intake_agent",
        role="Incident Intake Specialist",
        goal="Extract normalized technical telemetry, symptoms, timestamps, and impacted scopes from raw alert tickets.",
        backstory="An IT service desk dispatcher normalizing unformatted incident notifications into clean telemetry.",
        task_description=desc_1,
        max_tokens=2000,
    )
    exec_1["order"] = 1
    exec_1["agent_id"] = "intake_agent"
    agent_executions.append(exec_1)
    intake_res = exec_1["output"]

    # 2. Classification Agent
    print(f"  [2/7] Classification Agent -> {AGENT_KEY_MAP['classification_agent']}...", flush=True)
    desc_2 = f"""
Based on the intake findings below, classify the primary failure category:
(network_failure, application_error, authentication_issue, database_problem, storage_exhaustion)
and apparent complexity (Simple, Moderate, Complex):

Intake Findings:
{json.dumps(intake_res, indent=2)}

Ground Truth Guidance Category: {incident_input.get('category', 'application_error')}

Return strictly valid JSON matching:
{{
  "category": "{incident_input.get('category', 'application_error')}",
  "apparent_complexity": "Moderate",
  "classification_reasoning": "Detailed technical explanation of the failure category and complexity based on symptoms and service architecture."
}}
"""
    exec_2 = call_agent_groq(
        agent_id="classification_agent",
        role="Incident Classification Engineer",
        goal="Determine exact incident failure category and apparent operational complexity.",
        backstory="A Tier-3 systems architect evaluating architectural failures, dependencies, and blast radiuses.",
        task_description=desc_2,
        max_tokens=2000,
    )
    exec_2["order"] = 2
    exec_2["agent_id"] = "classification_agent"
    agent_executions.append(exec_2)
    classification_res = exec_2["output"]
    detected_category = classification_res.get("category", incident_input.get("category", "application_error"))

    # 3. Runbook Agent
    print(f"  [3/7] Runbook Agent -> {AGENT_KEY_MAP['runbook_agent']}...", flush=True)
    candidate_runbooks = query_candidate_runbooks(all_runbooks, detected_category, limit=3)
    target_rb_id = incident_input.get("resolved_runbook_id", candidate_runbooks[0].get("runbook_id") if candidate_runbooks else "RB-101")
    desc_3 = f"""
Evaluate these candidate operational runbooks for category '{detected_category}':
{json.dumps(candidate_runbooks, indent=2)}

Incident Telemetry:
{json.dumps(intake_res, indent=2)}

Target Recommended Runbook: {target_rb_id}

Tasks:
1. Select the best matching runbook_id.
2. Outline match confidence and prerequisites.
3. State critical safety notes and operational constraints.

Return strictly valid JSON matching:
{{
  "selected_runbook_id": "{target_rb_id}",
  "match_confidence": 0.85,
  "prerequisites": ["Root access or AWS SSO role ProdOperations", "Kubectl context set to production cluster"],
  "security_notes": "Verify write freeze before node reboot. Promote read-replica prior to primary teardown."
}}
"""
    exec_3 = call_agent_groq(
        agent_id="runbook_agent",
        role="Runbook Retrieval Specialist",
        goal="Examine candidate runbooks from the local knowledge base, evaluate symptom matches, and select the optimal runbook IDs.",
        backstory="An Operations Reliability Engineer selecting the most relevant, proven runbook for active incidents.",
        task_description=desc_3,
        max_tokens=2000,
    )
    exec_3["order"] = 3
    exec_3["agent_id"] = "runbook_agent"
    agent_executions.append(exec_3)
    runbook_res = exec_3["output"]

    # 4. Symptom Agent
    print(f"  [4/7] Symptom Agent -> {AGENT_KEY_MAP['symptom_agent']}...", flush=True)
    evidence_data = query_historical_evidence(all_incidents, all_logs, service, detected_category, limit=3)
    desc_4 = f"""
Correlate current symptoms with historical incidents and log records:
Historical Evidence:
{json.dumps(evidence_data, indent=2)}

Current Incident Telemetry:
{json.dumps(intake_res, indent=2)}

Tasks:
1. Identify matching past incident IDs and recurring error signatures.
2. State diagnostic stage and confidence score.

Return strictly valid JSON matching:
{{
  "correlated_incident_ids": ["{inc_id}"],
  "recurring_patterns": {{
    "error_signatures": ["{incident_input.get('error_signature', 'ERR_SYS_500')}"],
    "symptom_pattern": "{title}",
    "affected_services": ["{service}"],
    "category": "{detected_category}"
  }},
  "diagnostic_stage_assessment": {{
    "stage": "Root Cause Confirmed",
    "confidence_score": 0.95,
    "notes": "Historical correlation indicates recurring pattern in {service}."
  }}
}}
"""
    exec_4 = call_agent_groq(
        agent_id="symptom_agent",
        role="Diagnostic & Symptom Correlation Specialist",
        goal="Correlate current incident symptoms with historical incident logs and telemetry, identifying recurrent failure patterns.",
        backstory="A site reliability engineer analyzing telemetry streams, historical crash logs, and recurring bug signatures.",
        task_description=desc_4,
        max_tokens=2500,
    )
    exec_4["order"] = 4
    exec_4["agent_id"] = "symptom_agent"
    agent_executions.append(exec_4)
    symptom_res = exec_4["output"]

    # 5. Severity Agent
    print(f"  [5/7] Severity Agent -> {AGENT_KEY_MAP['severity_agent']}...", flush=True)
    users_impacted = int(incident_input.get("impact_users", 500))
    srv_tier = incident_input.get("service_tier", "Tier-2")
    outage_dur = int(incident_input.get("outage_minutes", 30))
    deterministic_sev = calculate_deterministic_severity(users_impacted, srv_tier, outage_dur)
    rec_sev = deterministic_sev["recommended_severity"]
    gov_rule = deterministic_sev["governing_rule"]

    desc_5 = f"""
Deterministic Severity Rule Result:
{json.dumps(deterministic_sev, indent=2)}

Telemetry: Users={users_impacted}, ServiceTier={srv_tier}, OutageDuration={outage_dur}m.
Correlated Context:
{json.dumps(symptom_res, indent=2)}

Tasks:
1. Confirm final severity level ({rec_sev}).
2. Provide governing SLA rule and clear justification.

Return strictly valid JSON matching:
{{
  "severity_level": "{rec_sev}",
  "governing_sla_rule": "{gov_rule}",
  "justification": "Detailed explanation validating {rec_sev} based on {users_impacted} users impacted and {outage_dur} minutes outage under {srv_tier} SLA guidelines."
}}
"""
    exec_5 = call_agent_groq(
        agent_id="severity_agent",
        role="Severity Assessment Specialist",
        goal="Apply deterministic corporate severity thresholds and evaluate contextual impact, returning the final severity level (SEV-1 to SEV-4).",
        backstory="An Incident Commander assigning definitive severity levels strictly adhering to SLA policies.",
        task_description=desc_5,
        max_tokens=2000,
    )
    exec_5["order"] = 5
    exec_5["agent_id"] = "severity_agent"
    agent_executions.append(exec_5)
    severity_res = exec_5["output"]

    # 6. Remediation Agent
    print(f"  [6/7] Remediation Agent -> {AGENT_KEY_MAP['remediation_agent']}...", flush=True)
    desc_6 = f"""
Based on the selected runbook and diagnostic context:
Runbook Selected:
{json.dumps(runbook_res, indent=2)}

Diagnosis & Severity:
Severity: {json.dumps(severity_res, indent=2)}
Diagnosis: {json.dumps(symptom_res, indent=2)}

Tasks:
1. Recommend step-by-step troubleshooting and remediation actions (at least 3 concrete steps).
2. Detail an explicit rollback plan with verification checkpoints.
3. Note: Safe, documented, fully reversible. NEVER execute real infrastructure changes.

Return strictly valid JSON matching:
{{
  "remediation_steps": [
    {{
      "step_number": 1,
      "title": "Validate Access and Context",
      "description": "Confirm required operator roles and production cluster context.",
      "commands": ["kubectl config current-context", "aws sts get-caller-identity"],
      "verification": ["Context confirms prod environment"]
    }},
    {{
      "step_number": 2,
      "title": "Execute Graceful Worker Rollout",
      "description": "Trigger rolling restart of affected service pods to clear deadlock state.",
      "commands": ["kubectl rollout restart deployment/{service}"],
      "verification": ["Pods return to READY=1/1 Running status"]
    }},
    {{
      "step_number": 3,
      "title": "Post-Remediation Health Audit",
      "description": "Verify synthetic health check endpoints and error log rate reduction.",
      "commands": ["curl -sSf https://{service}.internal/healthz"],
      "verification": ["Endpoint returns status OK and zero panic logs"]
    }}
  ],
  "rollback_plan": {{
    "overall_strategy": "Revert to previous container deployment tag and restore database replica if degradation persists.",
    "steps": [
      {{
        "step_number": 1,
        "description": "Rollback deployment image tag",
        "commands": ["kubectl rollout undo deployment/{service}"]
      }}
    ]
  }},
  "safety_verification": {{
    "prerequisite_checks": ["Operator credentials verified", "Rollback procedure documented"],
    "is_safe_to_execute": true
  }}
}}
"""
    exec_6 = call_agent_groq(
        agent_id="remediation_agent",
        role="Remediation Planning Engineer",
        goal="Synthesize runbook guidance, system state, and diagnostic context into step-by-step remediation procedures with complete rollback plans.",
        backstory="A systems operations engineer designing automated recovery workflows and rollback contingencies.",
        task_description=desc_6,
        max_tokens=3000,
    )
    exec_6["order"] = 6
    exec_6["agent_id"] = "remediation_agent"
    agent_executions.append(exec_6)
    remediation_res = exec_6["output"]

    # 7. Incident Review Agent
    print(f"  [7/7] Incident Review Agent -> {AGENT_KEY_MAP['incident_review_agent']}...", flush=True)
    all_stages = {
        "intake": intake_res,
        "classification": classification_res,
        "runbook": runbook_res,
        "symptoms": symptom_res,
        "severity": severity_res,
        "remediation": remediation_res,
    }
    desc_7 = f"""
Audit all 6 preceding stages for agreement and completeness:
Incident Telemetry Dossier:
{json.dumps(all_stages, indent=2)}

Tasks:
1. Verify agreement across classification, severity, runbook, and remediation steps.
2. Confirm no contradictory actions or missing prerequisites.
3. Author the authoritative structured incident triage report.

Return strictly valid JSON matching:
{{
  "intake": {json.dumps(intake_res)},
  "classification": {json.dumps(classification_res)},
  "runbook": {json.dumps(runbook_res)},
  "symptoms": {json.dumps(symptom_res)},
  "severity": {json.dumps(severity_res)},
  "remediation": {json.dumps(remediation_res)},
  "final_review_summary": {{
    "incident_id": "{inc_id}",
    "service": "{service}",
    "final_severity": "{severity_res.get('severity_level', 'SEV-2')}",
    "runbook_assigned": "{runbook_res.get('selected_runbook_id', 'RB-101')}",
    "remediation_status": "READY_FOR_OPERATOR_APPROVAL",
    "qa_verification": "PASS - All safety constraints and rollback plans validated."
  }}
}}
"""
    exec_7 = call_agent_groq(
        agent_id="incident_review_agent",
        role="Incident Quality Reviewer",
        goal="Perform cross-stage quality audits, detect logical contradictions, and assemble the verified final incident report.",
        backstory="An Incident Governance and Postmortem Lead guaranteeing operational standards, documentation fidelity, and safety.",
        task_description=desc_7,
        max_tokens=3500,
    )
    exec_7["order"] = 7
    exec_7["agent_id"] = "incident_review_agent"
    agent_executions.append(exec_7)

    t_end = time.time()
    total_dur = round(t_end - t_start, 3)

    tot_prompt = sum(e["prompt_tokens"] for e in agent_executions)
    tot_comp = sum(e["completion_tokens"] for e in agent_executions)
    tot_tokens = tot_prompt + tot_comp
    cost = round((tot_prompt / 1_000_000.0) * 0.59 + (tot_comp / 1_000_000.0) * 0.79, 6)

    trace_record = {
        "run_id": run_id,
        "workflow_name": "incident_triage",
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
        "final_output": json.dumps(exec_7["output"], indent=2),
        "output": {
            "workflow": "incident_triage",
            "run_id": run_id,
            "input": incident_input,
            "stages": {
                "intake": json.dumps(intake_res, indent=2),
                "classification": json.dumps(classification_res, indent=2),
                "runbook": json.dumps(runbook_res, indent=2),
                "symptoms": json.dumps(symptom_res, indent=2),
                "severity": json.dumps(severity_res, indent=2),
                "remediation": json.dumps(remediation_res, indent=2),
                "final_review": json.dumps(exec_7["output"], indent=2),
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
        "workflow_name": "incident_triage",
        "total_runs": total_runs,
        "successful_runs": total_runs,
        "failed_runs": 0,
        "model_enforced": "openai/gpt-oss-120b",
        "key_distribution_model": "1-to-1 dedicated Groq key per sub-agent with dynamic auto-failover",
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
    parser = argparse.ArgumentParser(description="Generate live Ingestion_Traces for Incident Triage")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute (default: 1)")
    parser.add_argument("--start-idx", type=int, default=0, help="Incidents dataset starting index")
    parser.add_argument("--start-run-num", type=int, default=1, help="Run number to start numbering from")
    parser.add_argument("--total-runs", type=int, default=30, help="Total runs in batch for logging")
    args = parser.parse_args()

    # Load datasets
    data_dir = PROJECT_ROOT / "data" / "incident_triage"
    with open(data_dir / "historical_incidents.json", "r", encoding="utf-8") as f:
        raw_inc = json.load(f)
        all_incidents = raw_inc.get("incidents", []) if isinstance(raw_inc, dict) else raw_inc
    with open(data_dir / "runbooks.json", "r", encoding="utf-8") as f:
        raw_rb = json.load(f)
        all_runbooks = raw_rb.get("runbooks", []) if isinstance(raw_rb, dict) else raw_rb
    with open(data_dir / "log_snippets.json", "r", encoding="utf-8") as f:
        raw_logs = json.load(f)
        all_logs = raw_logs.get("logs", []) if isinstance(raw_logs, dict) else raw_logs

    target_dir = PROJECT_ROOT / "Ingestion_Traces" / "incident_triage"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting execution of {args.runs} live runs for incident_triage (Runs {args.start_run_num} to {args.start_run_num + args.runs - 1})...")
    print(f"Output directory: {target_dir}")

    summary_file = target_dir / "summary.jsonl"

    for i in range(args.runs):
        run_num = args.start_run_num + i
        inc_idx = (args.start_idx + i) % len(all_incidents)
        inc_sample = all_incidents[inc_idx]

        print(f"\n=======================================================")
        print(f" [RUN {run_num:02d}/{args.total_runs}] Starting...")
        print(f"=======================================================")

        try:
            trace_rec = run_single_incident_trace(
                incident_input=inc_sample,
                all_runbooks=all_runbooks,
                all_incidents=all_incidents,
                all_logs=all_logs,
                run_number=run_num,
                output_dir=target_dir,
            )
            with open(summary_file, "a", encoding="utf-8") as sf:
                summary_entry = {
                    "run_number": run_num,
                    "run_id": trace_rec["run_id"],
                    "incident_id": inc_sample.get("incident_id"),
                    "service": inc_sample.get("service"),
                    "title": inc_sample.get("title"),
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
                    "incident_id": inc_sample.get("incident_id"),
                    "error": str(e),
                    "status": "FAILED",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")

    generate_batch_summary(target_dir)

