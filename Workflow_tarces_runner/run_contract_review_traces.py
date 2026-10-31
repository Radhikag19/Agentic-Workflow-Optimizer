"""
Dedicated Live Trace Generator for Contract Review & Compliance.

Enforces:
1. 1-to-1 Groq API Key Distribution across all 10 sub-agents (Keys 1..10).
2. Live, non-hallucinated execution using Groq API with openai/gpt-oss-120b.
3. Reference trace format matching runs/invoice_processing/live_20261023_070056_invoic.json.
4. Outputs persisted directly in Ingestion_Traces/contract_review/ (no agent-wise subfolders).
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from groq import Groq

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from common.validation import (
    extract_json_from_text,
    ContractIntakeOutput,
    ClauseExtractionOutput,
    PolicyInterpretationOutput,
    ClauseMappingOutput,
    ComplianceDeviationOutput,
    RiskAmbiguityOutput,
    CrossClauseConsistencyOutput,
    EvidenceValidationOutput,
    IndependentReviewOutput,
    ContractFinalReportOutput,
)

# 1-to-1 Sub-Agent to Groq API Key Mapping
AGENT_KEY_MAP = {
    "contract_intake_agent": "GROQ_API_KEY_1",
    "clause_extraction_agent": "GROQ_API_KEY_2",
    "policy_interpretation_agent": "GROQ_API_KEY_3",
    "clause_mapping_agent": "GROQ_API_KEY_4",
    "compliance_deviation_agent": "GROQ_API_KEY_5",
    "risk_ambiguity_agent": "GROQ_API_KEY_6",
    "cross_clause_consistency_agent": "GROQ_API_KEY_7",
    "evidence_validation_agent": "GROQ_API_KEY_8",
    "contract_independent_review_agent": "GROQ_API_KEY_9",
    "final_compliance_report_agent": "GROQ_API_KEY_10",
}

def mask_key(raw_key: str) -> str:
    if not raw_key:
        return "[NOT_SET]"
    s = raw_key.strip()
    return f"{s[:6]}...{s[-4:]}" if len(s) > 10 else "***"

def call_agent_groq(
    agent_id: str,
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    output_model: Any = None,
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
            # If model returned reasoning and empty content, check reasoning or choices
            if not raw_content and hasattr(resp.choices[0].message, "reasoning"):
                raw_content = resp.choices[0].message.reasoning or ""

            try:
                parsed = extract_json_from_text(raw_content)
            except Exception as parse_exc:
                # Try soft repair if truncated closing brackets
                import re
                s = raw_content.strip()
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
                if match:
                    s = match.group(1).strip()
                # If closing braces/brackets are missing at end
                open_braces = s.count("{") - s.count("}")
                open_brackets = s.count("[") - s.count("]")
                if open_braces > 0 or open_brackets > 0:
                    s_repaired = s.rstrip(", \n\r\t") + ("]" * open_brackets) + ("}" * open_braces)
                    try:
                        parsed = json.loads(s_repaired)
                    except Exception:
                        raise parse_exc
                else:
                    raise parse_exc
            
            # Validate with Pydantic model if provided
            if output_model:
                try:
                    val_obj = output_model.model_validate(parsed)
                    parsed = val_obj.model_dump()
                except Exception:
                    pass

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


def run_single_contract_trace(
    contract_data: Dict[str, Any],
    policy_data: Dict[str, Any],
    run_number: int = 1,
    output_dir: Path = Path("Ingestion_Traces/contract_review"),
) -> Dict[str, Any]:
    """Execute all 10 agents sequentially for one contract and persist the trace."""
    run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_trace{run_number:02d}"
    t_start = time.time()
    agent_executions = []

    c_id = contract_data.get("contract_id", "CTR-UNKNOWN")
    c_title = contract_data.get("contract_title", "Commercial Agreement")
    c_type = contract_data.get("contract_type", "Standard Agreement")
    c_ver = contract_data.get("contract_version", "1.0")
    p_id = policy_data.get("policy_id", "POL-UNKNOWN")
    p_ver = policy_data.get("version", "1.0")
    clauses = contract_data.get("clauses", [])

    print(f"\n--- [Run {run_number:02d}] Executing Contract Review for {c_id} ({c_title}) ---")

    # 1. Contract Intake Agent (Key 1)
    print("  [1/10] Contract Intake Agent -> GROQ_API_KEY_1...")
    desc_1 = f"""
Contract Record:
- Contract ID: {c_id}
- Title: {c_title}
- Type: {c_type}
- Version: {c_ver}
- Parties: {contract_data.get('party_a')} and {contract_data.get('party_b')}
- Effective Date: {contract_data.get('effective_date')}
- Expiration Date: {contract_data.get('expiration_date')}
- Clause Count: {len(clauses)}

Target Policy Record:
- Policy ID: {p_id}
- Policy Version: {p_ver}

Validate document structure and completeness. Return structured JSON with document_id, contract_id, contract_title, contract_type, contract_version, policy_id, policy_version, document_complete, section_count, source_metadata, intake_warnings.
"""
    exec_1 = call_agent_groq(
        agent_id="contract_intake_agent",
        role="Contract Intake and Ingestion Specialist",
        goal="Ingest and validate structure, metadata, versioning, and completeness.",
        backstory="Legal operations specialist cataloguing contracts without injecting unstated terms.",
        task_description=desc_1,
        output_model=ContractIntakeOutput,
    )
    exec_1["order"] = 1
    exec_1["agent_id"] = "contract_intake_agent"
    agent_executions.append(exec_1)

    # 2. Clause Extraction Agent (Key 2)
    print("  [2/10] Clause Extraction Agent -> GROQ_API_KEY_2...")
    desc_2 = f"""
Contract ID: {c_id}
Contract Text Clauses:
{json.dumps(clauses, indent=2)}

You are the Contract Clause Extraction Specialist.
Systematically extract every discrete clause from the Contract Text Clauses into a JSON object matching:
{{
  "contract_id": "{c_id}",
  "clauses": [
    {{
      "clause_id": "CLS-0001-PAY",
      "clause_type": "payment_terms",
      "clause_title": "Section 3: Invoicing and Payment Terms",
      "clause_text": "...",
      "section_reference": "Section 3.1"
    }}
  ],
  "absent_clause_types": [],
  "extraction_notes": ["Document extracted once for downstream policy comparison"]
}}
Include all clauses from the input in the 'clauses' array. Output JSON only.
"""
    exec_2 = call_agent_groq(
        agent_id="clause_extraction_agent",
        role="Contract Clause Extraction Specialist",
        goal="Extract discrete legal clauses tagging standardized types and stable identifiers.",
        backstory="Legal ops specialist inventorying contractual provisions without modifying text.",
        task_description=desc_2,
        output_model=ClauseExtractionOutput,
    )
    exec_2["order"] = 2
    exec_2["agent_id"] = "clause_extraction_agent"
    agent_executions.append(exec_2)

    # 3. Policy Interpretation Agent (Key 3)
    print("  [3/10] Policy Interpretation Agent -> GROQ_API_KEY_3...")
    desc_3 = f"""
Policy Data:
{json.dumps(policy_data, indent=2)}

Deconstruct compliance policy into discrete requirements.
Return JSON with policy_id, policy_version, policy_name, mandatory_requirements, recommended_requirements, ambiguities_in_policy.
"""
    exec_3 = call_agent_groq(
        agent_id="policy_interpretation_agent",
        role="Policy Interpretation Specialist",
        goal="Deconstruct compliance policy into structured discrete requirements with thresholds.",
        backstory="Compliance analyst decomposing internal regulations into operational criteria.",
        task_description=desc_3,
        output_model=PolicyInterpretationOutput,
    )
    exec_3["order"] = 3
    exec_3["agent_id"] = "policy_interpretation_agent"
    agent_executions.append(exec_3)

    # 4. Clause Mapping Agent (Key 4)
    print("  [4/10] Clause Mapping Agent -> GROQ_API_KEY_4...")
    desc_4 = f"""
Contract ID: {c_id}
Extracted Clauses:
{json.dumps(exec_2['output'].get('clauses', []), indent=2)}

Policy Requirements:
{json.dumps(exec_3['output'].get('mandatory_requirements', []), indent=2)}

Map each extracted clause to relevant policy requirement. Classify mapping_status as 'satisfied', 'partially_satisfied', 'conflicting', or 'unaddressed'.
Return JSON with contract_id, policy_id, mappings, coverage_percentage.
"""
    exec_4 = call_agent_groq(
        agent_id="clause_mapping_agent",
        role="Clause-to-Policy Mapping Specialist",
        goal="Align extracted clauses against structured corporate policy requirements.",
        backstory="Contract auditor establishing precise bidirectional cross-references.",
        task_description=desc_4,
        output_model=ClauseMappingOutput,
    )
    exec_4["order"] = 4
    exec_4["agent_id"] = "clause_mapping_agent"
    agent_executions.append(exec_4)

    # 5. Compliance Deviation Agent (Key 5)
    print("  [5/10] Compliance Deviation Agent -> GROQ_API_KEY_5...")
    desc_5 = f"""
Contract ID: {c_id}
Clause Mappings:
{json.dumps(exec_4['output'].get('mappings', []), indent=2)}

Policy Requirements:
{json.dumps(exec_3['output'].get('mandatory_requirements', []), indent=2)}

Identify specific compliance deviations. Grade severity ('low', 'medium', 'high', 'critical'). Provide supporting_evidence and recommended_remediation.
Return JSON with contract_id, deviations, has_critical_deviations.
"""
    exec_5 = call_agent_groq(
        agent_id="compliance_deviation_agent",
        role="Contract Compliance Deviation Evaluator",
        goal="Identify and grade specific deviations where contractual terms fall short of policy.",
        backstory="Legal compliance auditor grading contract terms against organizational thresholds.",
        task_description=desc_5,
        output_model=ComplianceDeviationOutput,
    )
    exec_5["order"] = 5
    exec_5["agent_id"] = "compliance_deviation_agent"
    agent_executions.append(exec_5)

    # 6. Risk and Ambiguity Agent (Key 6)
    print("  [6/10] Risk and Ambiguity Agent -> GROQ_API_KEY_6...")
    desc_6 = f"""
Contract ID: {c_id}
Clauses:
{json.dumps(exec_2['output'].get('clauses', []), indent=2)}

Detect vague wording, undefined timelines, one-sided covenants, and latent operational risks.
Return JSON with contract_id, risk_findings (finding_id, clause_id, risk_category, ambiguity_description, severity, human_interpretation_needed).
"""
    exec_6 = call_agent_groq(
        agent_id="risk_ambiguity_agent",
        role="Contract Risk and Ambiguity Analyst",
        goal="Detect vague wording, undefined timelines, one-sided covenants, and latent risks.",
        backstory="Risk auditor identifying contract language that could lead to commercial exposure.",
        task_description=desc_6,
        output_model=RiskAmbiguityOutput,
    )
    exec_6["order"] = 6
    exec_6["agent_id"] = "risk_ambiguity_agent"
    agent_executions.append(exec_6)

    # 7. Cross-Clause Consistency Agent (Key 7)
    print("  [7/10] Cross-Clause Consistency Agent -> GROQ_API_KEY_7...")
    desc_7 = f"""
Contract ID: {c_id}
Contract Metadata: {json.dumps(contract_data, indent=2)}
Extracted Clauses: {json.dumps(exec_2['output'].get('clauses', []), indent=2)}

Identify internal contradictions or chronological incompatibilities across separate clauses.
Return JSON with contract_id, inconsistencies, is_internally_consistent.
"""
    exec_7 = call_agent_groq(
        agent_id="cross_clause_consistency_agent",
        role="Contract Inter-Clause Consistency Auditor",
        goal="Identify mutual incompatibilities and chronological contradictions across separate clauses.",
        backstory="Legal contract auditor verifying multi-section internal harmony.",
        task_description=desc_7,
        output_model=CrossClauseConsistencyOutput,
    )
    exec_7["order"] = 7
    exec_7["agent_id"] = "cross_clause_consistency_agent"
    agent_executions.append(exec_7)

    # 8. Evidence Validation Agent (Key 8)
    print("  [8/10] Evidence Validation Agent -> GROQ_API_KEY_8...")
    desc_8 = f"""
Contract ID: {c_id}
Clauses: {json.dumps(exec_2['output'].get('clauses', []), indent=2)}
Deviations: {json.dumps(exec_5['output'].get('deviations', []), indent=2)}
Risks: {json.dumps(exec_6['output'].get('risk_findings', []), indent=2)}

Audit all findings to verify every citation references an actual verified clause.
Return JSON with contract_id, all_citations_valid, verified_finding_ids, invalid_or_unsupported_findings, evidence_notes.
"""
    exec_8 = call_agent_groq(
        agent_id="evidence_validation_agent",
        role="Legal Evidence and Citation Validator",
        goal="Confirm every finding cites a real verified clause ID and excerpt directly supports the claim.",
        backstory="Evidence auditor ensuring strict citation accuracy without unreferenced assertions.",
        task_description=desc_8,
        output_model=EvidenceValidationOutput,
    )
    exec_8["order"] = 8
    exec_8["agent_id"] = "evidence_validation_agent"
    agent_executions.append(exec_8)

    # 9. Independent Review Agent (Key 9)
    print("  [9/10] Independent Review Agent -> GROQ_API_KEY_9...")
    desc_9 = f"""
Contract ID: {c_id}
Deviations: {json.dumps(exec_5['output'].get('deviations', []), indent=2)}
Risks: {json.dumps(exec_6['output'].get('risk_findings', []), indent=2)}
Validation: {json.dumps(exec_8['output'], indent=2)}

Perform second-pass review: deduplicate redundant issues, calibrate severity ratings, highlight human review items.
Return JSON with contract_id, calibrated_deviations, calibrated_risks, review_notes.
"""
    exec_9 = call_agent_groq(
        agent_id="contract_independent_review_agent",
        role="Senior Contract Compliance Reviewer",
        goal="Deduplicate redundant issues, calibrate severity ratings, and highlight human review items.",
        backstory="Senior compliance counsel providing objective second-pass calibration.",
        task_description=desc_9,
        output_model=IndependentReviewOutput,
    )
    exec_9["order"] = 9
    exec_9["agent_id"] = "contract_independent_review_agent"
    agent_executions.append(exec_9)

    # 10. Final Compliance Report Agent (Key 10)
    print("  [10/10] Final Compliance Report Agent -> GROQ_API_KEY_10...")
    desc_10 = f"""
Contract ID: {c_id}
Title: {c_title}
Version: {c_ver}
Policy ID: {p_id}
Intake: {json.dumps(exec_1['output'], indent=2)}
Clauses: {json.dumps(exec_2['output'].get('clauses', []), indent=2)}
Mapping: {json.dumps(exec_4['output'], indent=2)}
Review Findings: {json.dumps(exec_9['output'], indent=2)}

Assemble the definitive contract compliance report.
Return JSON with run_id, contract_id, contract_version, policy_id, policy_version, contract_type, executive_summary, policy_coverage, clause_inventory, compliance_findings, risk_findings, cross_clause_inconsistencies, evidence_validation, unresolved_questions, recommended_actions, human_review_required, legal_disclaimer.
"""
    exec_10 = call_agent_groq(
        agent_id="final_compliance_report_agent",
        role="Lead Contract Compliance Report Author",
        goal="Assemble and author the definitive evidence-backed contract compliance report.",
        backstory="Senior report author synthesizing technical legal findings with disclaimers.",
        task_description=desc_10,
        output_model=ContractFinalReportOutput,
        max_tokens=4096,
    )
    exec_10["order"] = 10
    exec_10["agent_id"] = "final_compliance_report_agent"
    agent_executions.append(exec_10)

    t_end = time.time()
    total_dur = round(t_end - t_start, 3)

    tot_prompt = sum(e["prompt_tokens"] for e in agent_executions)
    tot_comp = sum(e["completion_tokens"] for e in agent_executions)
    tot_tokens = tot_prompt + tot_comp
    cost = round((tot_prompt / 1_000_000.0) * 0.59 + (tot_comp / 1_000_000.0) * 0.79, 6)

    # Reference structure matching runs/invoice_processing/live_20261023_070056_invoic.json
    trace_record = {
        "run_id": run_id,
        "workflow_name": "contract_review",
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
        "final_output": json.dumps(exec_10["output"], indent=2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save directly in Ingestion_Traces/contract_review/ (no agent-wise subfolders)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_file = output_dir / f"{run_id}.json"
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(trace_record, f, indent=2, ensure_ascii=False)

    print(f"--> Saved trace to: {trace_file}")
    print(f"--> Completed in {total_dur}s | Total Tokens: {tot_tokens:,} | Cost: ${cost:.6f}")
    return trace_record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate live Ingestion_Traces for Contract Review")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute (default: 1)")
    parser.add_argument("--start-idx", type=int, default=0, help="Contract dataset starting index")
    parser.add_argument("--start-run-num", type=int, default=1, help="Run number to start numbering from")
    parser.add_argument("--total-runs", type=int, default=30, help="Total runs in batch for logging")
    args = parser.parse_args()

    # Load contracts and policies
    with open(PROJECT_ROOT / "data/contract_review/contracts.json", "r", encoding="utf-8") as f:
        all_contracts = json.load(f)
    with open(PROJECT_ROOT / "data/contract_review/policies.json", "r", encoding="utf-8") as f:
        all_policies = json.load(f)

    target_dir = PROJECT_ROOT / "Ingestion_Traces" / "contract_review"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting execution of {args.runs} live runs for contract_review (Runs {args.start_run_num} to {args.start_run_num + args.runs - 1})...")
    print(f"Output directory: {target_dir}")

    summary_file = target_dir / "summary.jsonl"

    for i in range(args.runs):
        run_num = args.start_run_num + i
        c_idx = (args.start_idx + i) % len(all_contracts)
        p_idx = (args.start_run_num - 1 + i) % len(all_policies)
        c_sample = all_contracts[c_idx]
        p_sample = all_policies[p_idx]

        print(f"\n=======================================================")
        print(f" [RUN {run_num:02d}/{args.total_runs}] Starting...")
        print(f"=======================================================")

        try:
            trace_rec = run_single_contract_trace(
                contract_data=c_sample,
                policy_data=p_sample,
                run_number=run_num,
                output_dir=target_dir,
            )
            # Append to summary.jsonl
            with open(summary_file, "a", encoding="utf-8") as sf:
                summary_entry = {
                    "run_number": run_num,
                    "run_id": trace_rec["run_id"],
                    "contract_id": c_sample.get("contract_id"),
                    "contract_title": c_sample.get("contract_title"),
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
                    "contract_id": c_sample.get("contract_id"),
                    "error": str(e),
                    "status": "FAILED",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")

