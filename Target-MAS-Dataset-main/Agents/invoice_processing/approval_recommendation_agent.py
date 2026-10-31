"""
Agent 7: Approval Recommendation Agent
Evaluates consolidated evidence and formulates a structured approval outcome.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "approval_recommendation_agent"

def deterministic_recommendation_rule(consolidated_evidence: dict) -> dict:
    """Deterministic baseline decision rule engine."""
    val_status = consolidated_evidence.get("validation", {}).get("validation_status", "PASSED")
    dup_status = consolidated_evidence.get("duplicate", {}).get("duplicate_status", "NONE")
    po_status = consolidated_evidence.get("po_match", {}).get("po_match_status", "EXACT_MATCH")
    is_compliant = consolidated_evidence.get("policy", {}).get("is_compliant", True)
    exceptions = consolidated_evidence.get("exceptions", {})

    # Rejection triggers
    if val_status == "FAILED" or dup_status == "EXACT_DUPLICATE":
        return {
            "recommended_outcome": "rejected",
            "primary_reason": "Arithmetic calculation failure or confirmed exact duplicate invoice submission."
        }
    if not is_compliant:
        critical_violations = [v for v in consolidated_evidence.get("policy", {}).get("violations", []) if v.get("severity") == "critical"]
        if critical_violations:
            return {
                "recommended_outcome": "rejected",
                "primary_reason": "Critical policy violation (e.g., suspended supplier or unapproved vendor)."
            }
        return {
            "recommended_outcome": "needs_review",
            "primary_reason": "Policy threshold flag requiring manager review."
        }

    # Review triggers
    if po_status in ["PO_NOT_FOUND", "MISSING_PO_REFERENCE", "DISCREPANCY_DETECTED"] or dup_status == "POTENTIAL_DUPLICATE":
        return {
            "recommended_outcome": "needs_review",
            "primary_reason": "Purchase order mismatch, missing PO, or potential duplicate warning."
        }

    return {
        "recommended_outcome": "recommended_for_approval",
        "primary_reason": "All validation, 3-way matching, compliance checks, and duplicate scans passed successfully."
    }

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Accounts Payable Decision Authority",
        goal="Determine final payment recommendation (recommended_for_approval, rejected, or needs_review) based on factual evidence.",
        backstory="An executive accounts payable controller applying strict fiduciary oversight to payment approvals.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, consolidated_evidence: dict) -> Task:
    rule_guidance = deterministic_recommendation_rule(consolidated_evidence)
    description = (
        f"Formulate a structured approval recommendation for this invoice.\n"
        f"Consolidated Evidence from all preceding stages:\n{json.dumps(consolidated_evidence, indent=2)}\n\n"
        f"Deterministic Decision Guidance:\n{json.dumps(rule_guidance, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Select outcome: 'recommended_for_approval', 'rejected', or 'needs_review'.\n"
        f"2. Provide explicit justification referencing PO match status, policy IDs, math audit, and duplicate findings.\n"
        f"3. Note: NEVER execute real financial transactions.\n"
        f"Return strictly valid JSON with keys: 'recommendation' ('recommended_for_approval', 'rejected', or 'needs_review'), 'justification', and 'conditions'."
    )
    expected_output = "Strictly valid JSON containing recommendation, justification, and conditions."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
