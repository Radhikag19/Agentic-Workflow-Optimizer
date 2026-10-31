"""
API Key Manager for Dedicated Workflow and Per-Agent Groq Keys.

MANDATORY RULES:
1. Dedicated Groq API keys:
   - Workflows 1 to 7: Assigned dedicated workflow keys (GROQ_API_KEY_1 through GROQ_API_KEY_7).
   - Workflows 8 to 10: Supported with dedicated per-agent key assignments across GROQ_API_KEY_8 through GROQ_API_KEY_14.
2. Secret API keys must NEVER be exposed in logs, traces, or console outputs.
   Only safe identifiers (e.g. 'GROQ_API_KEY_1 (masked: gsk_***...PW)') are returned for tracing.
3. Fail clearly and loudly if a required key is missing. No silent fallbacks or mixing keys.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

# Mapping of Workflow IDs to their environment variable names and fallback aliases
WORKFLOW_KEY_ENV_MAP: Dict[str, Tuple[str, ...]] = {
    "invoice_processing": ("GROQ_API_KEY_1", "GROQ_API_KEY_INVOICE_PROCESSING"),
    "customer_support": ("GROQ_API_KEY_2", "GROQ_API_KEY_CUSTOMER_SUPPORT"),
    "procurement_supplier_selection": ("GROQ_API_KEY_3", "GROQ_API_KEY_PROCUREMENT"),
    "sales_lead_qualification": ("GROQ_API_KEY_4", "GROQ_API_KEY_SALES_LEAD"),
    "recruitment_screening": ("GROQ_API_KEY_5", "GROQ_API_KEY_RECRUITMENT"),
    "inventory_replenishment": ("GROQ_API_KEY_6", "GROQ_API_KEY_INVENTORY"),
    "contract_review": ("GROQ_API_KEY_7", "GROQ_API_KEY_CONTRACT_REVIEW"),
    "incident_triage": ("GROQ_API_KEY_8", "GROQ_API_KEY_INCIDENT_TRIAGE"),
    "expense_reimbursement": ("GROQ_API_KEY_9", "GROQ_API_KEY_EXPENSE_REIMBURSEMENT"),
    "market_research": ("GROQ_API_KEY_10", "GROQ_API_KEY_MARKET_RESEARCH"),
}

# Exactly 78 baseline agents across all 10 workflows
AGENT_WORKFLOW_MAP: Dict[str, str] = {
    # Workflow 1: Invoice Processing (8 agents)
    "invoice_extraction_agent": "invoice_processing",
    "invoice_validation_agent": "invoice_processing",
    "po_matching_agent": "invoice_processing",
    "policy_compliance_agent": "invoice_processing",
    "duplicate_detection_agent": "invoice_processing",
    "exception_analysis_agent": "invoice_processing",
    "approval_recommendation_agent": "invoice_processing",
    "final_review_agent_invoice": "invoice_processing",

    # Workflow 2: Customer Support (6 agents)
    "ticket_classification_agent": "customer_support",
    "policy_retrieval_agent": "customer_support",
    "issue_investigation_agent": "customer_support",
    "resolution_agent": "customer_support",
    "response_drafting_agent": "customer_support",
    "resolution_verification_agent": "customer_support",

    # Workflow 3: Procurement & Supplier Selection (9 agents)
    "quotation_analysis_agent": "procurement_supplier_selection",
    "supplier_profile_agent": "procurement_supplier_selection",
    "delivery_analysis_agent": "procurement_supplier_selection",
    "supplier_risk_agent": "procurement_supplier_selection",
    "contract_analysis_agent": "procurement_supplier_selection",
    "cost_comparison_agent": "procurement_supplier_selection",
    "weighted_evaluation_agent": "procurement_supplier_selection",
    "selection_recommendation_agent": "procurement_supplier_selection",
    "final_review_agent_procurement": "procurement_supplier_selection",

    # Workflow 4: Sales Lead Qualification (7 agents)
    "lead_enrichment_agent": "sales_lead_qualification",
    "lead_fit_agent": "sales_lead_qualification",
    "intent_analysis_agent": "sales_lead_qualification",
    "lead_scoring_agent": "sales_lead_qualification",
    "opportunity_assessment_agent": "sales_lead_qualification",
    "next_action_agent": "sales_lead_qualification",
    "final_review_agent_sales": "sales_lead_qualification",

    # Workflow 5: Recruitment and Resume Screening (8 agents)
    "resume_intake_agent": "recruitment_screening",
    "qualification_agent": "recruitment_screening",
    "job_criteria_agent": "recruitment_screening",
    "evidence_mapping_agent": "recruitment_screening",
    "experience_assessment_agent": "recruitment_screening",
    "missing_evidence_agent": "recruitment_screening",
    "screening_review_agent": "recruitment_screening",
    "recruiter_summary_agent": "recruitment_screening",

    # Workflow 6: Inventory Replenishment Planning (6 agents)
    "inventory_intake_agent": "inventory_replenishment",
    "stock_policy_agent": "inventory_replenishment",
    "demand_forecast_agent": "inventory_replenishment",
    "supplier_agent": "inventory_replenishment",
    "replenishment_planning_agent": "inventory_replenishment",
    "inventory_review_agent": "inventory_replenishment",

    # Workflow 7: Contract Review and Compliance (10 agents)
    "contract_intake_agent": "contract_review",
    "clause_extraction_agent": "contract_review",
    "policy_interpretation_agent": "contract_review",
    "clause_mapping_agent": "contract_review",
    "compliance_deviation_agent": "contract_review",
    "risk_ambiguity_agent": "contract_review",
    "cross_clause_consistency_agent": "contract_review",
    "evidence_validation_agent": "contract_review",
    "contract_independent_review_agent": "contract_review",
    "final_compliance_report_agent": "contract_review",

    # Workflow 8: IT Incident Triage (7 agents)
    "intake_agent": "incident_triage",
    "classification_agent": "incident_triage",
    "runbook_agent": "incident_triage",
    "symptom_agent": "incident_triage",
    "severity_agent": "incident_triage",
    "remediation_agent": "incident_triage",
    "incident_review_agent": "incident_triage",

    # Workflow 9: Employee Expense Reimbursement (8 agents)
    "receipt_agent": "expense_reimbursement",
    "normalize_agent": "expense_reimbursement",
    "category_agent": "expense_reimbursement",
    "amount_agent": "expense_reimbursement",
    "policy_agent": "expense_reimbursement",
    "duplicate_agent": "expense_reimbursement",
    "exception_agent": "expense_reimbursement",
    "expense_review_agent": "expense_reimbursement",

    # Workflow 10: Market & Competitor Research (9 agents)
    "planning_agent": "market_research",
    "profile_agent": "market_research",
    "product_agent": "market_research",
    "pricing_agent": "market_research",
    "feedback_agent": "market_research",
    "evidence_agent": "market_research",
    "comparison_agent": "market_research",
    "synthesis_agent": "market_research",
    "fact_check_agent": "market_research",
}


def mask_key(raw_key: str) -> str:
    """Safely mask a secret key for logging and tracing without exposing credentials."""
    if not raw_key:
        return "[NOT_SET]"
    stripped = raw_key.strip()
    if len(stripped) <= 8:
        return "***"
    return f"{stripped[:4]}...{stripped[-4:]}"


def get_workflow_for_agent(agent_id: str) -> str:
    """Return the workflow ID for a given agent ID."""
    if agent_id not in AGENT_WORKFLOW_MAP:
        raise KeyError(
            f"Unknown agent ID '{agent_id}'. Must be one of the registered baseline agents."
        )
    return AGENT_WORKFLOW_MAP[agent_id]


def get_key_for_workflow(workflow_id: str) -> Tuple[str, str]:
    """
    Retrieve the configured Groq API key for a workflow.
    Returns:
        (raw_key, env_var_name)
    Raises:
        ValueError if no key is configured.
    """
    if workflow_id not in WORKFLOW_KEY_ENV_MAP:
        raise KeyError(f"Unknown workflow ID '{workflow_id}'. Must be one of {list(WORKFLOW_KEY_ENV_MAP.keys())}")

    env_vars = WORKFLOW_KEY_ENV_MAP[workflow_id]
    for env_var in env_vars:
        val = os.getenv(env_var, "").strip()
        if val:
            return val, env_var

    primary_var = env_vars[0]
    raise ValueError(
        f"Missing dedicated Groq API key for workflow '{workflow_id}'. "
        f"Please set environment variable '{primary_var}' in your .env file."
    )


def get_key_for_agent(agent_id: str) -> Tuple[str, str, str]:
    """
    Retrieve the dedicated Groq API key for an agent based on its workflow or per-agent assignment.
    Returns:
        (raw_key, env_var_name, safe_masked_id)
    """
    from common.settings import AGENT_KEY_ASSIGNMENTS, GROQ_KEYS_POOL

    # If agent has a specific per-agent key assigned (Workflows 8-10)
    if agent_id in AGENT_KEY_ASSIGNMENTS:
        env_var_name = AGENT_KEY_ASSIGNMENTS[agent_id]
        raw_key = GROQ_KEYS_POOL.get(env_var_name) or os.getenv(env_var_name, "").strip()
        if raw_key:
            safe_masked = f"{env_var_name} ({mask_key(raw_key)})"
            return raw_key, env_var_name, safe_masked

    wf_id = get_workflow_for_agent(agent_id)
    raw_key, env_var_name = get_key_for_workflow(wf_id)
    safe_masked = f"{env_var_name} ({mask_key(raw_key)})"
    return raw_key, env_var_name, safe_masked


def validate_all_workflow_keys() -> Dict[str, str]:
    """
    Validate that all workflows have configured Groq API keys.
    Returns a dictionary of {workflow_id: safe_masked_identifier}.
    """
    results = {}
    missing = []
    for wf_id, env_vars in WORKFLOW_KEY_ENV_MAP.items():
        found = False
        for ev in env_vars:
            val = os.getenv(ev, "").strip()
            if val:
                results[wf_id] = f"{ev} ({mask_key(val)})"
                found = True
                break
        if not found:
            missing.append(f"Workflow '{wf_id}' (expected {env_vars[0]})")

    if missing:
        raise ValueError(
            f"API Key Validation Error: Missing required Groq API keys for: {', '.join(missing)}"
        )
    return results
