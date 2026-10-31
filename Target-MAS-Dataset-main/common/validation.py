"""
Structured Data Validation and Pydantic Schemas for AWO Workflows.

Provides strict validation for inputs, intermediate agent outputs, and final deliverables.
Handles JSON extraction from agent text outputs and enforces schema consistency.
"""

import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError


class JSONParsingError(ValueError):
    """Raised when an agent output cannot be parsed into valid JSON."""
    pass


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from agent text response.
    Handles markdown code blocks (```json ... ```) and leading/trailing chatter.
    """
    if not text or not text.strip():
        raise JSONParsingError("Empty response cannot be parsed as JSON")

    cleaned = text.strip()

    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        # Try finding the first '{' and last '}'
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise JSONParsingError(f"Failed to parse JSON content: {exc}\nRaw snippet: {cleaned[:200]}") from exc


# =====================================================================
# Workflow 1: Recruitment Screening Models
# =====================================================================

class ResumeSection(BaseModel):
    section_name: str
    content_summary: str
    evidence_ids: List[str] = Field(default_factory=list)


class ResumeIntakeOutput(BaseModel):
    application_id: str
    candidate_id: str
    candidate_name: str
    contact_info_present: bool
    sections_identified: List[str]
    missing_sections: List[str] = Field(default_factory=list)
    raw_sections: Dict[str, Any] = Field(default_factory=dict)
    source_evidence_ids: List[str] = Field(default_factory=list)
    intake_notes: List[str] = Field(default_factory=list)


class QualificationItem(BaseModel):
    category: str  # technical_skill, education, certification, project, experience
    item_name: str
    proficiency_or_duration: Optional[str] = None
    supporting_excerpt: str
    evidence_id: str


class QualificationExtractionOutput(BaseModel):
    candidate_id: str
    technical_skills: List[QualificationItem] = Field(default_factory=list)
    education: List[QualificationItem] = Field(default_factory=list)
    certifications: List[QualificationItem] = Field(default_factory=list)
    experience_records: List[QualificationItem] = Field(default_factory=list)
    projects: List[QualificationItem] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)


class JobCriterion(BaseModel):
    criterion_id: str
    category: str
    requirement_type: str  # mandatory or preferred
    description: str
    is_ambiguous: bool = False
    ambiguity_notes: Optional[str] = None


class JobCriteriaOutput(BaseModel):
    job_id: str
    job_title: str
    mandatory_criteria: List[JobCriterion] = Field(default_factory=list)
    preferred_criteria: List[JobCriterion] = Field(default_factory=list)
    total_criteria_count: int = 0
    parsing_notes: List[str] = Field(default_factory=list)


class CriterionMapping(BaseModel):
    criterion_id: str
    requirement_type: str  # mandatory or preferred
    status: str  # supported, partially_supported, unsupported, unclear
    candidate_evidence: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    is_missing_qualification: bool = False
    is_missing_evidence_only: bool = False
    notes: str = ""


class EvidenceMappingOutput(BaseModel):
    candidate_id: str
    job_id: str
    mappings: List[CriterionMapping] = Field(default_factory=list)
    supported_count: int = 0
    unsupported_count: int = 0
    partially_supported_count: int = 0
    unclear_count: int = 0


class ExperienceAssessmentOutput(BaseModel):
    candidate_id: str
    relevance_summary: str
    strengths: List[str] = Field(default_factory=list)
    experience_gaps: List[str] = Field(default_factory=list)
    transferable_skills: List[str] = Field(default_factory=list)
    evidence_references: List[str] = Field(default_factory=list)


class InconsistencyIssue(BaseModel):
    issue_type: str  # date_gap, conflicting_title, unsupported_claim, ambiguous_scope
    description: str
    is_confirmed_inconsistency: bool
    requires_recruiter_clarification: bool
    evidence_ids: List[str] = Field(default_factory=list)


class MissingEvidenceOutput(BaseModel):
    candidate_id: str
    missing_evidence_items: List[str] = Field(default_factory=list)
    unreliable_requirements: List[str] = Field(default_factory=list)
    inconsistencies: List[InconsistencyIssue] = Field(default_factory=list)


class ScreeningReviewOutput(BaseModel):
    candidate_id: str
    review_passed: bool
    contradictions_detected: List[str] = Field(default_factory=list)
    traceability_verified: bool = True
    mandatory_criteria_met: bool
    issues_requiring_human_review: List[str] = Field(default_factory=list)


class RecruiterSummaryOutput(BaseModel):
    run_id: str
    application_id: str
    candidate_id: str
    candidate_name: str
    job_id: str
    job_title: str
    executive_candidate_overview: str
    criteria_matrix: List[CriterionMapping]
    key_strengths: List[str]
    evidence_gaps: List[str]
    identified_inconsistencies: List[str]
    recommended_follow_up_questions: List[str]
    deterministic_match_score: float
    score_calculation_method: str
    score_limitations: str
    human_review_required: bool = True
    non_autonomous_decision_statement: str = (
        "This is an automated screening assistant summary. It does NOT make autonomous hiring decisions."
    )
    validation_warnings: List[str] = Field(default_factory=list)


# =====================================================================
# Workflow 2: Inventory Replenishment Models
# =====================================================================

class InventoryIntakeOutput(BaseModel):
    request_id: str
    sku_id: str
    warehouse_id: str
    sku_name: str
    source_record_ids: List[str]
    raw_inventory_record: Dict[str, Any]
    raw_supplier_record: Dict[str, Any]
    raw_policy_record: Dict[str, Any]
    sales_history_days: int
    data_quality_issues: List[str] = Field(default_factory=list)


class StockPolicyOutput(BaseModel):
    sku_id: str
    warehouse_id: str
    current_stock: int
    reserved_stock: int
    incoming_stock: int
    usable_inventory: int  # Deterministic: current_stock - reserved_stock + incoming_stock
    reorder_threshold: int
    safety_stock: int
    replenishment_needed: bool
    policy_notes: List[str] = Field(default_factory=list)


class DemandForecastOutput(BaseModel):
    sku_id: str
    forecast_horizon_days: int
    historical_avg_daily_demand: float
    forecast_demand_units: float  # Deterministic: avg_daily * horizon
    volatility_metric: str  # low, medium, high, intermittent
    demand_spikes_detected: bool
    has_sufficient_history: bool
    source_record_ids: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class SupplierLeadTimeOutput(BaseModel):
    supplier_id: str
    lead_time_days: int
    minimum_order_quantity: int
    pack_size: int
    expected_arrival_date: str
    supplier_reliability_notes: List[str] = Field(default_factory=list)
    source_record_ids: List[str] = Field(default_factory=list)


class ReplenishmentPlanningOutput(BaseModel):
    sku_id: str
    base_replenishment_quantity: int
    recommended_order_quantity: int
    calculation_breakdown: str
    adjustment_reasoning: str
    stockout_risk: str  # none, low, moderate, severe
    infeasible_flag: bool = False


class InventoryFinalPlanOutput(BaseModel):
    run_id: str
    planning_date: str
    warehouse_id: str
    sku_id: str
    current_stock: int
    reserved_stock: int
    available_stock: int
    incoming_stock: int
    forecast_demand: float
    forecast_horizon: str
    supplier_id: str
    lead_time_days: int
    safety_stock: int
    reorder_threshold: int
    base_replenishment_quantity: int
    recommended_order_quantity: int
    expected_arrival_date: str
    stockout_risk: str
    reasoning: str
    source_record_ids: List[str]
    validation_warnings: List[str] = Field(default_factory=list)
    human_review_required: bool = True
    disclaimer: str = "Planning tool recommendation only. Does NOT execute real purchases."


# =====================================================================
# Workflow 3: Contract Review Models
# =====================================================================

class ContractIntakeOutput(BaseModel):
    document_id: str
    contract_id: str
    contract_title: str
    contract_type: str
    contract_version: str
    policy_id: str
    policy_version: str
    document_complete: bool
    section_count: int
    source_metadata: Dict[str, Any] = Field(default_factory=dict)
    intake_warnings: List[str] = Field(default_factory=list)


class ExtractedClause(BaseModel):
    clause_id: str
    clause_type: str  # payment, termination, renewal, confidentiality, liability, etc.
    clause_title: str
    clause_text: str
    section_reference: str


class ClauseExtractionOutput(BaseModel):
    contract_id: str
    clauses: List[ExtractedClause] = Field(default_factory=list)
    absent_clause_types: List[str] = Field(default_factory=list)
    extraction_notes: List[str] = Field(default_factory=list)


class PolicyRequirement(BaseModel):
    requirement_id: str
    clause_type: str
    is_mandatory: bool
    requirement_text: str
    threshold_or_condition: Optional[str] = None
    exception_rules: List[str] = Field(default_factory=list)


class PolicyInterpretationOutput(BaseModel):
    policy_id: str
    policy_version: str
    policy_name: str
    mandatory_requirements: List[PolicyRequirement] = Field(default_factory=list)
    recommended_requirements: List[PolicyRequirement] = Field(default_factory=list)
    ambiguities_in_policy: List[str] = Field(default_factory=list)


class ClausePolicyMapping(BaseModel):
    mapping_id: str
    clause_id: Optional[str] = None
    requirement_id: str
    mapping_status: str  # satisfied, partially_satisfied, conflicting, unaddressed
    supporting_clause_text: Optional[str] = None
    notes: str = ""


class ClauseMappingOutput(BaseModel):
    contract_id: str
    policy_id: str
    mappings: List[ClausePolicyMapping] = Field(default_factory=list)
    coverage_percentage: float = 0.0


class ComplianceDeviation(BaseModel):
    deviation_id: str
    requirement_id: str
    clause_id: Optional[str] = None
    severity: str  # low, medium, high, critical
    deviation_summary: str
    supporting_evidence: str
    recommended_remediation: str


class ComplianceDeviationOutput(BaseModel):
    contract_id: str
    deviations: List[ComplianceDeviation] = Field(default_factory=list)
    has_critical_deviations: bool = False


class RiskAmbiguityFinding(BaseModel):
    finding_id: str
    clause_id: str
    risk_category: str  # operational, legal_exposure, vague_terms, renewal_trap
    ambiguity_description: str
    severity: str  # low, medium, high
    human_interpretation_needed: bool = True


class RiskAmbiguityOutput(BaseModel):
    contract_id: str
    risk_findings: List[RiskAmbiguityFinding] = Field(default_factory=list)


class CrossClauseContradiction(BaseModel):
    issue_id: str
    clause_id_a: str
    clause_id_b: str
    description: str
    interaction_type: str  # termination_vs_renewal, liability_vs_indemnity, etc.
    evidence_a: str
    evidence_b: str


class CrossClauseConsistencyOutput(BaseModel):
    contract_id: str
    inconsistencies: List[CrossClauseContradiction] = Field(default_factory=list)


class EvidenceValidationOutput(BaseModel):
    contract_id: str
    all_citations_valid: bool
    verified_finding_ids: List[str] = Field(default_factory=list)
    invalid_or_unsupported_findings: List[str] = Field(default_factory=list)
    evidence_notes: List[str] = Field(default_factory=list)


class IndependentReviewOutput(BaseModel):
    contract_id: str
    review_status: str  # approved_for_report, revisions_needed
    duplicate_findings_merged: int = 0
    severity_adjustments: List[str] = Field(default_factory=list)
    unresolved_matters: List[str] = Field(default_factory=list)


class ContractFinalReportOutput(BaseModel):
    run_id: str
    contract_id: str
    contract_version: str
    policy_id: str
    policy_version: str
    contract_type: str
    executive_summary: str
    policy_coverage: Dict[str, Any]
    clause_inventory: List[Dict[str, Any]]
    compliance_findings: List[Dict[str, Any]]
    risk_findings: List[Dict[str, Any]]
    cross_clause_inconsistencies: List[Dict[str, Any]]
    evidence_validation: Dict[str, Any]
    unresolved_questions: List[str]
    recommended_actions: List[str]
    human_review_required: bool = True
    validation_warnings: List[str] = Field(default_factory=list)
    legal_disclaimer: str = (
        "This AI-assisted review is for preliminary compliance screening only "
        "and is NOT a substitute for qualified legal or compliance counsel. "
        "No detected issues does NOT imply complete legal validity."
    )
