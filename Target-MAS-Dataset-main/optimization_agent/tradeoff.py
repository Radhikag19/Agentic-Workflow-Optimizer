"""
Feasibility and Trade-Off Assessment Engine for Optimization Discovery Agent.

Evaluates every discovered optimization candidate across:
1. Expected Benefit (latency, tokens, cost, quality)
2. Trade-Offs (speed vs quality, implementation complexity, risk)
3. Feasibility Classification into Tiers 1 through 4:
   - Tier 1: High potential, low trade-off (immediate implementation)
   - Tier 2: High potential, moderate trade-off (recommended with validation)
   - Tier 3: Potential benefit, high trade-off (proceed with caution)
   - Tier 4: Insufficient evidence for recommendation (further observation)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from optimization_agent.key_manager import GroqKeyManager, get_default_key_manager
from optimization_agent.state import OptimizationAgentState

logger = logging.getLogger("optimization_agent.tradeoff")

TIER_1_LABEL = "Tier 1: High potential, low trade-off"
TIER_2_LABEL = "Tier 2: High potential, moderate trade-off"
TIER_3_LABEL = "Tier 3: Potential benefit, high trade-off"
TIER_4_LABEL = "Tier 4: Insufficient evidence for recommendation"

TIER_SHORT_LABELS = {
    "tier_1": "Tier 1",
    "tier_2": "Tier 2",
    "tier_3": "Tier 3",
    "tier_4": "Tier 4",
}


class TradeoffEvaluator:
    """Evaluates feasibility, trade-offs, and risk tiers for candidates."""

    def __init__(self, key_manager: Optional[GroqKeyManager] = None):
        self.key_mgr = key_manager or get_default_key_manager()

    def evaluate_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        raw_key, env_var, masked = self.key_mgr.get_key_for_task("tradeoff_assessment")
        evaluated: List[Dict[str, Any]] = []
        tier_classifications: Dict[str, List[Dict[str, Any]]] = {
            "tier_1": [],
            "tier_2": [],
            "tier_3": [],
            "tier_4": [],
        }

        for c in candidates:
            c_id = c.get("candidate_id", "UNKNOWN")
            dim = c.get("dimension", "")
            exp = c.get("expected_benefit", {})
            risks = c.get("tradeoffs_and_risks", {})
            req_val = bool(c.get("requires_validation", False))
            # Agent consolidation & data handoff modify schemas/boundaries and always require validation
            if dim in ("agent_consolidation", "communication_data_handoff"):
                req_val = True

            risk_level = risks.get("output_quality_risk", "Low")
            complexity = risks.get("implementation_complexity", "Low")

            # Determine Tier Classification
            supp_ev = c.get("supporting_evidence")
            is_insufficient = (
                not supp_ev
                or (isinstance(supp_ev, dict) and supp_ev.get("evidence_status") == "insufficient_evidence")
                or (isinstance(supp_ev, str) and "insufficient" in supp_ev.lower())
            )
            if is_insufficient:
                tier = TIER_4_LABEL
                tier_key = "tier_4"
            elif "high" in risk_level.lower():
                tier = TIER_3_LABEL
                tier_key = "tier_3"
            elif req_val or "moderate" in risk_level.lower() or complexity in ["Medium", "High"]:
                tier = TIER_2_LABEL
                tier_key = "tier_2"
            else:
                tier = TIER_1_LABEL
                tier_key = "tier_1"

            tier_short = TIER_SHORT_LABELS[tier_key]

            tradeoffs_dict = {
                "latency_tradeoff": risks.get("latency_tradeoff", "Improved"),
                "token_cost_tradeoff": risks.get("token_cost_tradeoff", "Reduced"),
                "output_quality_risk": risk_level,
                "implementation_complexity": complexity,
            }
            if dim == "communication_data_handoff":
                tradeoffs_dict.update({
                    "fields_removed_count": exp.get("fields_removed_count", 0),
                    "payload_size_reduction_pct": exp.get("byte_savings_pct"),
                    "token_savings_pct": exp.get("token_savings_pct"),
                    "shared_storage_reference_dependency": bool(
                        risks.get("shared_storage_reference_dependency", False)
                    ),
                    "schema_compatibility_check_required": True,
                })

            assessment_record = {
                "candidate_id": c_id,
                "dimension": dim,
                "title": c.get("title", ""),
                "target_agents": c.get("target_agents", []),
                "expected_benefit": exp,
                "tradeoffs": tradeoffs_dict,
                "feasibility_tier": tier,
                "tier_short": tier_short,
                "requires_validation": req_val,
                "evaluated_with_key": masked,
            }

            evaluated.append(assessment_record)
            tier_classifications[tier_key].append(assessment_record)

        return {
            "tradeoff_assessments": evaluated,
            "tier_classifications": tier_classifications,
            "assessment_key": masked,
        }


def tradeoff_assessment(state: OptimizationAgentState) -> Dict[str, Any]:
    """LangGraph Node 12: Evaluate candidates for feasibility and trade-offs."""
    track_status = dict(state.get("stage_status") or {})
    retrieval_errors = list(state.get("retrieval_errors") or [])
    key_mgr = get_default_key_manager()

    try:
        candidates = [dict(c) for c in (state.get("optimization_candidates") or [])]
        evaluator = TradeoffEvaluator(key_manager=key_mgr)
        results = evaluator.evaluate_candidates(candidates)

        # Keep candidate tier and requires_validation strictly synchronized with Stage 12 evaluation
        for cand, rec in zip(candidates, results["tradeoff_assessments"]):
            cand["tier"] = rec["tier_short"]
            cand["feasibility_tier"] = rec["feasibility_tier"]
            if cand.get("dimension") in ("agent_consolidation", "communication_data_handoff"):
                cand["requires_validation"] = True

        track_status["tradeoff_assessment"] = "completed"
        return {
            "stage_status": track_status,
            "optimization_candidates": candidates,
            "tradeoff_assessments": results["tradeoff_assessments"],
            "tier_classifications": results["tier_classifications"],
            "key_usage_log": key_mgr.get_usage_log(),
        }
    except Exception as e:
        logger.error(f"Tradeoff assessment error: {e}")
        track_status["tradeoff_assessment"] = "failed"
        retrieval_errors.append({"stage": "tradeoff_assessment", "error": str(e)})
        return {
            "stage_status": track_status,
            "retrieval_errors": retrieval_errors,
            "tradeoff_assessments": [],
            "tier_classifications": {"tier_1": [], "tier_2": [], "tier_3": [], "tier_4": []},
        }
