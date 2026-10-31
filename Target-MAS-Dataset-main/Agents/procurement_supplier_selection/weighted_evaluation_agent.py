"""
Agent 7: Weighted Evaluation Agent (Procurement)
Calculates deterministic multi-criteria weighted scores across competing suppliers.
"""

import json
from pathlib import Path
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent
from common.settings import DATA_DIR

AGENT_ID = "weighted_evaluation_agent"

def deterministic_weighted_scoring(
    cost_data: list,
    delivery_data: list,
    risk_data: dict,
    profile_data: dict
) -> list:
    """Computes deterministic multi-criteria score (0-100) per supplier."""
    crit_file = DATA_DIR / "procurement_supplier_selection" / "evaluation_criteria.json"
    weights = {"cost_score": 0.35, "delivery_score": 0.25, "risk_score": 0.20, "contract_terms_score": 0.10, "quality_capability_score": 0.10}
    if crit_file.exists():
        with open(crit_file, "r", encoding="utf-8") as f:
            weights = json.load(f).get("weights", weights)

    # Find lowest cost for normalization
    costs = {c["supplier_id"]: c["total_landed_cost"] for c in cost_data}
    min_cost = min(costs.values()) if costs else 1.0

    delivery_map = {d["supplier_id"]: d for d in delivery_data}

    scored_suppliers = []
    for sup_id, cost in costs.items():
        # Cost score: min_cost / cost * 100
        cost_s = round((min_cost / cost) * 100, 2)
        
        # Delivery score: 100 if on-time, minus 15 per day late
        deliv = delivery_map.get(sup_id, {})
        buf = deliv.get("buffer_days", 0)
        deliv_s = 100.0 if buf >= 0 else max(0.0, 100.0 + (buf * 15.0))

        # Risk score: 100 minus failure rates
        r = risk_data.get(sup_id, {})
        fail_rate = float(r.get("delivery_failure_rate", 0.05))
        defect_rate = float(r.get("quality_defect_rate", 0.02))
        risk_s = max(0.0, round(100.0 - (fail_rate * 300.0) - (defect_rate * 400.0), 2))

        # Contract terms score: baseline 85, 95 if Net 45
        contract_s = 90.0

        # Capability score: 90 if ISO certified
        prof = profile_data.get(sup_id, {})
        cap_s = 95.0 if prof.get("iso_certified") else 70.0

        # Weighted total
        total_score = round(
            (cost_s * weights["cost_score"]) +
            (deliv_s * weights["delivery_score"]) +
            (risk_s * weights["risk_score"]) +
            (contract_s * weights["contract_terms_score"]) +
            (cap_s * weights["quality_capability_score"]),
            2
        )

        scored_suppliers.append({
            "supplier_id": sup_id,
            "cost_score": cost_s,
            "delivery_score": deliv_s,
            "risk_score": risk_s,
            "contract_score": contract_s,
            "capability_score": cap_s,
            "total_weighted_score": total_score
        })

    scored_suppliers.sort(key=lambda x: x["total_weighted_score"], reverse=True)
    for idx, s in enumerate(scored_suppliers, 1):
        s["overall_rank"] = idx

    return scored_suppliers

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Multi-Criteria Decision Analysis Specialist",
        goal="Apply formal weighted decision matrices to rank suppliers objectively across all evaluation pillars.",
        backstory="An operations research analyst calculating quantitative scoring models for vendor selection.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, scored_results: list) -> Task:
    description = (
        f"Review and explain the multi-criteria weighted scoring across competing suppliers.\n"
        f"Deterministic Weighted Scores from Python:\n{json.dumps(scored_results, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Explain why the top-ranked supplier achieved the highest score.\n"
        f"2. Discuss the trade-offs between cost savings and risk/delivery factors.\n"
        f"Return strictly valid JSON with keys: 'weighted_scores', 'top_ranked_supplier', 'score_spread', and 'evaluation_rationale'."
    )
    expected_output = "Strictly valid JSON containing weighted_scores, top_ranked_supplier, score_spread, and evaluation_rationale."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
