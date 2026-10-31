"""
Agent 6: Validation and Final Review Agent for Inventory Replenishment Planning.

Responsibilities:
- Independently validate proposed quantities, calculations, stock assumptions, and supplier rules.
- Check for double-counting of incoming inventory.
- Identify infeasible, excessive, or risky replenishment recommendations.
- Output the complete, structured final inventory plan matching Section 6 schema.
"""

from typing import Any, Dict, List, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import InventoryFinalPlanOutput
from common.metrics import MetricsTracker

AGENT_ID = "inventory_review_agent"
ROLE = "Replenishment Integrity and Compliance Auditor"
GOAL = (
    "Conduct an independent audit of proposed replenishment quantities, verifying mathematical "
    "consistency, constraints compliance, and absence of double-counting before assembling the final plan."
)
BACKSTORY = (
    "You are a supply chain compliance auditor. You cross-examine every proposed order, "
    "ensuring available stock arithmetic is sound, minimum pack increments were honored, "
    "and warnings are recorded for volatile or stockout-prone items."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Validation and Final Review Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_inventory_review(
    run_id: str,
    planning_date: str,
    intake_data: Dict[str, Any],
    stock_policy_data: Dict[str, Any],
    demand_forecast_data: Dict[str, Any],
    supplier_data: Dict[str, Any],
    planning_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Validation and Final Review Agent."""
    active_agent = agent or create_agent()

    sku_id = intake_data.get("sku_id", "SKU-UNKNOWN")
    warehouse_id = intake_data.get("warehouse_id", "WH-UNKNOWN")
    supplier_id = supplier_data.get("supplier_id", "SUP-UNKNOWN")

    cur_stock = int(stock_policy_data.get("current_stock", 0))
    res_stock = int(stock_policy_data.get("reserved_stock", 0))
    inc_stock = int(stock_policy_data.get("incoming_stock", 0))
    avail_stock = cur_stock - res_stock

    f_demand = float(demand_forecast_data.get("forecast_demand_units", 0.0))
    f_horizon = f"{demand_forecast_data.get('forecast_horizon_days', 14)} days"
    lead_time = int(supplier_data.get("lead_time_days", 7))
    safety_stock = int(stock_policy_data.get("safety_stock", 0))
    reorder_thresh = int(stock_policy_data.get("reorder_threshold", 0))

    base_qty = int(planning_data.get("base_replenishment_quantity", 0))
    rec_qty = int(planning_data.get("recommended_order_quantity", 0))
    expected_arr = supplier_data.get("expected_arrival_date", "")
    stockout_risk = planning_data.get("stockout_risk", "none")
    reasoning = planning_data.get("adjustment_reasoning", "")

    source_ids = list(set(intake_data.get("source_record_ids", []) + supplier_data.get("source_record_ids", [])))

    # Deterministic validation checks
    warnings: List[str] = []
    if cur_stock == 0:
        warnings.append("CRITICAL: SKU is currently completely stocked out (physical stock = 0).")
    if rec_qty > (f_demand * 5) and rec_qty > 0:
        warnings.append("WARNING: Recommended order quantity significantly exceeds 5x horizon forecast.")
    if demand_forecast_data.get("demand_spikes_detected", False):
        warnings.append("NOTE: Recent demand spikes detected; review for potential non-recurring spike.")

    human_review_required = len(warnings) > 0 or stockout_risk in ["moderate", "severe"]

    task_desc = f"""
Audit and assemble the final structured replenishment plan:
- Run ID: {run_id}
- Planning Date: {planning_date}
- Warehouse: {warehouse_id}
- SKU ID: {sku_id}
- Current Stock: {cur_stock}
- Reserved Stock: {res_stock}
- Available Stock (Current - Reserved): {avail_stock}
- Incoming Stock: {inc_stock}
- Forecast Demand: {f_demand} ({f_horizon})
- Supplier ID: {supplier_id} (Lead Time: {lead_time} days, Expected Arrival: {expected_arr})
- Safety Stock: {safety_stock}
- Reorder Threshold: {reorder_thresh}
- Base Replenishment Quantity: {base_qty}
- Recommended Order Quantity: {rec_qty}
- Stockout Risk: {stockout_risk}
- Reasoning: {reasoning}
- Source IDs: {source_ids}
- Pre-Audit Warnings: {warnings}
- Human Review Required: {str(human_review_required).lower()}

Return structured JSON strictly matching:
{{
  "run_id": "{run_id}",
  "planning_date": "{planning_date}",
  "warehouse_id": "{warehouse_id}",
  "sku_id": "{sku_id}",
  "current_stock": {cur_stock},
  "reserved_stock": {res_stock},
  "available_stock": {avail_stock},
  "incoming_stock": {inc_stock},
  "forecast_demand": {f_demand},
  "forecast_horizon": "{f_horizon}",
  "supplier_id": "{supplier_id}",
  "lead_time_days": {lead_time},
  "safety_stock": {safety_stock},
  "reorder_threshold": {reorder_thresh},
  "base_replenishment_quantity": {base_qty},
  "recommended_order_quantity": {rec_qty},
  "expected_arrival_date": "{expected_arr}",
  "stockout_risk": "{stockout_risk}",
  "reasoning": "{reasoning}",
  "source_record_ids": {source_ids},
  "validation_warnings": {warnings},
  "human_review_required": {str(human_review_required).lower()},
  "disclaimer": "Planning tool recommendation only. Does NOT execute real purchases."
}}
"""
    expected_output = "Valid JSON matching InventoryFinalPlanOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=InventoryFinalPlanOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
