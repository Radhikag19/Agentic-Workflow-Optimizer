"""
Agent 5: Replenishment Planning Agent for Inventory Replenishment Planning.

Responsibilities:
- Synthesize inventory snapshots, policy thresholds, demand forecasts, and supplier parameters.
- Calculate base replenishment quantity using deterministic Python arithmetic.
- Apply pack-size rounding and minimum order quantities (MOQ).
- Separate raw unadjusted need from final recommended order quantity.
- Assess stockout risk and explain adjustments.
"""

import math
from typing import Any, Dict, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import ReplenishmentPlanningOutput
from common.metrics import MetricsTracker

AGENT_ID = "replenishment_planning_agent"
ROLE = "Replenishment Optimization Planner"
GOAL = (
    "Synthesize stock levels, demand rates, and supplier constraints to compute "
    "exact baseline and adjusted order quantities using rigorous deterministic logic."
)
BACKSTORY = (
    "You are a master inventory replenishment planner. You use deterministic arithmetic "
    "for supply/demand balances, strictly rounding up to package increments and applying MOQs "
    "while explaining every step with absolute transparency."
)


def calculate_deterministic_replenishment(
    current_stock: int,
    reserved_stock: int,
    incoming_stock: int,
    avg_daily_demand: float,
    lead_time_days: int,
    safety_stock: int,
    pack_size: int,
    moq: int,
) -> Dict[str, Any]:
    """
    Deterministic calculation for inventory replenishment:
    1. Usable Inventory = Current Stock - Reserved Stock + Incoming Stock
    2. Expected Lead-Time Demand = avg_daily_demand * lead_time_days
    3. Gross Requirement = Expected Lead-Time Demand + Safety Stock
    4. Base Replenishment = max(0, math.ceil(Gross Requirement - Usable Inventory))
    5. Adjusted Order Quantity:
       - If base > 0: round up to nearest pack_size, then max(rounded, moq).
       - If base == 0: 0.
    """
    usable_inv = current_stock - reserved_stock + incoming_stock
    lead_time_demand = avg_daily_demand * lead_time_days
    gross_req = lead_time_demand + safety_stock
    raw_base = gross_req - usable_inv
    base_qty = max(0, int(math.ceil(raw_base)))

    if base_qty > 0:
        # Round up to nearest pack_size
        pack = max(1, pack_size)
        rounded_pack = ((base_qty + pack - 1) // pack) * pack
        recommended_qty = max(rounded_pack, moq)
    else:
        rounded_pack = 0
        recommended_qty = 0

    # Stockout risk classification
    if current_stock == 0 and usable_inv <= 0:
        stockout_risk = "severe"
    elif usable_inv < safety_stock:
        stockout_risk = "moderate"
    elif base_qty > 0:
        stockout_risk = "low"
    else:
        stockout_risk = "none"

    breakdown = (
        f"Usable Inv={usable_inv} (Cur={current_stock} - Res={reserved_stock} + Inc={incoming_stock}). "
        f"LeadTimeDemand={round(lead_time_demand, 1)} ({avg_daily_demand}*LT {lead_time_days}) + Safety={safety_stock} = Gross {round(gross_req, 1)}. "
        f"Raw Deficit={round(raw_base, 1)} -> Base Qty={base_qty}."
    )

    reasoning = (
        f"Base need of {base_qty} units was rounded up to pack size multiple of {pack_size} ({rounded_pack} units), "
        f"and benchmarked against MOQ of {moq} units, yielding final order recommendation of {recommended_qty} units."
        if base_qty > 0
        else "Net usable inventory satisfies safety stock and projected lead-time demand; no replenishment required."
    )

    return {
        "base_replenishment_quantity": base_qty,
        "recommended_order_quantity": recommended_qty,
        "calculation_breakdown": breakdown,
        "adjustment_reasoning": reasoning,
        "stockout_risk": stockout_risk,
        "infeasible_flag": False,
    }


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Replenishment Planning Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_replenishment_planning(
    sku_id: str,
    stock_policy_data: Dict[str, Any],
    demand_forecast_data: Dict[str, Any],
    supplier_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Replenishment Planning Agent."""
    active_agent = agent or create_agent()

    current_stock = int(stock_policy_data.get("current_stock", 0))
    reserved_stock = int(stock_policy_data.get("reserved_stock", 0))
    incoming_stock = int(stock_policy_data.get("incoming_stock", 0))
    safety_stock = int(stock_policy_data.get("safety_stock", 0))

    avg_daily_demand = float(demand_forecast_data.get("historical_avg_daily_demand", 10.0))
    lead_time_days = int(supplier_data.get("lead_time_days", 7))
    pack_size = int(supplier_data.get("pack_size", 10))
    moq = int(supplier_data.get("minimum_order_quantity", 50))

    # Deterministic Python Calculation
    calc = calculate_deterministic_replenishment(
        current_stock=current_stock,
        reserved_stock=reserved_stock,
        incoming_stock=incoming_stock,
        avg_daily_demand=avg_daily_demand,
        lead_time_days=lead_time_days,
        safety_stock=safety_stock,
        pack_size=pack_size,
        moq=moq,
    )

    task_desc = f"""
SKU ID: {sku_id}
Stock Policy Input: {stock_policy_data}
Demand Forecast Input: {demand_forecast_data}
Supplier Parameters: {supplier_data}

Deterministic Calculation Breakdown:
{calc['calculation_breakdown']}
Base Quantity: {calc['base_replenishment_quantity']}
Recommended Order Quantity (with MOQ & Pack Size): {calc['recommended_order_quantity']}
Stockout Risk: {calc['stockout_risk']}

Synthesize the replenishment plan and return structured JSON matching:
{{
  "sku_id": "{sku_id}",
  "base_replenishment_quantity": {calc['base_replenishment_quantity']},
  "recommended_order_quantity": {calc['recommended_order_quantity']},
  "calculation_breakdown": "{calc['calculation_breakdown']}",
  "adjustment_reasoning": "{calc['adjustment_reasoning']}",
  "stockout_risk": "{calc['stockout_risk']}",
  "infeasible_flag": false
}}
"""
    expected_output = "Valid JSON matching ReplenishmentPlanningOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=ReplenishmentPlanningOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
