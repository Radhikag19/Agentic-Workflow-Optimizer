"""
Agent 2: Stock and Reorder Policy Agent for Inventory Replenishment Planning.

Responsibilities:
- Review current stock, reserved stock, available stock, incoming inventory, outstanding POs.
- Retrieve reorder thresholds and safety stock requirements.
- Calculate usable inventory using deterministic Python arithmetic.
- Identify whether the SKU requires replenishment.
"""

from typing import Any, Dict, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import StockPolicyOutput
from common.metrics import MetricsTracker

AGENT_ID = "stock_policy_agent"
ROLE = "Stock Policy and Inventory Level Specialist"
GOAL = (
    "Evaluate physical stock, reservations, and incoming orders against safety thresholds, "
    "calculating exact usable inventory and determining replenishment need."
)
BACKSTORY = (
    "You are an inventory control analyst. You rigorously calculate usable and available stock "
    "using exact mathematical formulas to prevent double-counting of incoming goods, "
    "benchmarking net stock against defined reorder triggers."
)


def calculate_usable_inventory(current_stock: int, reserved_stock: int, incoming_stock: int) -> int:
    """
    Deterministic calculation of usable inventory.
    Usable Inventory = Current Stock - Reserved Stock + Incoming Stock
    """
    return current_stock - reserved_stock + incoming_stock


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Stock and Reorder Policy Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_stock_policy(
    intake_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Stock and Reorder Policy Agent."""
    active_agent = agent or create_agent()

    sku_id = intake_data.get("sku_id", "SKU-UNKNOWN")
    warehouse_id = intake_data.get("warehouse_id", "WH-UNKNOWN")
    raw_inv = intake_data.get("raw_inventory_record", {})

    current_stock = int(raw_inv.get("current_stock", 0))
    reserved_stock = int(raw_inv.get("reserved_stock", 0))
    incoming_stock = int(raw_inv.get("incoming_stock", 0))
    reorder_threshold = int(raw_inv.get("reorder_threshold", 0))
    safety_stock = int(raw_inv.get("safety_stock", 0))

    # Deterministic Python calculation
    usable_inventory = calculate_usable_inventory(current_stock, reserved_stock, incoming_stock)
    replenishment_needed = usable_inventory <= reorder_threshold

    task_desc = f"""
SKU ID: {sku_id}
Warehouse: {warehouse_id}
Stock Levels:
- Current Physical Stock: {current_stock}
- Reserved Stock: {reserved_stock}
- Incoming Stock (In-Transit / Open POs): {incoming_stock}
- Reorder Threshold: {reorder_threshold}
- Safety Stock Requirement: {safety_stock}

Deterministic Arithmetic Verified:
- Usable Inventory = Current ({current_stock}) - Reserved ({reserved_stock}) + Incoming ({incoming_stock}) = {usable_inventory}
- Replenishment Needed = (Usable <= Threshold) = {str(replenishment_needed).lower()}

Assess policy conformance and return structured JSON matching:
{{
  "sku_id": "{sku_id}",
  "warehouse_id": "{warehouse_id}",
  "current_stock": {current_stock},
  "reserved_stock": {reserved_stock},
  "incoming_stock": {incoming_stock},
  "usable_inventory": {usable_inventory},
  "reorder_threshold": {reorder_threshold},
  "safety_stock": {safety_stock},
  "replenishment_needed": {str(replenishment_needed).lower()},
  "policy_notes": [
    "Usable inventory incorporates incoming purchase orders without double-counting",
    "Current stock is below reorder threshold" if replenishment_needed else "Stock is healthy"
  ]
}}
"""
    expected_output = "Valid JSON matching StockPolicyOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=StockPolicyOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
