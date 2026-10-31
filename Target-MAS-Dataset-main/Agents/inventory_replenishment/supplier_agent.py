"""
Agent 4: Supplier and Lead-Time Agent for Inventory Replenishment Planning.

Responsibilities:
- Retrieve supplier records, delivery lead times, minimum order quantities, and pack sizes.
- Calculate expected arrival dates based on planning date and delivery lead time.
- Identify missing, conflicting, or unreliable supplier parameters.
- Preserve source record IDs.
"""

from datetime import date, timedelta
from typing import Any, Dict, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import SupplierLeadTimeOutput
from common.metrics import MetricsTracker

AGENT_ID = "supplier_agent"
ROLE = "Supplier Lead-Time and Constraints Specialist"
GOAL = (
    "Extract and validate vendor lead times, order increments (pack size), and minimum order "
    "quantities (MOQ), calculating projected shipment arrival dates."
)
BACKSTORY = (
    "You are a procurement vendor manager. You verify supplier performance terms, ensure lead "
    "time calculations reflect delivery reality, and safeguard purchasing constraints."
)


def calculate_expected_arrival(planning_date_str: str, lead_time_days: int) -> str:
    """Deterministic calculation of expected arrival date."""
    try:
        p_date = date.fromisoformat(planning_date_str)
    except Exception:
        p_date = date.today()
    return (p_date + timedelta(days=lead_time_days)).isoformat()


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Supplier and Lead-Time Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_supplier_analysis(
    intake_data: Dict[str, Any],
    planning_date: str = "2026-10-15",
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Supplier and Lead-Time Agent."""
    active_agent = agent or create_agent()

    raw_sup = intake_data.get("raw_supplier_record", {})
    sup_id = raw_sup.get("supplier_id", "SUP-UNKNOWN")
    lead_time = int(raw_sup.get("lead_time_days", 7))
    moq = int(raw_sup.get("minimum_order_quantity", 100))
    pack_size = int(raw_sup.get("pack_size", 10))
    reliability = float(raw_sup.get("reliability_score", 0.95))

    expected_arrival = calculate_expected_arrival(planning_date, lead_time)

    task_desc = f"""
Supplier Record:
- Supplier ID: {sup_id}
- Lead Time: {lead_time} days
- Minimum Order Quantity (MOQ): {moq} units
- Pack Size: {pack_size} units/pack
- Reliability Score: {reliability}
- Planning Date: {planning_date}
- Deterministic Expected Arrival Date: {expected_arrival}

Verify supplier parameters and constraints.
Return structured JSON matching:
{{
  "supplier_id": "{sup_id}",
  "lead_time_days": {lead_time},
  "minimum_order_quantity": {moq},
  "pack_size": {pack_size},
  "expected_arrival_date": "{expected_arrival}",
  "supplier_reliability_notes": [
    "Supplier reliability rated at {int(reliability * 100)}%",
    "Expected lead time of {lead_time} days from order placement"
  ],
  "source_record_ids": ["SUP-{sup_id}"]
}}
"""
    expected_output = "Valid JSON matching SupplierLeadTimeOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=SupplierLeadTimeOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
