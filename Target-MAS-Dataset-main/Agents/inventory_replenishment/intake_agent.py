"""
Agent 1: Inventory Intake Agent for Inventory Replenishment Planning.

Responsibilities:
- Accept planning request and SKU or warehouse identifiers.
- Retrieve relevant local inventory, sales, supplier, and policy records.
- Normalize identifiers and validate required fields.
- Identify missing or inconsistent information.
- Return structured input package with source IDs.
"""

from typing import Any, Dict, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import InventoryIntakeOutput
from common.metrics import MetricsTracker

AGENT_ID = "inventory_intake_agent"
ROLE = "Inventory Intake Coordinator"
GOAL = (
    "Assemble, normalize, and validate all relevant inventory snapshots, sales history, "
    "supplier profiles, and replenishment policies for the target SKU."
)
BACKSTORY = (
    "You are a supply chain data coordinator. You gather inventory levels, outstanding orders, "
    "and vendor parameters into a coherent, validated planning package, highlighting any data "
    "integrity defects before quantitative planning begins."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Inventory Intake Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_inventory_intake(
    request_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Inventory Intake Agent."""
    active_agent = agent or create_agent()

    req_id = request_data.get("request_id", "REQ-INV-001")
    sku = request_data.get("sku_record", {})
    supplier = request_data.get("supplier_record", {})
    policy = request_data.get("policy_record", {})
    sales = request_data.get("sales_record", {})

    sku_id = sku.get("sku_id", "SKU-UNKNOWN")
    sku_name = sku.get("sku_name", "Unknown SKU")
    warehouse_id = sku.get("warehouse_id", "WH-UNKNOWN")
    sales_days = sales.get("recorded_days", len(sales.get("daily_sales", [])))

    task_desc = f"""
Request ID: {req_id}
Target SKU ID: {sku_id}
Warehouse: {warehouse_id}

Inventory Data:
{sku}

Supplier Data:
{supplier}

Policy Data:
{policy}

Recorded Sales Days: {sales_days}

Validate and structure the intake package:
1. Normalize SKU ID, Warehouse ID, and product name.
2. Verify all required fields (current_stock, reserved_stock, lead_time_days) are present.
3. Check for data quality anomalies (e.g., negative stock, zero lead time, or missing sales).
4. Catalog all source record IDs (e.g., INV-{sku_id}, SUP-{supplier.get('supplier_id', '00')}).

Return structured JSON matching:
{{
  "request_id": "{req_id}",
  "sku_id": "{sku_id}",
  "warehouse_id": "{warehouse_id}",
  "sku_name": "{sku_name}",
  "source_record_ids": ["INV-{sku_id}", "SUP-{supplier.get('supplier_id', '00')}"],
  "raw_inventory_record": {sku},
  "raw_supplier_record": {supplier},
  "raw_policy_record": {policy},
  "sales_history_days": {sales_days},
  "data_quality_issues": []
}}
"""
    expected_output = "Valid JSON matching InventoryIntakeOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=InventoryIntakeOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
