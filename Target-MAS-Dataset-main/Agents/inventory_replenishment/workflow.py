"""
Workflow Orchestrator: Inventory Replenishment Planning (Workflow 2).

Executes exactly 6 CrewAI agents in sequential baseline order:
1. Inventory Intake Agent (intake_agent.py)
2. Stock and Reorder Policy Agent (stock_policy_agent.py)
3. Sales History and Demand Forecast Agent (demand_forecast_agent.py)
4. Supplier and Lead-Time Agent (supplier_agent.py)
5. Replenishment Planning Agent (replenishment_planning_agent.py)
6. Validation and Final Review Agent (review_agent.py)
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional
from common.metrics import MetricsTracker
from common.tracing import init_tracing

# Import all 6 agents
from Agents.inventory_replenishment import (
    intake_agent,
    stock_policy_agent,
    demand_forecast_agent,
    supplier_agent,
    replenishment_planning_agent,
    review_agent,
)

WORKFLOW_NAME = "inventory_replenishment"
EXPECTED_AGENT_COUNT = 6


def load_sample_inventory_input(
    sku_index: int = 0,
    data_dir: str = "data/inventory_replenishment",
) -> Dict[str, Any]:
    """Load sample inventory, supplier, sales, and policy data from local datasets."""
    skus_path = os.path.join(data_dir, "inventory_records.json")
    suppliers_path = os.path.join(data_dir, "suppliers.json")
    sales_path = os.path.join(data_dir, "sales_history.json")
    policies_path = os.path.join(data_dir, "policies.json")

    with open(skus_path, "r", encoding="utf-8") as f:
        skus = json.load(f)
    with open(suppliers_path, "r", encoding="utf-8") as f:
        suppliers = json.load(f)
    with open(sales_path, "r", encoding="utf-8") as f:
        sales_hist = json.load(f)
    with open(policies_path, "r", encoding="utf-8") as f:
        policies = json.load(f)

    sku = skus[sku_index % len(skus)]
    sup_id = sku.get("supplier_id")
    supplier = next((s for s in suppliers if s["supplier_id"] == sup_id), suppliers[0])
    sales = sales_hist.get(sku["sku_id"], {"daily_sales": [], "recorded_days": 0})

    return {
        "request_id": f"REQ-INV-{uuid.uuid4().hex[:6]}",
        "planning_date": "2026-10-15",
        "sku_record": sku,
        "supplier_record": supplier,
        "sales_record": sales,
        "policy_record": policies,
    }


class InventoryReplenishmentWorkflow:
    """Manages baseline execution of the 6-agent inventory replenishment workflow."""

    def __init__(self, llm: Optional[Any] = None, enable_tracing: bool = True):
        self.llm = llm
        if enable_tracing:
            init_tracing(project_name="awo-inventory-replenishment")

        # Instantiate all 6 CrewAI agents
        self.agent_1_intake = intake_agent.create_agent(llm=self.llm)
        self.agent_2_stock_policy = stock_policy_agent.create_agent(llm=self.llm)
        self.agent_3_demand_forecast = demand_forecast_agent.create_agent(llm=self.llm)
        self.agent_4_supplier = supplier_agent.create_agent(llm=self.llm)
        self.agent_5_planning = replenishment_planning_agent.create_agent(llm=self.llm)
        self.agent_6_review = review_agent.create_agent(llm=self.llm)

    def get_agent_count(self) -> int:
        """Return the exact number of agents."""
        return EXPECTED_AGENT_COUNT

    def run(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        runs_base_dir: str = "runs",
    ) -> Dict[str, Any]:
        """
        Execute the 6-agent replenishment workflow sequentially.
        """
        active_run_id = run_id or f"inv_{uuid.uuid4().hex[:8]}"
        tracker = MetricsTracker(workflow_name=WORKFLOW_NAME, run_id=active_run_id)

        if input_data is None:
            input_data = load_sample_inventory_input()

        planning_date = input_data.get("planning_date", "2026-10-15")

        try:
            # Stage 1: Inventory Intake Agent
            intake_res = intake_agent.run_inventory_intake(
                request_data=input_data,
                agent=self.agent_1_intake,
                tracker=tracker,
            )

            # Stage 2: Stock and Reorder Policy Agent
            stock_policy_res = stock_policy_agent.run_stock_policy(
                intake_data=intake_res,
                agent=self.agent_2_stock_policy,
                tracker=tracker,
            )

            # Stage 3: Sales History and Demand Forecast Agent
            demand_forecast_res = demand_forecast_agent.run_demand_forecast(
                intake_data=intake_res,
                sales_data=input_data.get("sales_record", {}),
                horizon_days=14,
                agent=self.agent_3_demand_forecast,
                tracker=tracker,
            )

            # Stage 4: Supplier and Lead-Time Agent
            supplier_res = supplier_agent.run_supplier_analysis(
                intake_data=intake_res,
                planning_date=planning_date,
                agent=self.agent_4_supplier,
                tracker=tracker,
            )

            # Stage 5: Replenishment Planning Agent
            planning_res = replenishment_planning_agent.run_replenishment_planning(
                sku_id=intake_res.get("sku_id", ""),
                stock_policy_data=stock_policy_res,
                demand_forecast_data=demand_forecast_res,
                supplier_data=supplier_res,
                agent=self.agent_5_planning,
                tracker=tracker,
            )

            # Stage 6: Validation and Final Review Agent
            final_plan = review_agent.run_inventory_review(
                run_id=active_run_id,
                planning_date=planning_date,
                intake_data=intake_res,
                stock_policy_data=stock_policy_res,
                demand_forecast_data=demand_forecast_res,
                supplier_data=supplier_res,
                planning_data=planning_res,
                agent=self.agent_6_review,
                tracker=tracker,
            )

            # Finalize metrics and persist run record
            record = tracker.finalize(
                output=final_plan,
                status="success",
                runs_base_dir=runs_base_dir,
            )
            output_dict = dict(final_plan) if isinstance(final_plan, dict) else (final_plan.model_dump() if hasattr(final_plan, "model_dump") else {})
            output_dict.update({
                "run_id": active_run_id,
                "status": "SUCCESS",
                "duration_seconds": record.total_duration_seconds,
                "total_tokens": record.token_usage.get("total_tokens", 0) if record.token_usage else 0,
                "prompt_tokens": record.token_usage.get("prompt_tokens", 0) if record.token_usage else 0,
                "completion_tokens": record.token_usage.get("completion_tokens", 0) if record.token_usage else 0,
                "estimated_cost_usd": record.estimated_cost_usd or 0.0,
                "agent_executions": [m.model_dump() for m in record.agent_metrics],
                "final_output": final_plan,
            })
            return output_dict

        except Exception as exc:
            tracker.finalize(
                output=None,
                status="failed",
                error=str(exc),
                runs_base_dir=runs_base_dir,
            )
            raise


def run_inventory_replenishment_workflow(
    input_data: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute inventory replenishment planning workflow."""
    wf = InventoryReplenishmentWorkflow(enable_tracing=True)
    return wf.run(input_data=input_data, run_id=run_id)

