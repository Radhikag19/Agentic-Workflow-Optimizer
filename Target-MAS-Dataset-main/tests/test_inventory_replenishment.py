"""
Unit Tests for Workflow 2: Inventory Replenishment Planning (6 Agents).

Tests execute with mocks to avoid consuming Groq API credits.
Validates inventory arithmetic, incoming order handling, MOQ/pack-size logic.
Smoke test is marked with @pytest.mark.smoke for live testing.
"""

import json
import os
from unittest.mock import MagicMock, patch
import pytest

from common.validation import (
    InventoryIntakeOutput,
    StockPolicyOutput,
    DemandForecastOutput,
    SupplierLeadTimeOutput,
    ReplenishmentPlanningOutput,
    InventoryFinalPlanOutput,
)
from Agents.inventory_replenishment.workflow import (
    InventoryReplenishmentWorkflow,
    load_sample_inventory_input,
    EXPECTED_AGENT_COUNT,
)
from Agents.inventory_replenishment.stock_policy_agent import calculate_usable_inventory
from Agents.inventory_replenishment.replenishment_planning_agent import calculate_deterministic_replenishment
from Agents.inventory_replenishment.demand_forecast_agent import calculate_deterministic_forecast


def test_inventory_agent_count():
    """Verify workflow implements exactly 6 agents."""
    workflow = InventoryReplenishmentWorkflow(enable_tracing=False)
    assert workflow.get_agent_count() == 6
    assert EXPECTED_AGENT_COUNT == 6


def test_inventory_datasets():
    """Verify local synthetic inventory datasets exist and adhere to schema."""
    skus_path = "data/inventory_replenishment/inventory_records.json"
    suppliers_path = "data/inventory_replenishment/suppliers.json"
    sales_path = "data/inventory_replenishment/sales_history.json"
    pos_path = "data/inventory_replenishment/purchase_orders.json"

    assert os.path.exists(skus_path)
    assert os.path.exists(suppliers_path)
    assert os.path.exists(sales_path)
    assert os.path.exists(pos_path)

    with open(skus_path, "r", encoding="utf-8") as f:
        skus = json.load(f)
    assert len(skus) >= 100

    sample = skus[0]
    assert "sku_id" in sample
    assert "current_stock" in sample
    assert "reserved_stock" in sample
    assert "incoming_stock" in sample
    assert "reorder_threshold" in sample
    assert "safety_stock" in sample


def test_usable_inventory_arithmetic():
    """Verify usable inventory = current_stock - reserved_stock + incoming_stock."""
    usable = calculate_usable_inventory(current_stock=100, reserved_stock=20, incoming_stock=30)
    assert usable == 110

    # Stockout case
    usable_zero = calculate_usable_inventory(current_stock=0, reserved_stock=0, incoming_stock=0)
    assert usable_zero == 0


def test_demand_forecast_calculation():
    """Verify deterministic moving average demand calculation."""
    daily_sales = [{"date": "2026-10-01", "units_sold": 10} for _ in range(30)]
    forecast = calculate_deterministic_forecast(daily_sales, horizon_days=14)
    assert forecast["avg_daily"] == 10.0
    assert forecast["forecast_units"] == 140.0
    assert forecast["sufficient_history"] is True
    assert forecast["volatility"] == "low"


def test_replenishment_quantity_and_moq_rounding():
    """Verify lead-time demand, safety stock, pack size, and MOQ logic."""
    # Scenario:
    # Usable = 50, avg_daily = 10, lead_time = 7 days -> LT demand = 70.
    # Safety stock = 30 -> Gross requirement = 100.
    # Base deficit = 100 - 50 = 50 units.
    # Pack size = 15 -> 50 rounded up to 15 is 60.
    # MOQ = 100 -> final recommended order = max(60, 100) = 100.
    res = calculate_deterministic_replenishment(
        current_stock=50,
        reserved_stock=10,
        incoming_stock=10,  # usable = 50
        avg_daily_demand=10.0,
        lead_time_days=7,
        safety_stock=30,
        pack_size=15,
        moq=100,
    )
    assert res["base_replenishment_quantity"] == 50
    assert res["recommended_order_quantity"] == 100
    assert res["stockout_risk"] in ["low", "none"]

    # Scenario where usable stock exceeds gross requirement -> zero order
    res_healthy = calculate_deterministic_replenishment(
        current_stock=200,
        reserved_stock=0,
        incoming_stock=0,
        avg_daily_demand=5.0,
        lead_time_days=7,
        safety_stock=20,
        pack_size=10,
        moq=50,
    )
    assert res_healthy["base_replenishment_quantity"] == 0
    assert res_healthy["recommended_order_quantity"] == 0


def test_inventory_workflow_mock_execution(tmp_path):
    """
    Test end-to-end execution of the 6-agent inventory replenishment workflow using mocks.
    """
    sample_input = load_sample_inventory_input()

    mock_intake = {
        "request_id": "REQ-INV-001",
        "sku_id": "SKU-0001",
        "warehouse_id": "WH-CENTRAL-01",
        "sku_name": "Electronics Component Model 001",
        "source_record_ids": ["INV-SKU-0001", "SUP-SUP-01"],
        "raw_inventory_record": {
            "current_stock": 20,
            "reserved_stock": 5,
            "incoming_stock": 10,
            "reorder_threshold": 80,
            "safety_stock": 30,
        },
        "raw_supplier_record": {"lead_time_days": 7, "minimum_order_quantity": 50, "pack_size": 10},
        "raw_policy_record": {},
        "sales_history_days": 90,
        "data_quality_issues": [],
    }

    mock_stock = {
        "sku_id": "SKU-0001",
        "warehouse_id": "WH-CENTRAL-01",
        "current_stock": 20,
        "reserved_stock": 5,
        "incoming_stock": 10,
        "usable_inventory": 25,
        "reorder_threshold": 80,
        "safety_stock": 30,
        "replenishment_needed": True,
        "policy_notes": ["Replenishment triggered"],
    }

    mock_demand = {
        "sku_id": "SKU-0001",
        "forecast_horizon_days": 14,
        "historical_avg_daily_demand": 10.0,
        "forecast_demand_units": 140.0,
        "volatility_metric": "low",
        "demand_spikes_detected": False,
        "has_sufficient_history": True,
        "source_record_ids": ["SALES-SKU-0001"],
        "assumptions": ["90-day moving average"],
    }

    mock_sup = {
        "supplier_id": "SUP-01",
        "lead_time_days": 7,
        "minimum_order_quantity": 50,
        "pack_size": 10,
        "expected_arrival_date": "2026-10-22",
        "supplier_reliability_notes": ["95% on-time"],
        "source_record_ids": ["SUP-SUP-01"],
    }

    mock_plan = {
        "sku_id": "SKU-0001",
        "base_replenishment_quantity": 75,
        "recommended_order_quantity": 80,
        "calculation_breakdown": "Gross 100 - Usable 25 = Deficit 75 -> Round to 80",
        "adjustment_reasoning": "Rounded up to nearest pack size of 10",
        "stockout_risk": "moderate",
        "infeasible_flag": False,
    }

    mock_final = {
        "run_id": "inv_test_001",
        "planning_date": "2026-10-15",
        "warehouse_id": "WH-CENTRAL-01",
        "sku_id": "SKU-0001",
        "current_stock": 20,
        "reserved_stock": 5,
        "available_stock": 15,
        "incoming_stock": 10,
        "forecast_demand": 140.0,
        "forecast_horizon": "14 days",
        "supplier_id": "SUP-01",
        "lead_time_days": 7,
        "safety_stock": 30,
        "reorder_threshold": 80,
        "base_replenishment_quantity": 75,
        "recommended_order_quantity": 80,
        "expected_arrival_date": "2026-10-22",
        "stockout_risk": "moderate",
        "reasoning": "Rounded up to pack size 10",
        "source_record_ids": ["INV-SKU-0001", "SUP-SUP-01"],
        "validation_warnings": [],
        "human_review_required": True,
        "disclaimer": "Planning tool recommendation only. Does NOT execute real purchases.",
    }

    returns = [
        mock_intake,
        mock_stock,
        mock_demand,
        mock_sup,
        mock_plan,
        mock_final,
    ]

    from common.agent_runner import set_mock_task_handler
    iterator = iter(returns)
    set_mock_task_handler(lambda agent_id, desc: next(iterator))
    try:
        workflow = InventoryReplenishmentWorkflow(enable_tracing=False)
        result = workflow.run(input_data=sample_input, run_id="inv_test_001", runs_base_dir=str(tmp_path))
    finally:
        set_mock_task_handler(None)

    assert result["run_id"] == "inv_test_001"
    assert result["sku_id"] == "SKU-0001"
    assert result["recommended_order_quantity"] == 80
    assert result["available_stock"] == 15
    assert "Does NOT execute real purchases" in result["disclaimer"]


@pytest.mark.smoke
def test_inventory_live_smoke():
    """Live smoke test calling Groq API (requires valid GROQ_API_KEY)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not configured; skipping live smoke test")

    workflow = InventoryReplenishmentWorkflow(enable_tracing=True)
    result = workflow.run()
    assert result is not None
    assert "recommended_order_quantity" in result
    assert result["human_review_required"] in [True, False]
