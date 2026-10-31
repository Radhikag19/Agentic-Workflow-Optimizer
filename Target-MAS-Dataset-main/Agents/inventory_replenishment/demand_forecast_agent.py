"""
Agent 3: Sales History and Demand Forecast Agent for Inventory Replenishment Planning.

Responsibilities:
- Analyze historical sales records.
- Calculate baseline forecasts using deterministic mathematical moving averages.
- Identify volatility, demand spikes, and insufficient sales history.
- Return forecast values, assumptions, horizon, and source IDs.
- Strictly never invent forecasts when data is insufficient.
"""

from typing import Any, Dict, List, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import DemandForecastOutput
from common.metrics import MetricsTracker

AGENT_ID = "demand_forecast_agent"
ROLE = "Demand Forecasting and Sales History Specialist"
GOAL = (
    "Compute transparent, deterministic demand projections from recorded sales history, "
    "characterizing demand volatility and flagging anomalies without fabricating numbers."
)
BACKSTORY = (
    "You are a supply chain quantitative demand planner. You calculate moving averages, "
    "analyze daily sales variance, detect sudden volume spikes, and declare uncertainty "
    "when sample size or historical duration is insufficient."
)


def calculate_deterministic_forecast(daily_sales: List[Dict[str, Any]], horizon_days: int = 14) -> Dict[str, Any]:
    """
    Deterministic calculation of daily average and horizon demand.
    """
    if not daily_sales:
        return {
            "avg_daily": 0.0,
            "forecast_units": 0.0,
            "volatility": "unknown",
            "has_spikes": False,
            "sufficient_history": False,
        }

    units = [d.get("units_sold", 0) for d in daily_sales]
    total_units = sum(units)
    count = len(units)
    avg_daily = round(total_units / max(1, count), 2)
    forecast_units = round(avg_daily * horizon_days, 1)

    sufficient_history = count >= 14

    # Variance and spike detection
    max_val = max(units) if units else 0
    has_spikes = max_val > (avg_daily * 2.5) if avg_daily > 0 else False

    # Volatility categorization
    zero_sales_count = units.count(0)
    if count > 0 and (zero_sales_count / count) > 0.4:
        volatility = "intermittent"
    elif has_spikes:
        volatility = "high"
    else:
        volatility = "medium" if avg_daily > 10 else "low"

    return {
        "avg_daily": avg_daily,
        "forecast_units": forecast_units,
        "volatility": volatility,
        "has_spikes": has_spikes,
        "sufficient_history": sufficient_history,
    }


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Sales History and Demand Forecast Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_demand_forecast(
    intake_data: Dict[str, Any],
    sales_data: Dict[str, Any],
    horizon_days: int = 14,
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Sales History and Demand Forecast Agent."""
    active_agent = agent or create_agent()

    sku_id = intake_data.get("sku_id", "SKU-UNKNOWN")
    daily_sales = sales_data.get("daily_sales", [])
    history_days = len(daily_sales)

    # Compute deterministic calculations
    det = calculate_deterministic_forecast(daily_sales, horizon_days=horizon_days)

    task_desc = f"""
SKU ID: {sku_id}
Forecast Horizon: {horizon_days} days
Total Recorded History: {history_days} days
Daily Sales Samples: {daily_sales[-10:] if len(daily_sales) > 10 else daily_sales}

Deterministic Moving Average Calculation:
- Average Daily Sales: {det['avg_daily']} units/day
- Forecast Demand over {horizon_days} days: {det['forecast_units']} units
- Volatility: {det['volatility']}
- Demand Spikes Detected: {det['has_spikes']}
- Sufficient Historical Data: {det['sufficient_history']}

Structure the forecast output with documented assumptions:
Return structured JSON matching:
{{
  "sku_id": "{sku_id}",
  "forecast_horizon_days": {horizon_days},
  "historical_avg_daily_demand": {det['avg_daily']},
  "forecast_demand_units": {det['forecast_units']},
  "volatility_metric": "{det['volatility']}",
  "demand_spikes_detected": {str(det['has_spikes']).lower()},
  "has_sufficient_history": {str(det['sufficient_history']).lower()},
  "source_record_ids": ["SALES-{sku_id}"],
  "assumptions": [
    "Baseline forecast computed using {history_days}-day arithmetic moving average",
    "Demand volatility classified as {det['volatility']}"
  ]
}}
"""
    expected_output = "Valid JSON matching DemandForecastOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=DemandForecastOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
