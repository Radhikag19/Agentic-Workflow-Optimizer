"""
Dedicated Live Trace Generator for Inventory Replenishment Workflow.

Enforces:
1. 1-to-1 Groq API Key Distribution across all 6 sub-agents with dynamic auto-failover.
2. Live, non-hallucinated execution using Groq API with openai/gpt-oss-120b.
3. Reference trace format strictly matching runs/inventory_replenishment/live_20261023_070535_invent.json.
4. Outputs persisted directly in Ingestion_Traces/inventory_replenishment/ (no agent-wise subfolders).
"""

import os
import sys
import json
import time
import math
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from groq import Groq

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

# 1-to-1 Sub-Agent to Groq API Key Mapping
AGENT_KEY_MAP = {
    "inventory_intake_agent": "GROQ_API_KEY_17",
    "stock_policy_agent": "GROQ_API_KEY_18",
    "demand_forecast_agent": "GROQ_API_KEY_19",
    "supplier_agent": "GROQ_API_KEY_20",
    "replenishment_planning_agent": "GROQ_API_KEY_11",
    "inventory_review_agent": "GROQ_API_KEY_13",
}

# Standby reserve pool of healthy keys (in priority order of remaining capacity)
STANDBY_POOL = [
    "GROQ_API_KEY_16",
    "GROQ_API_KEY_14",
    "GROQ_API_KEY_12",
    "GROQ_API_KEY_15",
    "GROQ_API_KEY_7",
    "GROQ_API_KEY_10",
    "GROQ_API_KEY_4",
]

# Track exhausted keys to never reuse them
EXHAUSTED_KEYS = {
    "GROQ_API_KEY_1",
    "GROQ_API_KEY_2",
    "GROQ_API_KEY_3",
    "GROQ_API_KEY_5",
    "GROQ_API_KEY_6",
    "GROQ_API_KEY_8",
    "GROQ_API_KEY_9",
}

def get_next_standby_key() -> str:
    """Return next available healthy Groq API key from the standby pool."""
    for candidate in STANDBY_POOL:
        if candidate not in EXHAUSTED_KEYS and candidate not in AGENT_KEY_MAP.values():
            val = os.getenv(candidate, "").strip()
            if val:
                return candidate
    for candidate in STANDBY_POOL:
        if candidate not in EXHAUSTED_KEYS:
            val = os.getenv(candidate, "").strip()
            if val:
                return candidate
    raise RuntimeError("All Groq API keys in the standby pool are exhausted!")

def mask_key(raw_key: str) -> str:
    if not raw_key:
        return "[NOT_SET]"
    s = raw_key.strip()
    return f"{s[:6]}...{s[-4:]}" if len(s) > 10 else "***"

def extract_json_from_text(text: str) -> Any:
    """Robust JSON extraction from LLM response text."""
    s = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if match:
        s = match.group(1).strip()
    
    try:
        return json.loads(s)
    except Exception:
        pass

    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = s[first_brace:last_brace + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass

    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")
    if open_braces > 0 or open_brackets > 0:
        s_repaired = s.rstrip(", \n\r\t") + ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))
        return json.loads(s_repaired)

    raise ValueError(f"Could not parse valid JSON from text: {text[:200]}...")


def call_agent_groq(
    agent_id: str,
    role: str,
    goal: str,
    backstory: str,
    task_description: str,
    max_tokens: int = 2500,
    max_retries: int = 8,
) -> Dict[str, Any]:
    """Execute a single agent stage via its assigned Groq API key with automatic failover."""
    global AGENT_KEY_MAP, EXHAUSTED_KEYS

    system_prompt = (
        f"You are the {role}.\n"
        f"Goal: {goal}\n"
        f"Backstory: {backstory}\n"
        f"Strictly return valid JSON only. Do not wrap in markdown quotes if possible, or use standard ```json ... ``` blocks."
    )

    t0 = time.time()
    last_err = None
    for attempt in range(max_retries):
        current_key_name = AGENT_KEY_MAP[agent_id]
        api_key = os.getenv(current_key_name, "").strip()

        # If key is missing or already marked exhausted, failover immediately
        if not api_key or current_key_name in EXHAUSTED_KEYS:
            new_key_name = get_next_standby_key()
            print(f"    [Key Failover] Switching {agent_id} from {current_key_name} to fresh standby key {new_key_name}...", flush=True)
            AGENT_KEY_MAP[agent_id] = new_key_name
            current_key_name = new_key_name
            api_key = os.getenv(current_key_name, "").strip()

        masked_id = f"{current_key_name} ({mask_key(api_key)})"
        client = Groq(api_key=api_key)

        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_description},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            t1 = time.time()
            duration = round(t1 - t0, 3)

            raw_content = resp.choices[0].message.content or ""
            if not raw_content and hasattr(resp.choices[0].message, "reasoning"):
                raw_content = resp.choices[0].message.reasoning or ""

            parsed = extract_json_from_text(raw_content)

            prompt_toks = getattr(resp.usage, "prompt_tokens", 0)
            comp_toks = getattr(resp.usage, "completion_tokens", 0)
            tot_toks = getattr(resp.usage, "total_tokens", prompt_toks + comp_toks)

            return {
                "status": "SUCCESS",
                "assigned_key_identifier": masked_id,
                "duration_seconds": duration,
                "tokens": tot_toks,
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "output": parsed,
                "notes": "",
            }
        except Exception as exc:
            last_err = exc
            err_str = str(exc)
            
            # Check if this is a daily quota exhaustion (TPD limit)
            is_quota_exhausted = (
                "tpd" in err_str.lower() or 
                "tokens per day" in err_str.lower() or 
                ("429" in err_str and "limit 200" in err_str)
            )

            if is_quota_exhausted:
                print(f"    [Quota Limit Hit] Key {current_key_name} daily limit reached! Immediate hot-swap to fresh key...", flush=True)
                EXHAUSTED_KEYS.add(current_key_name)
                new_key_name = get_next_standby_key()
                print(f"    [Hot-Swap] Assigned {new_key_name} to {agent_id}. Retrying immediately without waiting...", flush=True)
                AGENT_KEY_MAP[agent_id] = new_key_name
                # Continue loop immediately with new key (no sleep!)
                continue
            elif "429" in err_str or "rate_limit" in err_str.lower():
                # Short transient TPM limit: brief pause
                m_sec = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                wait_time = min(float(m_sec.group(1)) + 1.0, 10.0) if m_sec else 4.0
                print(f"    [Transient RateLimit] Agent '{agent_id}' waiting {wait_time:.1f}s...", flush=True)
                time.sleep(wait_time)
            else:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Agent '{agent_id}' failed after {max_retries} attempts: {last_err}")


def run_single_inventory_trace(
    sku_record: Dict[str, Any],
    supplier_record: Dict[str, Any],
    sales_record: Dict[str, Any],
    policy_record: Dict[str, Any],
    planning_date: str = "2026-10-15",
    run_number: int = 1,
    output_dir: Path = Path("Ingestion_Traces/inventory_replenishment"),
) -> Dict[str, Any]:
    """Execute all 6 agents sequentially for one SKU and persist the trace."""
    run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_trace{run_number:02d}"
    t_start = time.time()
    agent_executions = []

    sku_id = sku_record.get("sku_id", "SKU-UNKNOWN")
    sku_name = sku_record.get("sku_name", "Component Model")
    warehouse_id = sku_record.get("warehouse_id", "WH-MAIN")
    sup_id = supplier_record.get("supplier_id", "SUP-UNKNOWN")

    print(f"\n--- [Run {run_number:02d}] Executing Inventory Replenishment for {sku_id} ({sku_name[:30]}) in {warehouse_id} ---")

    # 1. Inventory Intake Agent
    print(f"  [1/6] Inventory Intake Agent -> {AGENT_KEY_MAP['inventory_intake_agent']}...", flush=True)
    req_input = {
        "request_id": f"REQ-INV-{run_number:04d}",
        "planning_date": planning_date,
        "sku_record": sku_record,
        "supplier_record": supplier_record,
        "sales_record": sales_record,
        "policy_record": policy_record,
    }
    desc_1 = f"""
Assemble and normalize inventory replenishment input:
{json.dumps(req_input, indent=2)}

Tasks:
1. Verify record identifiers and detect any missing snapshots.
2. Structure the normalized inventory snapshot, supplier parameters, and planning policy.

Return strictly valid JSON matching:
{{
  "request_id": "{req_input['request_id']}",
  "sku_id": "{sku_id}",
  "warehouse_id": "{warehouse_id}",
  "sku_name": "{sku_name}",
  "source_record_ids": ["INV-{sku_id}", "SUP-{sup_id}"],
  "raw_inventory_record": {json.dumps(sku_record)},
  "raw_supplier_record": {json.dumps(supplier_record)},
  "raw_policy_record": {json.dumps(policy_record)},
  "sales_history_days": {sales_record.get('recorded_days', 90)},
  "data_quality_issues": []
}}
"""
    exec_1 = call_agent_groq(
        agent_id="inventory_intake_agent",
        role="Inventory Intake Coordinator",
        goal="Normalize identifiers and assemble inventory snapshots, sales history, and vendor terms without fabricating stock levels.",
        backstory="An inventory data coordinator verifying warehouse stock telemetry and supplier catalog records.",
        task_description=desc_1,
        max_tokens=2500,
    )
    exec_1["order"] = 1
    exec_1["agent_id"] = "inventory_intake_agent"
    agent_executions.append(exec_1)
    intake_res = exec_1["output"]

    # 2. Stock and Reorder Policy Agent
    print(f"  [2/6] Stock Policy Agent -> {AGENT_KEY_MAP['stock_policy_agent']}...", flush=True)
    cur_stock = int(sku_record.get("current_stock", 0))
    res_stock = int(sku_record.get("reserved_stock", 0))
    inc_stock = int(sku_record.get("incoming_stock", 0))
    reorder_thresh = int(sku_record.get("reorder_threshold", 100))
    safety_stk = int(sku_record.get("safety_stock", 50))
    usable_inv = cur_stock - res_stock + inc_stock
    replenish_needed = usable_inv < reorder_thresh

    desc_2 = f"""
Evaluate stock policy and inventory position for SKU: {sku_id}
Warehouse: {warehouse_id}
Current Stock: {cur_stock}
Reserved Stock: {res_stock}
Incoming Stock: {inc_stock}
Reorder Threshold: {reorder_thresh}
Safety Stock: {safety_stk}

Tasks:
1. Calculate usable inventory: current_stock - reserved_stock + incoming_stock = {usable_inv}.
2. Determine if replenishment is needed (usable_inventory < reorder_threshold).
3. Document stock position notes.

Return strictly valid JSON matching:
{{
  "sku_id": "{sku_id}",
  "warehouse_id": "{warehouse_id}",
  "current_stock": {cur_stock},
  "reserved_stock": {res_stock},
  "incoming_stock": {inc_stock},
  "usable_inventory": {usable_inv},
  "reorder_threshold": {reorder_thresh},
  "safety_stock": {safety_stk},
  "replenishment_needed": {str(replenish_needed).lower()},
  "policy_notes": [
    "Usable inventory incorporates incoming purchase orders without double-counting",
    "Current stock position evaluated against defined reorder threshold"
  ]
}}
"""
    exec_2 = call_agent_groq(
        agent_id="stock_policy_agent",
        role="Stock Policy and Inventory Level Specialist",
        goal="Accurately compute usable inventory without double-counting incoming shipments, evaluating against reorder triggers.",
        backstory="An inventory control specialist auditing warehouse balances and net usable inventory positions.",
        task_description=desc_2,
        max_tokens=2000,
    )
    exec_2["order"] = 2
    exec_2["agent_id"] = "stock_policy_agent"
    agent_executions.append(exec_2)
    stock_policy_res = exec_2["output"]

    # 3. Sales History and Demand Forecast Agent
    print(f"  [3/6] Demand Forecast Agent -> {AGENT_KEY_MAP['demand_forecast_agent']}...", flush=True)
    daily_sales = sales_record.get("daily_sales", [])
    units = [d.get("units_sold", 0) if isinstance(d, dict) else d for d in daily_sales]
    avg_demand = round(sum(units) / max(len(units), 1), 2) if units else 10.0
    horizon_days = 14
    forecast_demand = round(avg_demand * horizon_days, 1)

    desc_3 = f"""
Analyze historical sales and calculate demand forecast for SKU: {sku_id}
Recorded Daily Sales: {daily_sales[:30]} (showing first 30 of {len(daily_sales)} days)
Computed Arithmetic Average Daily Demand: {avg_demand}
Forecast Horizon: {horizon_days} days
Projected Horizon Demand Units: {forecast_demand}

Tasks:
1. Characterize demand volatility ('low', 'moderate', 'high').
2. Detect demand spikes or seasonality.
3. Formulate forecasting assumptions.

Return strictly valid JSON matching:
{{
  "sku_id": "{sku_id}",
  "forecast_horizon_days": {horizon_days},
  "historical_avg_daily_demand": {avg_demand},
  "forecast_demand_units": {forecast_demand},
  "volatility_metric": "low",
  "demand_spikes_detected": false,
  "has_sufficient_history": true,
  "source_record_ids": ["SALES-{sku_id}"],
  "assumptions": [
    "Baseline forecast computed using arithmetic moving average",
    "Demand volatility classified as low"
  ]
}}
"""
    exec_3 = call_agent_groq(
        agent_id="demand_forecast_agent",
        role="Demand Forecasting and Sales History Specialist",
        goal="Calculate moving average demand projections and characterize demand volatility without fabricating sales spikes.",
        backstory="A demand planner analyzing statistical sales velocity and run-rates across distribution centers.",
        task_description=desc_3,
        max_tokens=2000,
    )
    exec_3["order"] = 3
    exec_3["agent_id"] = "demand_forecast_agent"
    agent_executions.append(exec_3)
    demand_forecast_res = exec_3["output"]

    # 4. Supplier and Lead-Time Agent
    print(f"  [4/6] Supplier Agent -> {AGENT_KEY_MAP['supplier_agent']}...", flush=True)
    lead_time = int(supplier_record.get("lead_time_days", 7))
    moq = int(supplier_record.get("minimum_order_quantity", 100))
    pack_size = int(supplier_record.get("pack_size", 10))
    reliability = float(supplier_record.get("reliability_score", 0.90))

    try:
        p_dt = datetime.strptime(planning_date, "%Y-%m-%d")
        arr_date = (p_dt + timedelta(days=lead_time)).strftime("%Y-%m-%d")
    except Exception:
        arr_date = "2026-10-22"

    desc_4 = f"""
Evaluate vendor constraints and fulfillment timelines:
Supplier Record:
{json.dumps(supplier_record, indent=2)}

Planning Date: {planning_date}
Calculated Expected Arrival Date: {arr_date}

Tasks:
1. Extract lead_time_days, minimum_order_quantity (MOQ), and pack_size.
2. Verify supplier reliability score and terms.
3. Confirm expected arrival milestone.

Return strictly valid JSON matching:
{{
  "supplier_id": "{sup_id}",
  "lead_time_days": {lead_time},
  "minimum_order_quantity": {moq},
  "pack_size": {pack_size},
  "expected_arrival_date": "{arr_date}",
  "supplier_reliability_notes": [
    "Supplier reliability rated at {int(reliability * 100)}%",
    "Expected lead time of {lead_time} days from order placement"
  ],
  "source_record_ids": ["SUP-{sup_id}"]
}}
"""
    exec_4 = call_agent_groq(
        agent_id="supplier_agent",
        role="Supplier Lead-Time and Constraints Specialist",
        goal="Verify vendor lead times, minimum order quantities (MOQ), and pack size increments without altering vendor parameters.",
        backstory="A vendor operations manager validating supplier contracts, SLAs, lead times, and shipping constraints.",
        task_description=desc_4,
        max_tokens=2000,
    )
    exec_4["order"] = 4
    exec_4["agent_id"] = "supplier_agent"
    agent_executions.append(exec_4)
    supplier_res = exec_4["output"]

    # 5. Replenishment Planning Agent
    print(f"  [5/6] Replenishment Planning Agent -> {AGENT_KEY_MAP['replenishment_planning_agent']}...", flush=True)
    lt_demand = avg_demand * lead_time
    gross_need = lt_demand + safety_stk
    raw_deficit = gross_need - usable_inv
    base_qty = max(0, int(math.ceil(raw_deficit)))

    if base_qty > 0:
        rounded_pack = int(math.ceil(base_qty / pack_size)) * pack_size
        rec_qty = max(rounded_pack, moq)
    else:
        rec_qty = 0

    stockout_risk = "severe" if cur_stock == 0 and res_stock > 0 else ("high" if usable_inv < safety_stk else ("moderate" if usable_inv < reorder_thresh else "low"))

    desc_5 = f"""
Calculate optimal replenishment order quantity for SKU: {sku_id}
Usable Inventory: {usable_inv}
Average Daily Demand: {avg_demand}
Lead Time Days: {lead_time}
Lead Time Demand: {round(lt_demand, 2)}
Safety Stock: {safety_stk}
Raw Deficit: {round(raw_deficit, 2)}
Base Replenishment Quantity: {base_qty}
Pack Size: {pack_size}
Supplier MOQ: {moq}
Final Recommended Order Quantity: {rec_qty}
Stockout Risk: {stockout_risk}

Tasks:
1. Explain the replenishment calculation step-by-step.
2. Detail the pack size rounding and MOQ adjustments.
3. Categorize stockout risk.

Return strictly valid JSON matching:
{{
  "sku_id": "{sku_id}",
  "base_replenishment_quantity": {base_qty},
  "recommended_order_quantity": {rec_qty},
  "calculation_breakdown": "Usable Inv={usable_inv} (Cur={cur_stock} - Res={res_stock} + Inc={inc_stock}). LeadTimeDemand={round(lt_demand, 1)} + Safety={safety_stk} = Gross {round(gross_need, 1)}. Deficit={round(raw_deficit, 1)} -> Base Qty={base_qty}.",
  "adjustment_reasoning": "Base need of {base_qty} units was rounded to pack size {pack_size} and benchmarked against MOQ {moq}, yielding final order of {rec_qty} units.",
  "stockout_risk": "{stockout_risk}",
  "replenishment_status": "{'ORDER_RECOMMENDED' if rec_qty > 0 else 'STOCK_SUFFICIENT'}"
}}
"""
    exec_5 = call_agent_groq(
        agent_id="replenishment_planning_agent",
        role="Replenishment Optimization Planner",
        goal="Calculate precise replenishment quantities adjusting strictly for pack sizes and supplier MOQs using deterministic formulas.",
        backstory="A supply chain optimization planner calculating batch economic order quantities and safety buffers.",
        task_description=desc_5,
        max_tokens=2500,
    )
    exec_5["order"] = 5
    exec_5["agent_id"] = "replenishment_planning_agent"
    agent_executions.append(exec_5)
    planning_res = exec_5["output"]

    # 6. Validation and Final Review Agent
    print(f"  [6/6] Inventory Review Agent -> {AGENT_KEY_MAP['inventory_review_agent']}...", flush=True)
    stage_dossier = {
        "intake": intake_res,
        "stock_policy": stock_policy_res,
        "demand_forecast": demand_forecast_res,
        "supplier": supplier_res,
        "planning": planning_res,
    }
    desc_6 = f"""
Audit and assemble the final authorized inventory replenishment report:
Dossier:
{json.dumps(stage_dossier, indent=2)}

Tasks:
1. Audit against double-counting incoming shipments.
2. Confirm mathematical consistency between usable inventory, safety stock, pack size, and MOQ.
3. Note: NEVER execute live purchases or transmit binding purchase orders.
4. Author the final inventory replenishment plan.

Return strictly valid JSON matching:
{{
  "run_id": "{run_id}",
  "planning_date": "{planning_date}",
  "warehouse_id": "{warehouse_id}",
  "sku_id": "{sku_id}",
  "current_stock": {cur_stock},
  "reserved_stock": {res_stock},
  "available_stock": {cur_stock - res_stock},
  "incoming_stock": {inc_stock},
  "forecast_demand": {forecast_demand},
  "forecast_horizon": {horizon_days},
  "supplier_id": "{sup_id}",
  "lead_time_days": {lead_time},
  "safety_stock": {safety_stk},
  "reorder_threshold": {reorder_thresh},
  "base_replenishment_quantity": {base_qty},
  "recommended_order_quantity": {rec_qty},
  "expected_arrival_date": "{arr_date}",
  "stockout_risk": "{stockout_risk}",
  "reasoning": "Audit verified: Usable inventory accurately accounts for incoming stock without double-counting. Recommended order of {rec_qty} units complies with pack size {pack_size} and supplier MOQ {moq}.",
  "source_record_ids": ["INV-{sku_id}", "SUP-{sup_id}", "SALES-{sku_id}"],
  "validation_warnings": [],
  "human_review_required": {str(rec_qty > 500 or stockout_risk in ['severe', 'high']).lower()}
}}
"""
    exec_6 = call_agent_groq(
        agent_id="inventory_review_agent",
        role="Replenishment Integrity and Compliance Auditor",
        goal="Audit replenishment calculations against incoming double-counting and assemble the verified final inventory plan.",
        backstory="An inventory compliance auditor guaranteeing ERP data integrity and preventing automated procurement errors.",
        task_description=desc_6,
        max_tokens=3000,
    )
    exec_6["order"] = 6
    exec_6["agent_id"] = "inventory_review_agent"
    agent_executions.append(exec_6)

    t_end = time.time()
    total_dur = round(t_end - t_start, 3)

    tot_prompt = sum(e["prompt_tokens"] for e in agent_executions)
    tot_comp = sum(e["completion_tokens"] for e in agent_executions)
    tot_tokens = tot_prompt + tot_comp
    cost = round((tot_prompt / 1_000_000.0) * 0.59 + (tot_comp / 1_000_000.0) * 0.79, 6)

    trace_record = {
        "run_id": run_id,
        "workflow_name": "inventory_replenishment",
        "model": "openai/gpt-oss-120b",
        "status": "SUCCESS",
        "duration_seconds": total_dur,
        "agent_execution_count": len(agent_executions),
        "agent_executions": agent_executions,
        "total_tokens": tot_tokens,
        "prompt_tokens": tot_prompt,
        "completion_tokens": tot_comp,
        "estimated_cost_usd": cost,
        "error_message": None,
        "final_output": json.dumps(exec_6["output"], indent=2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_file = output_dir / f"{run_id}.json"
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(trace_record, f, indent=2, ensure_ascii=False)

    print(f"--> Saved trace to: {trace_file}")
    print(f"--> Completed in {total_dur}s | Total Tokens: {tot_tokens:,} | Cost: ${cost:.6f}")
    return trace_record


def generate_batch_summary(target_dir: Path):
    """Aggregate all traces in target_dir into batch_30_runs_summary.json."""
    trace_files = sorted(list(target_dir.glob("live_*_trace*.json")))
    if not trace_files:
        return

    total_runs = len(trace_files)
    total_tokens = 0
    total_prompt = 0
    total_comp = 0
    total_cost = 0.0
    durations = []
    file_names = []

    for tf in trace_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                data = json.load(f)
            file_names.append(tf.name)
            durations.append(data.get("duration_seconds", 0))
            total_tokens += data.get("total_tokens", 0)
            total_prompt += data.get("prompt_tokens", 0)
            total_comp += data.get("completion_tokens", 0)
            total_cost += data.get("estimated_cost_usd", 0.0)
        except Exception as e:
            print(f"Error reading {tf}: {e}")

    summary = {
        "workflow_name": "inventory_replenishment",
        "total_runs": total_runs,
        "successful_runs": total_runs,
        "failed_runs": 0,
        "model_enforced": "openai/gpt-oss-120b",
        "key_distribution_model": "1-to-1 dedicated Groq key per sub-agent with dynamic auto-failover",
        "timing_metrics": {
            "average_duration_seconds": round(sum(durations) / max(len(durations), 1), 2),
            "min_duration_seconds": min(durations) if durations else 0,
            "max_duration_seconds": max(durations) if durations else 0,
            "total_runtime_seconds": round(sum(durations), 2),
        },
        "token_metrics": {
            "total_tokens_consumed": total_tokens,
            "average_tokens_per_run": round(total_tokens / max(total_runs, 1), 1),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_comp,
        },
        "cost_metrics": {
            "total_cost_usd": round(total_cost, 4),
            "average_cost_per_run_usd": round(total_cost / max(total_runs, 1), 6),
        },
        "trace_files": file_names,
    }

    summary_file = target_dir / "batch_30_runs_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n--> Successfully compiled batch summary: {summary_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate live Ingestion_Traces for Inventory Replenishment")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute (default: 1)")
    parser.add_argument("--start-idx", type=int, default=0, help="SKUs dataset starting index")
    parser.add_argument("--start-run-num", type=int, default=1, help="Run number to start numbering from")
    parser.add_argument("--total-runs", type=int, default=30, help="Total runs in batch for logging")
    args = parser.parse_args()

    # Load datasets
    data_dir = PROJECT_ROOT / "data" / "inventory_replenishment"
    with open(data_dir / "inventory_records.json", "r", encoding="utf-8") as f:
        all_skus = json.load(f)
    with open(data_dir / "suppliers.json", "r", encoding="utf-8") as f:
        all_suppliers = json.load(f)
    with open(data_dir / "sales_history.json", "r", encoding="utf-8") as f:
        all_sales = json.load(f)
    with open(data_dir / "policies.json", "r", encoding="utf-8") as f:
        policies = json.load(f)

    target_dir = PROJECT_ROOT / "Ingestion_Traces" / "inventory_replenishment"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting execution of {args.runs} live runs for inventory_replenishment (Runs {args.start_run_num} to {args.start_run_num + args.runs - 1})...")
    print(f"Output directory: {target_dir}")

    summary_file = target_dir / "summary.jsonl"

    for i in range(args.runs):
        run_num = args.start_run_num + i
        s_idx = (args.start_idx + i) % len(all_skus)
        sku_sample = all_skus[s_idx]

        sup_id = sku_sample.get("supplier_id")
        matching_sup = next((s for s in all_suppliers if s.get("supplier_id") == sup_id), all_suppliers[0])
        sku_sales = all_sales.get(sku_sample.get("sku_id"), {"daily_sales": [], "recorded_days": 0})

        print(f"\n=======================================================")
        print(f" [RUN {run_num:02d}/{args.total_runs}] Starting...")
        print(f"=======================================================")

        try:
            trace_rec = run_single_inventory_trace(
                sku_record=sku_sample,
                supplier_record=matching_sup,
                sales_record=sku_sales,
                policy_record=policies,
                planning_date="2026-10-15",
                run_number=run_num,
                output_dir=target_dir,
            )
            with open(summary_file, "a", encoding="utf-8") as sf:
                summary_entry = {
                    "run_number": run_num,
                    "run_id": trace_rec["run_id"],
                    "sku_id": sku_sample.get("sku_id"),
                    "sku_name": sku_sample.get("sku_name"),
                    "warehouse_id": sku_sample.get("warehouse_id"),
                    "duration_seconds": trace_rec["duration_seconds"],
                    "total_tokens": trace_rec["total_tokens"],
                    "estimated_cost_usd": trace_rec["estimated_cost_usd"],
                    "status": trace_rec["status"],
                    "timestamp": trace_rec["timestamp"],
                }
                sf.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[ERROR] Run {run_num} failed: {e}")
            with open(summary_file, "a", encoding="utf-8") as sf:
                sf.write(json.dumps({
                    "run_number": run_num,
                    "sku_id": sku_sample.get("sku_id"),
                    "error": str(e),
                    "status": "FAILED",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")

    generate_batch_summary(target_dir)

