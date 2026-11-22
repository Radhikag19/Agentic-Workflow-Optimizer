"""
MCP-Based Agent Context Builder for JEV Model Selection.

Retrieves workflow and agent evidence exclusively through TraceProfiler MCP Evidence Tools
(mcp_server.tools) and constructs a normalized, compact, agent-specific evidence context.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from mcp_server import tools as mcp_tools

logger = logging.getLogger("optimization_agent.model_selection.mcp_context_builder")


def _first_present(mapping: Any, *keys: str, default: Any = None) -> Any:
    """Helper to safely retrieve the first present, non-null key from a dict."""
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def compact_text(value: Any, max_chars: int = 3500) -> Optional[str]:
    """Truncate text or serialized structures cleanly with a truncation marker."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    val_str = str(value)
    if len(val_str) > max_chars:
        return val_str[:max_chars] + " ...[truncated]"
    return val_str


def calculate_workflow_token_reference(
    workflow_id: str,
    evidence_dir: Optional[str] = None,
) -> Tuple[float, float, str]:
    """
    Calculate relative one-third and two-third quantile boundaries across all agents in the workflow.

    Uses mcp_tools.get_token_metrics(workflow_id).

    Returns:
        (token_low_cut, token_high_cut, scope_description)
    """
    tok_data = mcp_tools.get_token_metrics(workflow_id, evidence_dir=evidence_dir)
    if not tok_data or "error" in tok_data:
        raise ValueError(f"MCP get_token_metrics failed for workflow '{workflow_id}': {tok_data.get('error')}")

    agent_metrics = tok_data.get("agent_token_metrics", {})
    all_mean_tokens: List[float] = []

    for aid, m in agent_metrics.items():
        if isinstance(m, dict):
            val = _first_present(m, "mean_tokens", "mean_total_tokens", "total_tokens")
            if val is not None:
                try:
                    all_mean_tokens.append(float(val))
                except (ValueError, TypeError):
                    pass

    if not all_mean_tokens:
        raise ValueError(
            f"No valid per-agent mean token metrics found via MCP for workflow '{workflow_id}'."
        )

    series = pd.Series(all_mean_tokens)
    low_cut = float(series.quantile(1 / 3))
    high_cut = float(series.quantile(2 / 3))
    scope = f"Per-agent mean token consumption within workflow {workflow_id} ({len(all_mean_tokens)} agents)"

    return low_cut, high_cut, scope


def build_agent_context_from_mcp(
    workflow_id: str,
    agent_id: str,
    evidence_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the compact agent-specific evidence context using MCP tools.

    MCP Tools queried:
      1. get_workflow_overview
      2. get_agent_profiles (with agent_id filter)
      3. get_prompt_metrics (with full_prompt=True)
      4. get_latency_metrics (for agent_id)
      5. get_token_metrics (for agent_id and workflow-wide quantiles)
      6. get_execution_evidence (for sample outputs and execution orders)
      7. get_workflow_graph (for execution order resolution)

    Returns:
        Standardized agent_context dict matching the Colab decision schema.
    """
    # 1. Workflow overview
    overview = mcp_tools.get_workflow_overview(workflow_id, evidence_dir=evidence_dir)
    if not overview or "error" in overview:
        raise ValueError(f"MCP get_workflow_overview failed: {overview.get('error', 'empty overview')}")

    workflow_name = overview.get("workflow_name") or workflow_id
    workflow_desc = overview.get("description")
    model_reconciliation = overview.get("model_reconciliation") or {}
    baseline_model = model_reconciliation.get("declared_model") or "openai/gpt-oss-120b"
    total_declared_agents = overview.get("total_declared_agents") or overview.get("total_runtime_agents") or 0

    # 2. Agent profiles
    profiles_res = mcp_tools.get_agent_profiles(workflow_id, agent_ids=[agent_id], evidence_dir=evidence_dir)
    if not profiles_res or "error" in profiles_res or not profiles_res.get("profiles"):
        raise ValueError(
            f"Agent '{agent_id}' profile not found in MCP evidence for workflow '{workflow_id}'. "
            f"Error: {profiles_res.get('error', 'No profiles returned')}"
        )
    profile = profiles_res["profiles"][0]

    # Baseline model fallback from agent profile model configuration
    model_cfg = profile.get("model_configuration") or {}
    if not baseline_model and model_cfg.get("declared_model"):
        baseline_model = model_cfg.get("declared_model")

    # 3. Tasks and tools
    raw_tasks = profile.get("tasks", [])
    task_summaries = []
    task_tools = []
    if isinstance(raw_tasks, list):
        for task in raw_tasks[:3]:
            if isinstance(task, dict):
                t_tools = task.get("tools") or []
                if isinstance(t_tools, list):
                    task_tools.extend(t_tools)
                elif t_tools:
                    task_tools.append(t_tools)

                task_summaries.append({
                    "task_name": _first_present(task, "task_name", "name", default=f"{agent_id}_task"),
                    "description": compact_text(_first_present(task, "description", "task"), 3000),
                    "expected_output": compact_text(_first_present(task, "expected_output", "output"), 1200),
                    "tools": t_tools,
                })

    # 4. Prompt metrics
    prompt_res = mcp_tools.get_prompt_metrics(
        workflow_id,
        agent_ids=[agent_id],
        full_prompt=True,
        evidence_dir=evidence_dir,
    )
    prompt_text: List[str] = []
    if prompt_res and "prompt_metrics" in prompt_res:
        for p in prompt_res.get("prompt_metrics", []):
            if isinstance(p, dict):
                raw_text = p.get("prompt_text") or p.get("sample_prompt_text") or ""
                if raw_text:
                    prompt_text.append(compact_text(raw_text, 3000))

    # 5. Token metrics (specific agent)
    tok_res = mcp_tools.get_token_metrics(workflow_id, agent_ids=[agent_id], evidence_dir=evidence_dir)
    agent_tok_map = tok_res.get("agent_token_metrics", {}) if tok_res else {}
    agent_tok = agent_tok_map.get(agent_id, {}) if isinstance(agent_tok_map, dict) else {}

    mean_total_tokens = _first_present(agent_tok, "mean_tokens", "mean_total_tokens")
    mean_prompt_tokens = _first_present(agent_tok, "mean_prompt_tokens", "avg_prompt_tokens")
    mean_completion_tokens = _first_present(agent_tok, "mean_completion_tokens", "avg_completion_tokens")
    min_total_tokens = _first_present(agent_tok, "min_tokens", "min_total_tokens")
    max_total_tokens = _first_present(agent_tok, "max_tokens", "max_total_tokens")

    # 6. Latency metrics (specific agent)
    lat_res = mcp_tools.get_latency_metrics(workflow_id, agent_ids=[agent_id], evidence_dir=evidence_dir)
    agent_lat_map = lat_res.get("agent_latency_metrics", {}) if lat_res else {}
    agent_lat = agent_lat_map.get(agent_id, {}) if isinstance(agent_lat_map, dict) else {}

    mean_latency = _first_present(agent_lat, "mean_seconds", "mean_latency_seconds")
    median_latency = _first_present(agent_lat, "median_seconds", "median_latency_seconds")
    max_latency = _first_present(agent_lat, "max_seconds", "max_latency_seconds")

    # 7. Execution evidence (sample outputs and observed execution order)
    exec_ev = mcp_tools.get_execution_evidence(workflow_id, agent_id=agent_id, evidence_dir=evidence_dir)
    sample_outputs = exec_ev.get("sample_outputs") if isinstance(exec_ev, dict) else None
    rep_output = compact_text(sample_outputs, 2200)

    # 8. Execution order position
    execution_order_pos = None
    observed_orders = exec_ev.get("observed_execution_orders") if isinstance(exec_ev, dict) else None
    if observed_orders and isinstance(observed_orders, list) and len(observed_orders) > 0:
        execution_order_pos = observed_orders[0]
    else:
        graph = mcp_tools.get_workflow_graph(workflow_id, evidence_dir=evidence_dir)
        nodes = graph.get("nodes", [])
        for idx, n in enumerate(nodes):
            nid = n.get("id") or n.get("agent_id")
            if nid == agent_id:
                execution_order_pos = idx + 1
                break

    # 9. Token reference quantile boundaries (workflow-wide)
    token_low_cut, token_high_cut, token_scope = calculate_workflow_token_reference(
        workflow_id, evidence_dir=evidence_dir
    )

    registered_tools = profile.get("registered_tools") or []
    if not registered_tools and task_tools:
        registered_tools = sorted(list(set(task_tools)))

    # Assemble normalized agent context
    context: Dict[str, Any] = {
        "workflow": {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "workflow_description": workflow_desc,
            "baseline_model": baseline_model,
            "agent_count": total_declared_agents,
            "execution_order_position": execution_order_pos,
        },
        "agent_definition": {
            "agent_id": agent_id,
            "name": profile.get("agent_id"),
            "role": profile.get("role"),
            "responsibilities": profile.get("declared_responsibilities"),
            "goal": compact_text(profile.get("goal"), 1800),
            "backstory": compact_text(profile.get("backstory"), 1500),
            "declared_inputs": profile.get("declared_inputs", []),
            "declared_outputs": profile.get("declared_outputs", []),
            "dependencies": profile.get("dependencies", []),
            "registered_tools": registered_tools,
            "helper_functions": profile.get("helper_functions", []),
        },
        "task_information": task_summaries,
        "prompt_information": prompt_text,
        "execution_metrics": {
            "invocation_count": overview.get("total_runs_recorded") or _first_present(agent_tok, "observation_count"),
            "success_count": overview.get("successful_runs"),
            "failure_count": overview.get("failed_runs"),
            "error_count": 0,
            "mean_total_tokens": mean_total_tokens,
            "mean_prompt_tokens": mean_prompt_tokens,
            "mean_completion_tokens": mean_completion_tokens,
            "min_total_tokens": min_total_tokens,
            "max_total_tokens": max_total_tokens,
            "mean_latency_seconds": mean_latency,
            "median_latency_seconds": median_latency,
            "max_latency_seconds": max_latency,
        },
        "representative_output": rep_output,
        "token_consumption_reference": {
            "comparison_scope": token_scope,
            "low_upper_boundary_approx": token_low_cut,
            "medium_upper_boundary_approx": token_high_cut,
            "interpretation": "Classify observed mean total tokens per invocation relative to these workflow-specific bands.",
        },
    }

    return context

