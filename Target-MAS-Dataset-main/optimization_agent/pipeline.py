"""
Sequential 10-Stage MCP Evidence Retrieval Pipeline for LangGraph.

Strict Requirements:
1. Sequential execution through Stages 1 to 10 in exact order.
2. Interface with existing TraceProfiler MCP Evidence Server tools (mcp_server.tools).
3. Stage 9 calls compare_agent_runs for EVERY discovered agent.
4. Stage 10 calls get_execution_evidence for EVERY discovered agent across sample runs.
5. All errors and limitations are recorded cleanly in state without crashing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mcp_server import tools as evidence_tools
from optimization_agent.state import OptimizationAgentState

logger = logging.getLogger("optimization_agent.pipeline")


def _init_tracking_fields(state: OptimizationAgentState) -> Dict[str, Any]:
    """Helper to ensure status tracking and collections are initialized."""
    stage_status = dict(state.get("stage_status") or {})
    retrieval_errors = list(state.get("retrieval_errors") or [])
    analysis_limitations = list(state.get("analysis_limitations") or [])
    return {
        "stage_status": stage_status,
        "retrieval_errors": retrieval_errors,
        "analysis_limitations": analysis_limitations,
    }


def stage_1_discovery(state: OptimizationAgentState) -> Dict[str, Any]:
    """Stage 1: Call list_workflows() to discover available workflows and verify target."""
    track = _init_tracking_fields(state)
    wf_id = state.get("workflow_id", "")
    ev_dir = state.get("evidence_dir")

    try:
        res = evidence_tools.list_workflows(evidence_dir=ev_dir)
        if "error" in res:
            track["retrieval_errors"].append({"stage": "stage_1_discovery", "error": res["error"]})
            track["stage_status"]["stage_1_discovery"] = "failed"
            return {**track, "available_workflows": []}

        workflows = res.get("workflows", [])
        wf_ids = [w.get("workflow_id") for w in workflows if isinstance(w, dict)]

        if wf_id not in wf_ids:
            msg = f"Target workflow '{wf_id}' not found among available workflows: {wf_ids}"
            track["retrieval_errors"].append({"stage": "stage_1_discovery", "error": msg})
            track["analysis_limitations"].append(msg)
            track["stage_status"]["stage_1_discovery"] = "warning"
        else:
            track["stage_status"]["stage_1_discovery"] = "completed"

        return {
            **track,
            "available_workflows": workflows,
        }
    except Exception as e:
        logger.error(f"Stage 1 error: {e}")
        track["retrieval_errors"].append({"stage": "stage_1_discovery", "error": str(e)})
        track["stage_status"]["stage_1_discovery"] = "failed"
        return track


def stage_2_overview(state: OptimizationAgentState) -> Dict[str, Any]:
    """Stage 2: Call get_workflow_overview(workflow_id) to retrieve metadata and high-level summaries."""
    track = _init_tracking_fields(state)
    wf_id = state.get("workflow_id", "")
    ev_dir = state.get("evidence_dir")

    try:
        res = evidence_tools.get_workflow_overview(workflow_id=wf_id, evidence_dir=ev_dir)
        if "error" in res:
            track["retrieval_errors"].append({"stage": "stage_2_overview", "error": res["error"]})
            track["stage_status"]["stage_2_overview"] = "failed"
            return {**track, "workflow_overview": {}, "workflow_metadata": {}}

        metadata = {
            "workflow_id": res.get("workflow_id", wf_id),
            "workflow_name": res.get("workflow_name", wf_id),
            "description": res.get("description", ""),
            "total_declared_agents": res.get("total_declared_agents", 0),
            "total_runtime_agents": res.get("total_runtime_agents", 0),
            "total_runs_recorded": res.get("total_runs_recorded", 0),
            "successful_runs": res.get("successful_runs", 0),
            "failed_runs": res.get("failed_runs", 0),
            "model_reconciliation": res.get("model_reconciliation", {}),
        }
        track["stage_status"]["stage_2_overview"] = "completed"
        return {
            **track,
            "workflow_overview": res,
            "workflow_metadata": metadata,
        }
    except Exception as e:
        logger.error(f"Stage 2 error: {e}")
        track["retrieval_errors"].append({"stage": "stage_2_overview", "error": str(e)})
        track["stage_status"]["stage_2_overview"] = "failed"
        return track


def stage_3_graph(state: OptimizationAgentState) -> Dict[str, Any]:
    """Stage 3: Call get_workflow_graph(workflow_id, include_isolated_nodes=True)."""
    track = _init_tracking_fields(state)
    wf_id = state.get("workflow_id", "")
    ev_dir = state.get("evidence_dir")

    try:
        res = evidence_tools.get_workflow_graph(
            workflow_id=wf_id,
            include_isolated_nodes=True,
            evidence_dir=ev_dir,
        )
        if "error" in res:
            track["retrieval_errors"].append({"stage": "stage_3_graph", "error": res["error"]})
            track["stage_status"]["stage_3_graph"] = "failed"
            return {**track, "workflow_graph": {}}

        track["stage_status"]["stage_3_graph"] = "completed"
        return {
            **track,
            "workflow_graph": res,
        }
    except Exception as e:
        logger.error(f"Stage 3 error: {e}")
        track["retrieval_errors"].append({"stage": "stage_3_graph", "error": str(e)})
        track["stage_status"]["stage_3_graph"] = "failed"
        return track


def stage_4_profiles(state: OptimizationAgentState) -> Dict[str, Any]:
    """Stage 4: Call get_agent_profiles(workflow_id) and discover all agent IDs."""
    track = _init_tracking_fields(state)
    wf_id = state.get("workflow_id", "")
    ev_dir = state.get("evidence_dir")

    try:
        res = evidence_tools.get_agent_profiles(workflow_id=wf_id, evidence_dir=ev_dir)
        if "error" in res:
            track["retrieval_errors"].append({"stage": "stage_4_profiles", "error": res["error"]})
            track["stage_status"]["stage_4_profiles"] = "failed"
            return {**track, "agent_profiles": {}, "discovered_agents": []}

        profiles_list = res.get("profiles", [])
        discovered_agents: List[str] = []
        for p in profiles_list:
            if isinstance(p, dict) and p.get("agent_id"):
                discovered_agents.append(p["agent_id"])

        discovered_agents = sorted(list(set(discovered_agents)))
        if not discovered_agents:
            track["analysis_limitations"].append(
                f"No agents discovered in agent_profiles for workflow '{wf_id}'."
            )

        track["stage_status"]["stage_4_profiles"] = "completed"
        return {
            **track,
            "agent_profiles": res,
            "discovered_agents": discovered_agents,
        }
    except Exception as e:
        logger.error(f"Stage 4 error: {e}")
        track["retrieval_errors"].append({"stage": "stage_4_profiles", "error": str(e)})
        track["stage_status"]["stage_4_profiles"] = "failed"
        return track


def stage_5_prompts(state: OptimizationAgentState) -> Dict[str, Any]:
    """Stage 5: Call get_prompt_metrics(workflow_id, full_prompt=True)."""
    track = _init_tracking_fields(state)
    wf_id = state.get("workflow_id", "")
    ev_dir = state.get("evidence_dir")

    try:
        res = evidence_tools.get_prompt_metrics(
            workflow_id=wf_id,
            full_prompt=True,
            evidence_dir=ev_dir,
        )
        if "error" in res:
            track["retrieval_errors"].append({"stage": "stage_5_prompts", "error": res["error"]})
            track["stage_status"]["stage_5_prompts"] = "failed"
            return {**track, "prompt_metrics": {}}

        track["stage_status"]["stage_5_prompts"] = "completed"
        return {
            **track,
            "prompt_metrics": res,
        }
    except Exception as e:
        logger.error(f"Stage 5 error: {e}")
        track["retrieval_errors"].append({"stage": "stage_5_prompts", "error": str(e)})
        track["stage_status"]["stage_5_prompts"] = "failed"
        return track


def stage_6_latency(state: OptimizationAgentState) -> Dict[str, Any]:
    """Stage 6: Call get_latency_metrics(workflow_id)."""
    track = _init_tracking_fields(state)
    wf_id = state.get("workflow_id", "")
    ev_dir = state.get("evidence_dir")

    try:
        res = evidence_tools.get_latency_metrics(workflow_id=wf_id, evidence_dir=ev_dir)
        if "error" in res:
            track["retrieval_errors"].append({"stage": "stage_6_latency", "error": res["error"]})
            track["stage_status"]["stage_6_latency"] = "failed"
            return {**track, "latency_metrics": {}}

        track["stage_status"]["stage_6_latency"] = "completed"
        return {
            **track,
            "latency_metrics": res,
        }
    except Exception as e:
        logger.error(f"Stage 6 error: {e}")
        track["retrieval_errors"].append({"stage": "stage_6_latency", "error": str(e)})
        track["stage_status"]["stage_6_latency"] = "failed"
        return track


def stage_7_tokens(state: OptimizationAgentState) -> Dict[str, Any]:
    """Stage 7: Call get_token_metrics(workflow_id)."""
    track = _init_tracking_fields(state)
    wf_id = state.get("workflow_id", "")
    ev_dir = state.get("evidence_dir")

    try:
        res = evidence_tools.get_token_metrics(workflow_id=wf_id, evidence_dir=ev_dir)
        if "error" in res:
            track["retrieval_errors"].append({"stage": "stage_7_tokens", "error": res["error"]})
            track["stage_status"]["stage_7_tokens"] = "failed"
            return {**track, "token_metrics": {}}

        track["stage_status"]["stage_7_tokens"] = "completed"
        return {
            **track,
            "token_metrics": res,
        }
    except Exception as e:
        logger.error(f"Stage 7 error: {e}")
        track["retrieval_errors"].append({"stage": "stage_7_tokens", "error": str(e)})
        track["stage_status"]["stage_7_tokens"] = "failed"
        return track


def stage_8_cost(state: OptimizationAgentState) -> Dict[str, Any]:
    """Stage 8: Call get_cost_metrics(workflow_id)."""
    track = _init_tracking_fields(state)
    wf_id = state.get("workflow_id", "")
    ev_dir = state.get("evidence_dir")

    try:
        res = evidence_tools.get_cost_metrics(workflow_id=wf_id, evidence_dir=ev_dir)
        if "error" in res:
            track["retrieval_errors"].append({"stage": "stage_8_cost", "error": res["error"]})
            track["stage_status"]["stage_8_cost"] = "failed"
            return {**track, "cost_metrics": {}}

        track["stage_status"]["stage_8_cost"] = "completed"
        return {
            **track,
            "cost_metrics": res,
        }
    except Exception as e:
        logger.error(f"Stage 8 error: {e}")
        track["retrieval_errors"].append({"stage": "stage_8_cost", "error": str(e)})
        track["stage_status"]["stage_8_cost"] = "failed"
        return track


def stage_9_comparison(state: OptimizationAgentState) -> Dict[str, Any]:
    """
    Stage 9: Call compare_agent_runs(workflow_id, agent_id, limit=30)
    for EVERY agent discovered in Stage 4.
    """
    track = _init_tracking_fields(state)
    wf_id = state.get("workflow_id", "")
    ev_dir = state.get("evidence_dir")
    discovered_agents = list(state.get("discovered_agents", []))

    comparisons: Dict[str, Any] = {}
    observed_runs: set = set()

    for agent_id in discovered_agents:
        try:
            res = evidence_tools.compare_agent_runs(
                workflow_id=wf_id,
                agent_id=agent_id,
                limit=30,
                evidence_dir=ev_dir,
            )
            comparisons[agent_id] = res

            # Extract run IDs for Stage 10 sampling
            runs = res.get("agent_runs", [])
            for r in runs:
                if isinstance(r, dict) and r.get("run_id"):
                    observed_runs.add(r["run_id"])
        except Exception as e:
            logger.error(f"Error comparing runs for agent '{agent_id}': {e}")
            track["retrieval_errors"].append({
                "stage": "stage_9_comparison",
                "agent_id": agent_id,
                "error": str(e),
            })
            comparisons[agent_id] = {"error": str(e), "agent_id": agent_id}

    sample_run_ids = sorted(list(observed_runs))[:5]

    track["stage_status"]["stage_9_comparison"] = "completed"
    return {
        **track,
        "agent_run_comparisons": comparisons,
        "sample_run_ids": sample_run_ids,
    }


def stage_10_evidence(state: OptimizationAgentState) -> Dict[str, Any]:
    """
    Stage 10: Call get_execution_evidence(workflow_id, agent_id, run_id)
    for EVERY agent discovered in Stage 4 across representative sample runs.
    """
    track = _init_tracking_fields(state)
    wf_id = state.get("workflow_id", "")
    ev_dir = state.get("evidence_dir")
    discovered_agents = list(state.get("discovered_agents", []))
    sample_run_ids = list(state.get("sample_run_ids", []))
    primary_run_id = sample_run_ids[0] if sample_run_ids else None

    execution_evidence: Dict[str, Any] = {}

    for agent_id in discovered_agents:
        try:
            res = evidence_tools.get_execution_evidence(
                workflow_id=wf_id,
                agent_id=agent_id,
                run_id=primary_run_id,
                evidence_dir=ev_dir,
            )
            execution_evidence[agent_id] = res
        except Exception as e:
            logger.error(f"Error fetching execution evidence for agent '{agent_id}': {e}")
            track["retrieval_errors"].append({
                "stage": "stage_10_evidence",
                "agent_id": agent_id,
                "error": str(e),
            })
            execution_evidence[agent_id] = {"error": str(e), "agent_id": agent_id}

    track["stage_status"]["stage_10_evidence"] = "completed"
    return {
        **track,
        "execution_evidence": execution_evidence,
    }

