"""
MCP Tool Implementations for TraceProfiler Evidence Access.

Exposes 10 focused, read-only tools that inspect workflow_evidence.json
artifacts without dumping full files, fabricating metrics, or allowing
path traversal.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _resolve_evidence_dir(custom_path: Optional[str] = None) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """
    Locate the Workflow_Evidence directory.
    If custom_path is provided, validates that it exists and is a directory.
    Does NOT silently fall back if an invalid custom_path is given.
    """
    if custom_path is not None:
        p = Path(custom_path).resolve()
        if not p.exists() or not p.is_dir():
            return None, {
                "error": f"Invalid custom evidence directory: '{custom_path}' does not exist or is not a directory."
            }
        return p, None

    cwd = Path.cwd()
    candidates = [
        cwd / "Target-MAS-Dataset-main" / "Workflow_Evidence",
        cwd / "Workflow_Evidence",
        cwd / ".." / "Target-MAS-Dataset-main" / "Workflow_Evidence",
        cwd / ".." / "Workflow_Evidence",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve(), None

    fallback = (cwd / "Workflow_Evidence").resolve()
    return fallback, None


def _validate_workflow_id(workflow_id: str, base_dir: Path) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """
    Validate workflow_id to prevent path traversal and ensure target file
    is strictly within base_dir.
    """
    if not workflow_id or not isinstance(workflow_id, str):
        return None, {"error": "workflow_id must be a non-empty string."}

    # Reject path traversal tokens and invalid directory characters
    invalid_patterns = ["..", "/", "\\", ":", "\0", "~", "*", "?", "<", ">", "|"]
    if any(p in workflow_id for p in invalid_patterns):
        return None, {
            "error": f"Invalid workflow_id '{workflow_id}': path traversal or invalid path characters detected."
        }

    resolved_base = base_dir.resolve()
    target_path = (resolved_base / workflow_id / "workflow_evidence.json").resolve()

    # Ensure target_path remains inside resolved_base
    try:
        target_path.relative_to(resolved_base)
    except ValueError:
        return None, {
            "error": f"Path traversal detected: '{workflow_id}' resolves outside the evidence directory."
        }

    return target_path, None


def _load_evidence(
    workflow_id: str, evidence_dir: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Safely load workflow_evidence.json for a given workflow_id with path validation."""
    base_dir, dir_err = _resolve_evidence_dir(evidence_dir)
    if dir_err:
        return None, dir_err

    wf_file, path_err = _validate_workflow_id(workflow_id, base_dir)
    if path_err:
        return None, path_err

    if not wf_file.exists():
        available = [
            d.name
            for d in sorted(base_dir.iterdir())
            if d.is_dir() and (d / "workflow_evidence.json").exists()
        ] if base_dir.exists() else []
        return None, {
            "error": f"Workflow '{workflow_id}' not found in evidence store.",
            "available_workflows": available,
            "searched_path": str(wf_file),
        }

    try:
        data = json.loads(wf_file.read_text(encoding="utf-8"))
        return data, None
    except Exception as e:
        return None, {
            "error": f"Failed to parse evidence file for '{workflow_id}': {str(e)}",
            "file_path": str(wf_file),
        }


def _find_traces_dir(workflow_id: str, data: Dict[str, Any]) -> Optional[Path]:
    """Locate the runtime traces directory for a workflow."""
    # 1. From metadata input_files
    meta_tdir = data.get("metadata", {}).get("input_files", {}).get("traces_directory")
    if meta_tdir:
        p = Path(meta_tdir)
        if p.exists() and p.is_dir():
            return p

    # 2. Check candidate relative directories
    cwd = Path.cwd()
    candidates = [
        cwd / "Ingestion_Traces_final" / "Ingestion_Traces" / workflow_id,
        cwd / "Target-MAS-Dataset-main" / "Ingestion_Traces_final" / "Ingestion_Traces" / workflow_id,
        cwd / ".." / "Ingestion_Traces_final" / "Ingestion_Traces" / workflow_id,
        cwd / "Ingestion_Traces" / "Ingestion_Traces" / workflow_id,
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


# ============================================================================
# Tool 1: list_workflows
# ============================================================================
def list_workflows(evidence_dir: Optional[str] = None) -> Dict[str, Any]:
    """List all available workflows in the evidence store and their processing status."""
    base_dir, err = _resolve_evidence_dir(evidence_dir)
    if err:
        return err

    if not base_dir.exists():
        return {"total_workflows": 0, "workflows": [], "evidence_dir": str(base_dir)}

    workflows = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        ev_file = d / "workflow_evidence.json"
        if ev_file.exists():
            try:
                data = json.loads(ev_file.read_text(encoding="utf-8"))
                meta = data.get("metadata", {})
                runtime = data.get("runtime_observations", {})
                workflows.append(
                    {
                        "workflow_id": meta.get("workflow_id", d.name),
                        "workflow_name": meta.get("workflow_name", d.name),
                        "aggregation_status": meta.get("aggregation_status", "unknown"),
                        "traces_processed": meta.get("traces_processed", 0),
                        "agent_count": len(runtime.get("agent_metrics", {})),
                        "evidence_file": str(ev_file),
                    }
                )
            except Exception:
                workflows.append({"workflow_id": d.name, "status": "corrupt_evidence_file"})

    return {
        "total_workflows": len(workflows),
        "workflows": workflows,
        "evidence_dir": str(base_dir),
    }


# ============================================================================
# Tool 2: get_workflow_overview
# ============================================================================
def get_workflow_overview(workflow_id: str, evidence_dir: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve high-level metadata, agent count, task count, and execution summary for a workflow."""
    data, err = _load_evidence(workflow_id, evidence_dir)
    if err:
        return err

    meta = data.get("metadata", {})
    definition = data.get("workflow_definition", {})
    runtime = data.get("runtime_observations", {})
    reconciliation = data.get("reconciliation_and_limitations", {})

    return {
        "workflow_id": workflow_id,
        "workflow_name": meta.get("workflow_name"),
        "description": definition.get("description"),
        "total_declared_agents": len(definition.get("declared_agents", [])),
        "total_runtime_agents": len(runtime.get("agent_metrics", {})),
        "total_runs_recorded": runtime.get("total_runs", 0),
        "successful_runs": runtime.get("successful_runs", 0),
        "failed_runs": runtime.get("failed_runs", 0),
        "latency_summary": runtime.get("workflow_latency", {}),
        "token_summary": runtime.get("workflow_tokens", {}),
        "cost_summary": runtime.get("workflow_cost", {}),
        "model_reconciliation": reconciliation.get("model_reconciliation", {}),
        "aggregation_status": meta.get("aggregation_status"),
    }


# ============================================================================
# Tool 3: get_workflow_graph
# ============================================================================
def get_workflow_graph(
    workflow_id: str,
    relationship_types: Optional[List[str]] = None,
    provenance_filter: Optional[str] = None,
    include_isolated_nodes: bool = True,
    evidence_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve graph nodes and dependency/handoff edges.
    Supports filtering edges by relationship or provenance.
    If include_isolated_nodes is True (default), preserves all workflow nodes even if
    they have no matching edges under the filter.
    """
    data, err = _load_evidence(workflow_id, evidence_dir)
    if err:
        return err

    graph = data.get("workflow_graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # Filter edges if requested
    filtered_edges = edges
    if relationship_types:
        rel_set = set(relationship_types)
        filtered_edges = [e for e in filtered_edges if e.get("relationship") in rel_set]

    if provenance_filter:
        filtered_edges = [e for e in filtered_edges if e.get("provenance") == provenance_filter]

    # Handle isolated nodes preservation
    if (relationship_types or provenance_filter) and not include_isolated_nodes:
        active_node_ids = {e["source"] for e in filtered_edges} | {e["target"] for e in filtered_edges}
        filtered_nodes = [n for n in nodes if n["id"] in active_node_ids]
    else:
        filtered_nodes = nodes

    return {
        "workflow_id": workflow_id,
        "total_nodes": len(filtered_nodes),
        "total_edges": len(filtered_edges),
        "relationship_filter": relationship_types,
        "provenance_filter": provenance_filter,
        "include_isolated_nodes": include_isolated_nodes,
        "nodes": filtered_nodes,
        "edges": filtered_edges,
    }


# ============================================================================
# Tool 4: get_agent_profiles
# ============================================================================
def get_agent_profiles(
    workflow_id: str,
    agent_ids: Optional[List[str]] = None,
    evidence_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve agent definitions, roles, goals, backstories, declared responsibilities, and tasks.
    Explicitly reports any requested agent IDs that do not exist instead of silently ignoring them.
    """
    data, err = _load_evidence(workflow_id, evidence_dir)
    if err:
        return err

    declared_agents = {a["agent_id"]: a for a in data.get("workflow_definition", {}).get("declared_agents", [])}
    static_agents = {a["agent_id"]: a for a in data.get("static_implementation_context", {}).get("agents", [])}
    runtime_agents = data.get("runtime_observations", {}).get("agent_metrics", {})

    all_ids = sorted(list(set(declared_agents.keys()) | set(static_agents.keys()) | set(runtime_agents.keys())))

    invalid_agent_ids = []
    target_ids = []
    if agent_ids:
        all_ids_set = set(all_ids)
        for aid in agent_ids:
            if aid in all_ids_set:
                target_ids.append(aid)
            else:
                invalid_agent_ids.append(aid)
    else:
        target_ids = all_ids

    profiles = []
    for aid in target_ids:
        dec = declared_agents.get(aid, {})
        stat = static_agents.get(aid, {})
        run = runtime_agents.get(aid, {})

        profiles.append(
            {
                "agent_id": aid,
                "role": stat.get("role") or dec.get("role"),
                "goal": stat.get("goal"),
                "backstory": stat.get("backstory"),
                "declared_responsibilities": dec.get("responsibilities"),
                "declared_inputs": dec.get("declared_inputs", []),
                "declared_outputs": dec.get("declared_outputs", []),
                "tasks": stat.get("tasks", []),
                "model_configuration": {
                    "declared_model": data.get("workflow_definition", {}).get("declared_model_configuration", {}).get("model"),
                    "static_model": stat.get("model"),
                    "resolution_status": stat.get("resolution_status"),
                },
                "assigned_keys": run.get("assigned_keys", []),
                "source_location": {
                    "file": stat.get("source_file"),
                    "start_line": stat.get("start_line"),
                    "end_line": stat.get("end_line"),
                },
            }
        )

    response: Dict[str, Any] = {
        "workflow_id": workflow_id,
        "agent_count": len(profiles),
        "profiles": profiles,
        "requested_ids": agent_ids,
    }
    if invalid_agent_ids:
        response["invalid_agent_ids"] = invalid_agent_ids
        response["error"] = f"The following requested agent IDs were not found in workflow '{workflow_id}': {invalid_agent_ids}"

    return response


# ============================================================================
# Tool 5: get_prompt_metrics
# ============================================================================
def get_prompt_metrics(
    workflow_id: str,
    agent_ids: Optional[List[str]] = None,
    full_prompt: bool = False,
    max_chars: Optional[int] = None,
    evidence_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve prompt descriptions, lengths, word counts, and runtime prompt token usage.
    If full_prompt is True, returns the complete prompt without truncation.
    Otherwise returns a compact preview (default 400 chars, or max_chars if given).
    """
    data, err = _load_evidence(workflow_id, evidence_dir)
    if err:
        return err

    static_agents = {a["agent_id"]: a for a in data.get("static_implementation_context", {}).get("agents", [])}
    runtime_agents = data.get("runtime_observations", {}).get("agent_metrics", {})

    all_ids = sorted(list(set(static_agents.keys()) | set(runtime_agents.keys())))

    invalid_agent_ids = []
    target_ids = []
    if agent_ids:
        all_ids_set = set(all_ids)
        for aid in agent_ids:
            if aid in all_ids_set:
                target_ids.append(aid)
            else:
                invalid_agent_ids.append(aid)
    else:
        target_ids = all_ids

    prompt_metrics = []
    char_limit = max_chars or 400

    for aid in target_ids:
        stat = static_agents.get(aid, {})
        run = runtime_agents.get(aid, {})
        prompts = stat.get("prompts", [])
        combined_text = "\n\n".join(prompts) if prompts else ""

        is_truncated = (len(combined_text) > char_limit) and not full_prompt
        sample_text = combined_text if full_prompt else (
            combined_text[:char_limit] + ("..." if is_truncated else "")
        )

        entry = {
            "agent_id": aid,
            "prompt_count": len(prompts),
            "static_prompt_char_length": len(combined_text),
            "static_prompt_word_count": len(combined_text.split()),
            "sample_prompt_text": sample_text,
            "is_truncated": is_truncated,
            "runtime_prompt_tokens": {
                "total": run.get("tokens", {}).get("total_prompt_tokens"),
                "mean": run.get("tokens", {}).get("mean_prompt_tokens"),
                "observation_count": run.get("tokens", {}).get("observation_count"),
            },
        }
        if full_prompt:
            entry["prompt_text"] = combined_text

        prompt_metrics.append(entry)

    response: Dict[str, Any] = {
        "workflow_id": workflow_id,
        "agent_count": len(prompt_metrics),
        "full_prompt_requested": full_prompt,
        "prompt_metrics": prompt_metrics,
    }
    if invalid_agent_ids:
        response["invalid_agent_ids"] = invalid_agent_ids
        response["warning"] = f"Unrecognized agent IDs: {invalid_agent_ids}"

    return response


# ============================================================================
# Tool 6: get_latency_metrics
# ============================================================================
def get_latency_metrics(
    workflow_id: str,
    agent_ids: Optional[List[str]] = None,
    evidence_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve empirical latency statistics (mean, median, min, max) for the workflow and agents."""
    data, err = _load_evidence(workflow_id, evidence_dir)
    if err:
        return err

    runtime = data.get("runtime_observations", {})
    wf_latency = runtime.get("workflow_latency", {})
    all_agent_metrics = runtime.get("agent_metrics", {})

    invalid_agent_ids = []
    target_ids = []
    if agent_ids:
        all_ids_set = set(all_agent_metrics.keys())
        for aid in agent_ids:
            if aid in all_ids_set:
                target_ids.append(aid)
            else:
                invalid_agent_ids.append(aid)
    else:
        target_ids = sorted(list(all_agent_metrics.keys()))

    agent_latencies = {}
    for aid in target_ids:
        agent_latencies[aid] = all_agent_metrics[aid].get("latency", {})

    response: Dict[str, Any] = {
        "workflow_id": workflow_id,
        "workflow_latency": wf_latency,
        "agent_latency_metrics": agent_latencies,
        "unit": "seconds",
    }
    if invalid_agent_ids:
        response["invalid_agent_ids"] = invalid_agent_ids
        response["warning"] = f"Unrecognized agent IDs: {invalid_agent_ids}"

    return response


# ============================================================================
# Tool 7: get_token_metrics
# ============================================================================
def get_token_metrics(
    workflow_id: str,
    agent_ids: Optional[List[str]] = None,
    evidence_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve empirical token consumption metrics (total, prompt, completion) for the workflow and agents."""
    data, err = _load_evidence(workflow_id, evidence_dir)
    if err:
        return err

    runtime = data.get("runtime_observations", {})
    wf_tokens = runtime.get("workflow_tokens", {})
    all_agent_metrics = runtime.get("agent_metrics", {})

    invalid_agent_ids = []
    target_ids = []
    if agent_ids:
        all_ids_set = set(all_agent_metrics.keys())
        for aid in agent_ids:
            if aid in all_ids_set:
                target_ids.append(aid)
            else:
                invalid_agent_ids.append(aid)
    else:
        target_ids = sorted(list(all_agent_metrics.keys()))

    agent_tokens = {}
    for aid in target_ids:
        agent_tokens[aid] = all_agent_metrics[aid].get("tokens", {})

    response: Dict[str, Any] = {
        "workflow_id": workflow_id,
        "workflow_tokens": wf_tokens,
        "observed_models": runtime.get("observed_models", []),
        "agent_token_metrics": agent_tokens,
    }
    if invalid_agent_ids:
        response["invalid_agent_ids"] = invalid_agent_ids
        response["warning"] = f"Unrecognized agent IDs: {invalid_agent_ids}"

    return response


# ============================================================================
# Tool 8: get_cost_metrics
# ============================================================================
def get_cost_metrics(
    workflow_id: str,
    agent_ids: Optional[List[str]] = None,
    evidence_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve recorded workflow-level cost metrics, estimation basis, and agent-level availability."""
    data, err = _load_evidence(workflow_id, evidence_dir)
    if err:
        return err

    runtime = data.get("runtime_observations", {})
    wf_cost = runtime.get("workflow_cost", {})
    all_agent_metrics = runtime.get("agent_metrics", {})

    invalid_agent_ids = []
    target_ids = []
    if agent_ids:
        all_ids_set = set(all_agent_metrics.keys())
        for aid in agent_ids:
            if aid in all_ids_set:
                target_ids.append(aid)
            else:
                invalid_agent_ids.append(aid)
    else:
        target_ids = sorted(list(all_agent_metrics.keys()))

    agent_costs = {}
    for aid in target_ids:
        agent_costs[aid] = all_agent_metrics[aid].get("cost", {
            "total_cost_usd": None,
            "mean_cost_usd": None,
            "calculation_basis": "unresolved_at_agent_level",
        })

    response: Dict[str, Any] = {
        "workflow_id": workflow_id,
        "workflow_cost": wf_cost,
        "calculation_basis": wf_cost.get("pricing_source", "trace_telemetry_estimates"),
        "agent_cost_metrics": agent_costs,
        "note": "Per-agent cost is unresolved at the individual agent level in traces; workflow-level cost is recorded.",
    }
    if invalid_agent_ids:
        response["invalid_agent_ids"] = invalid_agent_ids
        response["warning"] = f"Unrecognized agent IDs: {invalid_agent_ids}"

    return response


# ============================================================================
# Tool 9: compare_agent_runs
# ============================================================================
def compare_agent_runs(
    workflow_id: str,
    agent_id: str,
    run_ids: Optional[List[str]] = None,
    limit: int = 10,
    evidence_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare an agent's specific metrics (latency, token usage, execution status, prompt/completion
    tokens, assigned keys) across selected runs. Retrieves actual agent-level execution data.
    """
    data, err = _load_evidence(workflow_id, evidence_dir)
    if err:
        return err

    runtime = data.get("runtime_observations", {})
    agent_metrics = runtime.get("agent_metrics", {})

    if agent_id not in agent_metrics:
        return {
            "error": f"Agent '{agent_id}' not found in runtime observations for workflow '{workflow_id}'.",
            "available_agents": sorted(list(agent_metrics.keys())),
        }

    runs_summary = runtime.get("runs_summary", [])
    if run_ids:
        run_id_set = set(run_ids)
        selected_runs = [r for r in runs_summary if r["run_id"] in run_id_set]
    else:
        selected_runs = runs_summary[:limit]

    tdir = _find_traces_dir(workflow_id, data)

    agent_runs = []
    durations = []
    tokens = []

    for r_meta in selected_runs:
        tf_name = r_meta.get("trace_file")
        agent_exec_data = None

        if tdir and tf_name:
            trace_path = tdir / tf_name
            if trace_path.exists():
                try:
                    tdata = json.loads(trace_path.read_text(encoding="utf-8"))
                    for a in tdata.get("agent_executions", []):
                        if a.get("agent_id") == agent_id:
                            agent_exec_data = a
                            break
                except Exception:
                    pass

        if agent_exec_data:
            dur = agent_exec_data.get("duration_seconds")
            tok = agent_exec_data.get("tokens")
            if dur is not None:
                durations.append(dur)
            if tok is not None:
                tokens.append(tok)

            agent_runs.append(
                {
                    "run_id": r_meta["run_id"],
                    "agent_id": agent_id,
                    "status": agent_exec_data.get("status", "SUCCESS"),
                    "duration_seconds": dur,
                    "tokens": tok,
                    "prompt_tokens": agent_exec_data.get("prompt_tokens"),
                    "completion_tokens": agent_exec_data.get("completion_tokens"),
                    "order": agent_exec_data.get("order"),
                    "assigned_key": agent_exec_data.get("assigned_key_identifier"),
                    "has_output": bool(agent_exec_data.get("output")),
                    "trace_file": tf_name,
                }
            )
        else:
            # Fallback if trace file is unreadable on disk
            agent_runs.append(
                {
                    "run_id": r_meta["run_id"],
                    "agent_id": agent_id,
                    "status": "RECORDED_IN_SUMMARY",
                    "workflow_duration_seconds": r_meta.get("duration_seconds"),
                    "workflow_total_tokens": r_meta.get("total_tokens"),
                    "trace_file": tf_name,
                }
            )

    comparative_metrics = {}
    if durations:
        comparative_metrics["mean_duration_seconds"] = round(sum(durations) / len(durations), 4)
        comparative_metrics["min_duration_seconds"] = round(min(durations), 4)
        comparative_metrics["max_duration_seconds"] = round(max(durations), 4)
        comparative_metrics["latency_range_seconds"] = round(max(durations) - min(durations), 4)

    if tokens:
        comparative_metrics["mean_tokens"] = round(sum(tokens) / len(tokens), 1)
        comparative_metrics["min_tokens"] = min(tokens)
        comparative_metrics["max_tokens"] = max(tokens)
        comparative_metrics["token_range"] = max(tokens) - min(tokens)

    return {
        "workflow_id": workflow_id,
        "agent_id": agent_id,
        "runs_compared_count": len(agent_runs),
        "agent_aggregate_latency": agent_metrics[agent_id].get("latency"),
        "agent_aggregate_tokens": agent_metrics[agent_id].get("tokens"),
        "comparative_metrics": comparative_metrics,
        "agent_runs": agent_runs,
    }


# ============================================================================
# Tool 10: get_execution_evidence
# ============================================================================
def get_execution_evidence(
    workflow_id: str,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    evidence_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve detailed execution evidence, sample inputs/outputs, or run-specific trace details.
    When BOTH agent_id and run_id are provided, returns evidence for that specific agent in that specific run.
    """
    data, err = _load_evidence(workflow_id, evidence_dir)
    if err:
        return err

    runtime = data.get("runtime_observations", {})
    runs = runtime.get("runs_summary", [])
    agent_metrics = runtime.get("agent_metrics", {})
    tdir = _find_traces_dir(workflow_id, data)

    # Case 1: BOTH agent_id AND run_id provided -> specific agent in specific run
    if agent_id and run_id:
        if agent_id not in agent_metrics:
            return {
                "error": f"Agent '{agent_id}' not found in runtime observations for workflow '{workflow_id}'.",
                "available_agents": sorted(list(agent_metrics.keys())),
            }

        target_run = next((r for r in runs if r["run_id"] == run_id), None)
        if not target_run:
            return {
                "error": f"Run ID '{run_id}' not found in workflow '{workflow_id}'.",
                "available_sample_run_ids": [r["run_id"] for r in runs[:5]],
            }

        tf_name = target_run.get("trace_file")
        agent_exec = None
        if tdir and tf_name:
            trace_path = tdir / tf_name
            if trace_path.exists():
                try:
                    tdata = json.loads(trace_path.read_text(encoding="utf-8"))
                    for a in tdata.get("agent_executions", []):
                        if a.get("agent_id") == agent_id:
                            agent_exec = a
                            break
                except Exception:
                    pass

        static_agents = {a["agent_id"]: a for a in data.get("static_implementation_context", {}).get("agents", [])}
        stat = static_agents.get(agent_id, {})

        if agent_exec:
            return {
                "workflow_id": workflow_id,
                "run_id": run_id,
                "agent_id": agent_id,
                "status": agent_exec.get("status", "SUCCESS"),
                "duration_seconds": agent_exec.get("duration_seconds"),
                "tokens": agent_exec.get("tokens"),
                "prompt_tokens": agent_exec.get("prompt_tokens"),
                "completion_tokens": agent_exec.get("completion_tokens"),
                "order": agent_exec.get("order"),
                "assigned_key": agent_exec.get("assigned_key_identifier"),
                "input": agent_exec.get("input"),
                "output": agent_exec.get("output"),
                "notes": agent_exec.get("notes"),
                "trace_file": tf_name,
                "source_location": {
                    "file": stat.get("source_file"),
                    "start_line": stat.get("start_line"),
                    "end_line": stat.get("end_line"),
                    "evidence_refs": stat.get("evidence_refs"),
                },
            }
        else:
            return {
                "workflow_id": workflow_id,
                "run_id": run_id,
                "agent_id": agent_id,
                "status": "NOT_EXECUTED_OR_UNAVAILABLE",
                "trace_file": tf_name,
                "message": f"Execution evidence for agent '{agent_id}' was not found in run '{run_id}'.",
            }

    # Case 2: Specific Run only
    if run_id:
        target_run = next((r for r in runs if r["run_id"] == run_id), None)
        if not target_run:
            return {
                "error": f"Run ID '{run_id}' not found in workflow '{workflow_id}'.",
                "available_sample_run_ids": [r["run_id"] for r in runs[:5]],
            }

        # Retrieve executed agent list from trace if possible
        executed_agents = []
        tf_name = target_run.get("trace_file")
        if tdir and tf_name:
            trace_path = tdir / tf_name
            if trace_path.exists():
                try:
                    tdata = json.loads(trace_path.read_text(encoding="utf-8"))
                    executed_agents = [a.get("agent_id") for a in tdata.get("agent_executions", [])]
                except Exception:
                    pass

        return {
            "workflow_id": workflow_id,
            "run_id": run_id,
            "run_details": target_run,
            "executed_agents": executed_agents,
            "trace_file": tf_name,
        }

    # Case 3: Specific Agent only
    if agent_id:
        if agent_id not in agent_metrics:
            return {
                "error": f"Agent '{agent_id}' not found in runtime observations.",
                "available_agents": sorted(list(agent_metrics.keys())),
            }
        am = agent_metrics[agent_id]
        static_agents = {a["agent_id"]: a for a in data.get("static_implementation_context", {}).get("agents", [])}
        stat = static_agents.get(agent_id, {})

        return {
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "assigned_keys": am.get("assigned_keys"),
            "observed_execution_orders": am.get("observed_execution_orders"),
            "sample_inputs": am.get("sample_inputs"),
            "sample_outputs": am.get("sample_outputs"),
            "source_location": {
                "file": stat.get("source_file"),
                "start_line": stat.get("start_line"),
                "end_line": stat.get("end_line"),
                "evidence_refs": stat.get("evidence_refs"),
            },
        }

    # Case 4: Neither specified -> return summary of available evidence references
    meta = data.get("metadata", {})
    return {
        "workflow_id": workflow_id,
        "input_files": meta.get("input_files"),
        "total_runs": runtime.get("total_runs"),
        "agents_available": sorted(list(agent_metrics.keys())),
        "first_5_run_ids": [r["run_id"] for r in runs[:5]],
        "instructions": "Specify both 'agent_id' and 'run_id' for exact agent run evidence, or either one individually.",
    }
