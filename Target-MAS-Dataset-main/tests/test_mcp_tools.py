"""
Unit and Integration Tests for MCP Tools and Server.

Validates that:
1. All 10 required MCP tools are registered and functional.
2. compare_agent_runs retrieves agent-specific metrics across runs.
3. get_execution_evidence returns specific agent evidence when both agent_id and run_id are given.
4. Path traversal attempts are strictly rejected.
5. Invalid custom evidence directories return clear errors without silent fallback.
6. Invalid agent IDs are explicitly reported in get_agent_profiles.
7. get_prompt_metrics supports full_prompt retrieval and compact defaults.
8. get_workflow_graph supports preserving or pruning isolated nodes.
9. Tools are strictly read-only and never alter evidence files.
"""

import asyncio
import json
from pathlib import Path
import pytest

from mcp_server import tools as t
from mcp_server.server import create_mcp_server


@pytest.fixture(scope="module")
def mcp_server():
    return create_mcp_server()


# ============================================================================
# Tool Registration and Call Tests (MCPServer via asyncio.run)
# ============================================================================
def test_mcp_server_tool_registration(mcp_server):
    """Verify that all 10 required tools are registered in the MCP server."""
    async def _test():
        registered = await mcp_server.list_tools()
        tool_names = {tool.name for tool in registered}
        expected_tools = {
            "list_workflows",
            "get_workflow_overview",
            "get_workflow_graph",
            "get_agent_profiles",
            "get_prompt_metrics",
            "get_latency_metrics",
            "get_token_metrics",
            "get_cost_metrics",
            "compare_agent_runs",
            "get_execution_evidence",
        }
        assert expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"
        assert len(registered) == 10

    asyncio.run(_test())


def test_mcp_server_call_tool(mcp_server):
    """Verify calling tools through the MCPServer interface."""
    async def _test():
        res = await mcp_server.call_tool("list_workflows", {})
        assert not res.is_error
        data = json.loads(res.content[0].text)
        assert data["total_workflows"] == 10

    asyncio.run(_test())


# ============================================================================
# Core Functional Tests
# ============================================================================
def test_list_workflows():
    res = t.list_workflows()
    assert res["total_workflows"] == 10
    wf_ids = {w["workflow_id"] for w in res["workflows"]}
    assert "contract_review" in wf_ids
    assert "customer_support" in wf_ids


def test_get_workflow_overview():
    res = t.get_workflow_overview("contract_review")
    assert res["workflow_id"] == "contract_review"
    assert res["total_runtime_agents"] == 10
    assert res["total_runs_recorded"] == 30
    assert "latency_summary" in res
    assert "token_summary" in res
    assert "cost_summary" in res


# ============================================================================
# Critical Case 1: compare_agent_runs (Agent-Specific Metrics Across Runs)
# ============================================================================
def test_compare_agent_runs_agent_specific_metrics():
    """Verify compare_agent_runs retrieves agent-specific metrics across at least 3 runs."""
    res = t.compare_agent_runs("contract_review", agent_id="clause_extraction_agent", limit=5)
    assert res["workflow_id"] == "contract_review"
    assert res["agent_id"] == "clause_extraction_agent"
    assert res["runs_compared_count"] >= 3

    # Check comparative aggregates
    comp = res["comparative_metrics"]
    assert "mean_duration_seconds" in comp
    assert "min_duration_seconds" in comp
    assert "max_duration_seconds" in comp
    assert "mean_tokens" in comp
    assert comp["min_duration_seconds"] > 0.0

    # Verify agent-specific fields are present in each run
    for r in res["agent_runs"]:
        assert r["agent_id"] == "clause_extraction_agent"
        assert r["status"] == "SUCCESS"
        assert r["duration_seconds"] is not None
        assert r["duration_seconds"] > 0.0
        assert r["tokens"] is not None
        assert r["tokens"] > 0
        assert r["prompt_tokens"] is not None
        assert r["completion_tokens"] is not None
        assert "assigned_key" in r
        assert r["assigned_key"].startswith("GROQ_API_KEY_2")


# ============================================================================
# Critical Case 2: get_execution_evidence (Specific Agent in Specific Run)
# ============================================================================
def test_get_execution_evidence_specific_agent_in_specific_run():
    """Verify get_execution_evidence returns specific agent evidence when both agent_id and run_id are passed."""
    run_id = "live_20261031_115354_trace01"
    agent_id = "clause_extraction_agent"
    res = t.get_execution_evidence("contract_review", agent_id=agent_id, run_id=run_id)

    assert res["workflow_id"] == "contract_review"
    assert res["run_id"] == run_id
    assert res["agent_id"] == agent_id
    assert res["status"] == "SUCCESS"
    assert res["order"] == 2
    assert res["assigned_key"].startswith("GROQ_API_KEY_2")
    assert res["duration_seconds"] > 0.0
    assert res["tokens"] > 0
    assert "output" in res
    assert isinstance(res["output"], dict)
    assert "clauses" in res["output"]  # output from clause extraction
    assert "source_location" in res
    assert res["source_location"]["file"] is not None


# ============================================================================
# Path Traversal and Directory Validation Tests
# ============================================================================
def test_path_traversal_rejection():
    """Verify that path traversal attempts in workflow_id are strictly rejected."""
    traversal_attempts = [
        "../contract_review",
        "..\\contract_review",
        "contract_review/../customer_support",
        "../../etc/passwd",
        "C:\\Windows\\System32",
        "/etc/shadow",
        "contract:review",
    ]
    for attempt in traversal_attempts:
        res = t.get_workflow_overview(attempt)
        assert "error" in res, f"Failed to reject traversal attempt: {attempt}"
        assert "path traversal" in res["error"].lower() or "invalid" in res["error"].lower()


def test_invalid_custom_evidence_dir_rejection():
    """Verify that invalid custom evidence directories return clear errors and do not silently fall back."""
    fake_dir = "non_existent_evidence_dir_12345"
    res = t.list_workflows(evidence_dir=fake_dir)
    assert "error" in res
    assert "Invalid custom evidence directory" in res["error"]

    res2 = t.get_workflow_overview("contract_review", evidence_dir=fake_dir)
    assert "error" in res2
    assert "Invalid custom evidence directory" in res2["error"]


# ============================================================================
# Invalid Agent ID Handling Tests
# ============================================================================
def test_invalid_agent_id_reporting():
    """Verify that get_agent_profiles explicitly reports unrecognized agent IDs."""
    res = t.get_agent_profiles(
        "contract_review",
        agent_ids=["clause_extraction_agent", "ghost_agent_x", "phantom_agent_y"],
    )
    assert res["agent_count"] == 1
    assert res["profiles"][0]["agent_id"] == "clause_extraction_agent"
    assert "invalid_agent_ids" in res
    assert set(res["invalid_agent_ids"]) == {"ghost_agent_x", "phantom_agent_y"}
    assert "error" in res


def test_all_invalid_agent_ids():
    """Verify behavior when all requested agent IDs are invalid."""
    res = t.get_agent_profiles("contract_review", agent_ids=["ghost_1", "ghost_2"])
    assert res["agent_count"] == 0
    assert len(res["profiles"]) == 0
    assert set(res["invalid_agent_ids"]) == {"ghost_1", "ghost_2"}


# ============================================================================
# Prompt Retrieval Tests
# ============================================================================
def test_prompt_metrics_full_prompt_and_truncation():
    """Verify prompt retrieval supports both compact previews and complete verbatim prompts."""
    # Compact preview (default)
    compact_res = t.get_prompt_metrics("contract_review", agent_ids=["contract_intake_agent"], full_prompt=False)
    entry = compact_res["prompt_metrics"][0]
    assert entry["is_truncated"] is True
    assert len(entry["sample_prompt_text"]) <= 405
    assert "prompt_text" not in entry

    # Full prompt requested
    full_res = t.get_prompt_metrics("contract_review", agent_ids=["contract_intake_agent"], full_prompt=True)
    full_entry = full_res["prompt_metrics"][0]
    assert full_entry["is_truncated"] is False
    assert "prompt_text" in full_entry
    assert len(full_entry["prompt_text"]) == full_entry["static_prompt_char_length"]


# ============================================================================
# Isolated Graph Node Preservation Tests
# ============================================================================
def test_graph_isolated_node_preservation():
    """Verify get_workflow_graph can retain or prune isolated nodes after filtering."""
    full_graph = t.get_workflow_graph("customer_support")
    total_nodes_count = full_graph["total_nodes"]

    # Retain isolated nodes (default)
    res_isolated = t.get_workflow_graph(
        "customer_support",
        relationship_types=["declared_dependency"],
        include_isolated_nodes=True,
    )
    # Total nodes should include all nodes in workflow graph
    assert res_isolated["total_nodes"] == total_nodes_count
    assert res_isolated["include_isolated_nodes"] is True

    # Prune isolated nodes
    res_pruned = t.get_workflow_graph(
        "customer_support",
        relationship_types=["declared_dependency"],
        include_isolated_nodes=False,
    )
    # Pruned nodes should only contain nodes attached to declared_dependency edges
    assert res_pruned["total_nodes"] < total_nodes_count
    assert res_pruned["include_isolated_nodes"] is False


# ============================================================================
# Read-Only and Error Robustness Tests
# ============================================================================
def test_invalid_agent_id_in_compare():
    res = t.compare_agent_runs("contract_review", agent_id="ghost_agent")
    assert "error" in res
    assert "available_agents" in res


def test_invalid_run_id_in_execution_evidence():
    res = t.get_execution_evidence("contract_review", run_id="invalid_run_999")
    assert "error" in res
    assert "available_sample_run_ids" in res


def test_tools_read_only_integrity():
    """Verify that calling MCP tools never alters evidence files."""
    base_dir, err = t._resolve_evidence_dir()
    assert err is None
    ev_file = base_dir / "contract_review" / "workflow_evidence.json"
    mtime_before = ev_file.stat().st_mtime_ns
    size_before = ev_file.stat().st_size

    t.list_workflows()
    t.get_workflow_overview("contract_review")
    t.get_workflow_graph("contract_review")
    t.get_latency_metrics("contract_review")
    t.compare_agent_runs("contract_review", "clause_extraction_agent", limit=3)
    t.get_execution_evidence("contract_review", "clause_extraction_agent", "live_20261031_115354_trace01")

    mtime_after = ev_file.stat().st_mtime_ns
    size_after = ev_file.stat().st_size

    assert mtime_before == mtime_after
    assert size_before == size_after
