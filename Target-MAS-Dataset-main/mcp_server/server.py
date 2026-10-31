"""
TraceProfiler MCP Evidence Server.

Exposes the 10 unified workflow evidence tools through the official Python MCP SDK.
Provides focused, read-only access to workflow_evidence.json artifacts for the
downstream Trace Profiler Agent.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer

from mcp_server import tools as evidence_tools


def create_mcp_server(name: str = "traceprofiler-evidence-server") -> MCPServer:
    """Create and configure the MCPServer instance with all 10 evidence tools."""
    server = MCPServer(name=name, version="1.0.0")

    @server.tool(
        name="list_workflows",
        description="List all multi-agent workflows available in the evidence store, including agent counts and processing status.",
    )
    def list_workflows(evidence_dir: Optional[str] = None) -> str:
        """List available workflows in the evidence store."""
        res = evidence_tools.list_workflows(evidence_dir=evidence_dir)
        return json.dumps(res, indent=2)

    @server.tool(
        name="get_workflow_overview",
        description="Retrieve high-level metadata, agent counts, latency/token/cost summaries, and model reconciliation for a workflow.",
    )
    def get_workflow_overview(workflow_id: str, evidence_dir: Optional[str] = None) -> str:
        """Retrieve workflow metadata, agent count, task count, and execution summary."""
        res = evidence_tools.get_workflow_overview(workflow_id=workflow_id, evidence_dir=evidence_dir)
        return json.dumps(res, indent=2)

    @server.tool(
        name="get_workflow_graph",
        description="Retrieve workflow graph nodes and edges. Supports filtering by relationship types or provenance tiers (declared, statically_observed, runtime_observed), and optionally preserving isolated nodes.",
    )
    def get_workflow_graph(
        workflow_id: str,
        relationship_types: Optional[List[str]] = None,
        provenance_filter: Optional[str] = None,
        include_isolated_nodes: bool = True,
        evidence_dir: Optional[str] = None,
    ) -> str:
        """Retrieve nodes, dependencies, handoffs, and execution order relationships."""
        res = evidence_tools.get_workflow_graph(
            workflow_id=workflow_id,
            relationship_types=relationship_types,
            provenance_filter=provenance_filter,
            include_isolated_nodes=include_isolated_nodes,
            evidence_dir=evidence_dir,
        )
        return json.dumps(res, indent=2)

    @server.tool(
        name="get_agent_profiles",
        description="Retrieve detailed agent definitions, roles, goals, backstories, tasks, and declared inputs/outputs. Explicitly reports unrecognized agent IDs.",
    )
    def get_agent_profiles(
        workflow_id: str,
        agent_ids: Optional[List[str]] = None,
        evidence_dir: Optional[str] = None,
    ) -> str:
        """Retrieve agent responsibilities, task assignments, and model information."""
        res = evidence_tools.get_agent_profiles(
            workflow_id=workflow_id, agent_ids=agent_ids, evidence_dir=evidence_dir
        )
        return json.dumps(res, indent=2)

    @server.tool(
        name="get_prompt_metrics",
        description="Retrieve prompt text, character length, word count, and observed runtime prompt token usage across executions. Supports full_prompt retrieval.",
    )
    def get_prompt_metrics(
        workflow_id: str,
        agent_ids: Optional[List[str]] = None,
        full_prompt: bool = False,
        max_chars: Optional[int] = None,
        evidence_dir: Optional[str] = None,
    ) -> str:
        """Retrieve prompt-related information and available prompt statistics."""
        res = evidence_tools.get_prompt_metrics(
            workflow_id=workflow_id,
            agent_ids=agent_ids,
            full_prompt=full_prompt,
            max_chars=max_chars,
            evidence_dir=evidence_dir,
        )
        return json.dumps(res, indent=2)

    @server.tool(
        name="get_latency_metrics",
        description="Retrieve empirical latency metrics (mean, median, min, max seconds) at workflow and agent levels.",
    )
    def get_latency_metrics(
        workflow_id: str,
        agent_ids: Optional[List[str]] = None,
        evidence_dir: Optional[str] = None,
    ) -> str:
        """Retrieve per-agent and workflow-level latency metrics."""
        res = evidence_tools.get_latency_metrics(
            workflow_id=workflow_id, agent_ids=agent_ids, evidence_dir=evidence_dir
        )
        return json.dumps(res, indent=2)

    @server.tool(
        name="get_token_metrics",
        description="Retrieve empirical token consumption statistics (total, prompt, completion) for workflow and individual agents.",
    )
    def get_token_metrics(
        workflow_id: str,
        agent_ids: Optional[List[str]] = None,
        evidence_dir: Optional[str] = None,
    ) -> str:
        """Retrieve per-agent, per-model, and workflow-level token usage."""
        res = evidence_tools.get_token_metrics(
            workflow_id=workflow_id, agent_ids=agent_ids, evidence_dir=evidence_dir
        )
        return json.dumps(res, indent=2)

    @server.tool(
        name="get_cost_metrics",
        description="Retrieve estimated execution costs in USD, pricing basis, and agent-level cost availability.",
    )
    def get_cost_metrics(
        workflow_id: str,
        agent_ids: Optional[List[str]] = None,
        evidence_dir: Optional[str] = None,
    ) -> str:
        """Retrieve available cost metrics and their calculation basis."""
        res = evidence_tools.get_cost_metrics(
            workflow_id=workflow_id, agent_ids=agent_ids, evidence_dir=evidence_dir
        )
        return json.dumps(res, indent=2)

    @server.tool(
        name="compare_agent_runs",
        description="Compare an agent's latency, token consumption, status, and assigned keys across individual execution runs.",
    )
    def compare_agent_runs(
        workflow_id: str,
        agent_id: str,
        run_ids: Optional[List[str]] = None,
        limit: int = 10,
        evidence_dir: Optional[str] = None,
    ) -> str:
        """Compare an agent's execution metrics across selected runs."""
        res = evidence_tools.compare_agent_runs(
            workflow_id=workflow_id,
            agent_id=agent_id,
            run_ids=run_ids,
            limit=limit,
            evidence_dir=evidence_dir,
        )
        return json.dumps(res, indent=2)

    @server.tool(
        name="get_execution_evidence",
        description="Retrieve specific runtime trace evidence, structured outputs, sample inputs, or source code line references. When both agent_id and run_id are specified, returns evidence for that specific agent in that specific run.",
    )
    def get_execution_evidence(
        workflow_id: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        evidence_dir: Optional[str] = None,
    ) -> str:
        """Retrieve selected runtime evidence and source references for investigation."""
        res = evidence_tools.get_execution_evidence(
            workflow_id=workflow_id, agent_id=agent_id, run_id=run_id, evidence_dir=evidence_dir
        )
        return json.dumps(res, indent=2)

    return server


async def run_stdio():
    """Run the MCP server over standard I/O."""
    server = create_mcp_server()
    await server.run_stdio_async()


def main():
    """CLI entry point for running the TraceProfiler MCP server."""
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
