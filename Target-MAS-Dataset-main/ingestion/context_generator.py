"""
TraceProfiler Context Generator Module.

Consolidates AST extraction and workflow analysis results into a fully validated,
compact, and LLM-ready `WorkflowContext` dictionary or JSON document.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from schemas.workflow_context_schema import (
    TOKEN_ESTIMATION_DEFAULT_METHOD,
    WorkflowContext,
    validate_workflow_context,
)
from ingestion.workflow_analyzer import WorkflowAnalysisResult


# ============================================================================
# Context Generator Engine
# ============================================================================

class ContextGenerator:
    """Generates the validated workflow_context.json structure from analysis results."""

    def __init__(self, token_ratio: int = 4) -> None:
        self.token_ratio = token_ratio

    def generate_context(
        self,
        analysis: WorkflowAnalysisResult,
        final_validation: bool = True,
        strict_evidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Assemble all extraction outputs into a compliant WorkflowContext dictionary.

        Args:
            analysis: Result from WorkflowAnalyzer.
            final_validation: If True, enforces strict cross-section count consistency.
            strict_evidence: If True, enforces that all evidence_refs resolve.

        Returns:
            Fully validated dictionary adhering to schemas.workflow_context_schema.
        """
        # 1. Metadata
        analysis_status = "completed"
        if analysis.failed_files:
            analysis_status = "partial" if analysis.parsed_files else "failed"

        metadata = {
            "workflow_id": analysis.workflow_id,
            "workflow_name": analysis.workflow_name,
            "source_directory": analysis.source_dir,
            "workflow_entrypoint": analysis.entrypoint or "workflow.py",
            "analysis_status": analysis_status,
        }

        # 2. Source inventory
        all_files = analysis.parsed_files + analysis.failed_files
        source_inv = {
            "total_python_files": len(all_files),
            "agent_candidate_files": analysis.discovered_agent_files,
            "parsed_files": len(analysis.parsed_files),
            "failed_files": len(analysis.failed_files),
            "files": all_files,
        }

        # 3. Agent inventory & Agent details
        agent_inv_entries: List[Dict[str, Any]] = []
        agent_details_list: List[Dict[str, Any]] = []

        # Deduplicate agents by agent_id
        seen_agent_ids: Set[str] = set()
        deduped_agents = []
        for ag in analysis.identified_agents:
            if ag.agent_id not in seen_agent_ids:
                seen_agent_ids.add(ag.agent_id)
                deduped_agents.append(ag)

        invoked_count = 0
        uninvoked_count = 0

        for ag in deduped_agents:
            is_invoked = ag.agent_id in analysis.invoked_agent_ids
            inv_status = "invoked" if is_invoked else "uninvoked"
            if is_invoked:
                invoked_count += 1
            else:
                uninvoked_count += 1

            agent_inv_entries.append({
                "agent_id": ag.agent_id,
                "source_file": ag.source_file,
                "definition_status": "static",
                "invocation_status": inv_status,
                "evidence_refs": ag.evidence_refs,
            })

            agent_details_list.append(ag.to_dict())

        agent_inv = {
            "discovered_agent_files": analysis.discovered_agent_files,
            "identified_agent_definitions": len(deduped_agents),
            "workflow_invoked_agents": invoked_count,
            "uninvoked_agent_definitions": uninvoked_count,
            "agents": agent_inv_entries,
        }

        # 4. Dependency graph
        dep_graph = {
            "nodes": analysis.dependency_nodes,
            "edges": analysis.dependency_edges,
        }

        # 5. Tools & Helper inventory
        tools_inventory = {
            "registered_tools": analysis.registered_tools,
            "explicit_code_invocations": analysis.explicit_code_invocations,
            "helper_functions": analysis.helper_functions,
            "runtime_llm_tool_selection": {
                "observable": False,
                "reason": "Runtime LLM tool selection is unobservable via static AST analysis",
            },
        }

        # 6. Evidence & provenance
        evidence_prov = {
            "evidence": analysis.evidence_records,
        }

        # 7. Analysis limitations
        limitations = {
            "unresolved_references": analysis.unresolved_references,
            "dynamic_expressions": [],
            "unsupported_patterns": [],
            "static_analysis_limitations": [
                "Dynamic runtime LLM decisions and tool selection cannot be observed via static AST",
                "Procedural workflow statement order does not imply internal framework agent scheduling",
            ],
        }

        # 8. Summary statistics
        total_tasks = len(analysis.tasks)
        total_dependencies = len(analysis.dependency_edges)
        unresolved_relationships = sum(
            1 for e in analysis.dependency_edges if e.get("resolution_status") == "unresolved"
        )
        completeness = "complete" if analysis_status == "completed" else "partial"

        # Pre-build summary to calculate metrics
        summary = {
            "total_agents": len(deduped_agents),
            "invoked_agents": invoked_count,
            "uninvoked_agents": uninvoked_count,
            "total_tasks": total_tasks,
            "total_dependencies": total_dependencies,
            "unresolved_relationships": unresolved_relationships,
            "completeness": completeness,
            "output_metrics": {
                "serialized_bytes": 0,
                "estimated_tokens": 0,
                "token_estimation_method": TOKEN_ESTIMATION_DEFAULT_METHOD,
            },
        }

        context_dict: Dict[str, Any] = {
            "schema_version": "1.0",
            "workflow_metadata": metadata,
            "source_inventory": source_inv,
            "agent_inventory": agent_inv,
            "agent_details": agent_details_list,
            "outer_workflow_flow": analysis.outer_flow,
            "orchestrator_process": analysis.orchestrator,
            "dependency_graph": dep_graph,
            "tool_and_helper_inventory": tools_inventory,
            "evidence_and_provenance": evidence_prov,
            "analysis_limitations": limitations,
            "analysis_summary": summary,
        }

        # 9. Compute serialized bytes and tokens (converge to fixed point)
        for _ in range(3):
            json_temp = json.dumps(context_dict, indent=2)
            byte_count = len(json_temp.encode("utf-8"))
            est_tokens = max(1, len(json_temp) // self.token_ratio)
            summary["output_metrics"]["serialized_bytes"] = byte_count
            summary["output_metrics"]["estimated_tokens"] = est_tokens
            context_dict["analysis_summary"] = summary

        # 10. Schema Validation
        validate_workflow_context(
            context_dict,
            final_validation=final_validation,
            strict_evidence=strict_evidence,
        )

        return context_dict

    def serialize_to_json(
        self,
        context_dict: Dict[str, Any],
        indent: int = 2,
    ) -> str:
        """Serialize context dictionary to formatted JSON string."""
        return json.dumps(context_dict, indent=indent)
