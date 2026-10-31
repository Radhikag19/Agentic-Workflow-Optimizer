"""
TraceProfiler Ingestion Orchestrator Module.

Coordinates the complete ingestion pipeline across all multi-agent workflows:
discovers workflows, parses sources, extracts structures, generates validated
workflow_context.json files, isolates failures, and reports execution summaries.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from schemas.workflow_context_schema import (
    ValidationError,
    validate_workflow_context,
)
from ingestion.ast_parser import ASTParser, default_ast_parser
from ingestion.workflow_analyzer import WorkflowAnalyzer
from ingestion.context_generator import ContextGenerator


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WorkflowExecutionReport:
    workflow_id: str
    workflow_dir: str
    output_path: Optional[str]
    status: str  # "completed" | "partial" | "failed"
    total_files: int = 0
    candidate_agent_files: int = 0
    identified_agents: int = 0
    invoked_agents: int = 0
    uninvoked_agents: int = 0
    total_tasks: int = 0
    total_dependencies: int = 0
    serialized_bytes: int = 0
    estimated_tokens: int = 0
    error_message: Optional[str] = None


@dataclass
class IngestionBatchSummary:
    total_workflows: int = 0
    successful_workflows: int = 0
    partial_workflows: int = 0
    failed_workflows: int = 0
    reports: List[WorkflowExecutionReport] = field(default_factory=list)


# ============================================================================
# Ingestion Orchestrator Engine
# ============================================================================

class IngestionOrchestrator:
    """Discovers, analyzes, and produces validated workflow_context.json files."""

    def __init__(self, parser: Optional[ASTParser] = None) -> None:
        self.parser = parser or default_ast_parser
        self.analyzer = WorkflowAnalyzer(parser=self.parser)
        self.generator = ContextGenerator()

    def discover_workflows(self, root_dir: Union[str, Path]) -> List[Path]:
        """
        Locate all workflow directories under root_dir.
        A workflow directory is any subdirectory containing a `workflow.py`.
        """
        root_path = Path(root_dir).resolve()
        discovered: List[Path] = []

        if not root_path.exists():
            return []

        # If root_path itself contains workflow.py
        if (root_path / "workflow.py").is_file():
            return [root_path]

        # Scan subdirectories
        for item in sorted(root_path.iterdir()):
            if item.is_dir() and not item.name.startswith((".", "__")):
                if (item / "workflow.py").is_file():
                    discovered.append(item)

        return discovered

    def process_workflow(
        self,
        workflow_dir: Union[str, Path],
        write_output: bool = True,
    ) -> WorkflowExecutionReport:
        """
        Analyze a single workflow directory and generate workflow_context.json.
        Preserves isolation: failures do not crash the batch.
        """
        wf_path = Path(workflow_dir).resolve()
        wf_id = wf_path.name
        output_file = wf_path / "workflow_context.json"

        try:
            # 1. Analyze workflow directory
            analysis_res = self.analyzer.analyze_workflow_dir(wf_path)

            # Determine status: if workflow.py failed to parse or no files parsed, workflow failed
            workflow_py_failed = any(f["path"] == "workflow.py" for f in analysis_res.failed_files)
            if workflow_py_failed or not analysis_res.parsed_files:
                status = "failed"
            elif analysis_res.failed_files:
                status = "partial"
            else:
                status = "completed"

            is_completed = (status == "completed")
            final_val = is_completed
            context_dict = self.generator.generate_context(
                analysis=analysis_res,
                final_validation=final_val,
                strict_evidence=True,
            )

            # 3. Write output to disk if requested
            if write_output:
                json_content = self.generator.serialize_to_json(context_dict, indent=2)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(json_content)

            summary = context_dict["analysis_summary"]
            metrics = summary["output_metrics"]
            inv = context_dict["agent_inventory"]

            return WorkflowExecutionReport(
                workflow_id=wf_id,
                workflow_dir=str(wf_path),
                output_path=str(output_file) if write_output else None,
                status=status,
                total_files=context_dict["source_inventory"]["total_python_files"],
                candidate_agent_files=inv["discovered_agent_files"],
                identified_agents=inv["identified_agent_definitions"],
                invoked_agents=inv["workflow_invoked_agents"],
                uninvoked_agents=inv["uninvoked_agent_definitions"],
                total_tasks=summary["total_tasks"],
                total_dependencies=summary["total_dependencies"],
                serialized_bytes=metrics["serialized_bytes"],
                estimated_tokens=metrics["estimated_tokens"],
                error_message=None,
            )

        except Exception as e:
            return WorkflowExecutionReport(
                workflow_id=wf_id,
                workflow_dir=str(wf_path),
                output_path=None,
                status="failed",
                error_message=str(e),
            )

    def process_all_workflows(
        self,
        agents_dir: Union[str, Path],
        write_output: bool = True,
    ) -> IngestionBatchSummary:
        """Process all discovered workflows with isolation and batch reporting."""
        workflow_dirs = self.discover_workflows(agents_dir)
        batch = IngestionBatchSummary(total_workflows=len(workflow_dirs))

        for wf_dir in workflow_dirs:
            report = self.process_workflow(wf_dir, write_output=write_output)
            batch.reports.append(report)
            if report.status == "completed":
                batch.successful_workflows += 1
            elif report.status == "partial":
                batch.partial_workflows += 1
            else:
                batch.failed_workflows += 1

        return batch


# ============================================================================
# CLI Entrypoint
# ============================================================================

def main() -> None:
    """CLI entrypoint for running ingestion across workflows."""
    parser = argparse.ArgumentParser(description="TraceProfiler AST Ingestion Pipeline")
    parser.add_argument(
        "target",
        nargs="?",
        default="Agents",
        help="Target Agents directory or specific workflow directory (default: Agents)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run analysis and validation in memory without writing to disk",
    )
    args = parser.parse_args()

    target_path = Path(args.target).resolve()
    if not target_path.exists():
        print(f"Error: Target path does not exist: {target_path}", file=sys.stderr)
        sys.exit(1)

    orchestrator = IngestionOrchestrator()
    print(f"Starting TraceProfiler Ingestion for: {target_path}")

    batch = orchestrator.process_all_workflows(target_path, write_output=not args.no_write)

    print("\n" + "=" * 80)
    print("TraceProfiler Ingestion Execution Summary")
    print("=" * 80)
    print(f"Total Workflows Discovered: {batch.total_workflows}")
    print(f"Successful (Completed):    {batch.successful_workflows}")
    print(f"Partial (Completed w/err): {batch.partial_workflows}")
    print(f"Failed:                    {batch.failed_workflows}")
    print("-" * 80)
    print(f"{'Workflow ID':<32} {'Status':<10} {'Agents':<8} {'Invoked':<8} {'Tasks':<8} {'Deps':<8} {'Bytes':<8}")
    print("-" * 80)

    for r in batch.reports:
        if r.status != "failed":
            print(f"{r.workflow_id:<32} {r.status:<10} {r.identified_agents:<8} {r.invoked_agents:<8} {r.total_tasks:<8} {r.total_dependencies:<8} {r.serialized_bytes:<8}")
        else:
            print(f"{r.workflow_id:<32} {r.status:<10} ERROR: {r.error_message}")

    print("=" * 80)

    if batch.failed_workflows > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
