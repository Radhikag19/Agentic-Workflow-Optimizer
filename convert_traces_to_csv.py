"""
JSON-to-CSV Conversion Pipeline for Multi-Agent Workflow Execution Traces.

Converts hierarchical JSON execution traces into tabular CSV datasets where:
- Each row represents one agent execution step in a workflow run.
- Workflow-level metadata is merged into every corresponding agent row.
- Exact specified schema keys are extracted, omitting internal payloads like 'output'.
- Missing or null values are preserved cleanly without fabrication.
- Modular architecture allowing single-workflow execution or batch conversion across all workflows.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("TraceToCSV")

# 1. Workflow-level fields (exactly as specified)
WORKFLOW_FIELDS = [
    "run_id",
    "workflow_name",
    "model",
    "status",
    "duration_seconds",
    "agent_execution_count",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "estimated_cost_usd",
    "error_message",
    "timestamp",
]

# 2. Agent-level fields inside agent_executions (exactly as specified)
AGENT_FIELDS = [
    "status",
    "duration_seconds",
    "tokens",
    "prompt_tokens",
    "completion_tokens",
    "notes",
    "order",
    "agent_id",
]

# Fields that collide between workflow and agent scopes
COLLIDING_FIELDS = {"status", "duration_seconds", "prompt_tokens", "completion_tokens"}

# Disambiguated CSV column header mapping (preserves exact order and original key names)
# Workflow scope: colliding keys prefixed with workflow_
# Agent scope: colliding keys prefixed with agent_
DISAMBIGUATED_COLUMNS = [
    # Workflow-level fields
    "run_id",
    "workflow_name",
    "model",
    "workflow_status",
    "workflow_duration_seconds",
    "agent_execution_count",
    "total_tokens",
    "workflow_prompt_tokens",
    "workflow_completion_tokens",
    "estimated_cost_usd",
    "error_message",
    "timestamp",
    # Agent-level fields
    "agent_status",
    "agent_duration_seconds",
    "tokens",
    "agent_prompt_tokens",
    "agent_completion_tokens",
    "notes",
    "order",
    "agent_id",
]

# Agent-prefixed column mapping (workflow keys verbatim, agent colliding keys prefixed with agent_)
AGENT_PREFIXED_COLUMNS = [
    "run_id",
    "workflow_name",
    "model",
    "status",
    "duration_seconds",
    "agent_execution_count",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "estimated_cost_usd",
    "error_message",
    "timestamp",
    "agent_status",
    "agent_duration_seconds",
    "tokens",
    "agent_prompt_tokens",
    "agent_completion_tokens",
    "notes",
    "order",
    "agent_id",
]


class TraceToCSVConverter:
    """Modular converter that transforms workflow execution JSON traces to CSV format."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        output_dir_name: str = "worklow_traces_csv",
        naming_style: str = "disambiguated",
    ):
        """
        Initialize the converter.

        :param base_dir: Root directory of the repository (auto-detected if None).
        :param output_dir_name: Name of the output directory (defaults to 'worklow_traces_csv').
        :param naming_style: 'disambiguated' (default: workflow_status & agent_status)
                             or 'agent_prefixed' (status & agent_status).
        """
        self.base_dir = Path(base_dir).resolve() if base_dir else self._find_project_root()
        self.output_dir_name = output_dir_name
        self.naming_style = naming_style
        self.columns = (
            DISAMBIGUATED_COLUMNS
            if naming_style == "disambiguated"
            else AGENT_PREFIXED_COLUMNS
        )

    @staticmethod
    def _find_project_root() -> Path:
        """Locates the repository root by checking for Ingestion_Traces_final."""
        cwd = Path.cwd().resolve()
        for candidate in [cwd, cwd.parent, cwd / "Target-MAS-Dataset-main"]:
            if (candidate / "Ingestion_Traces_final").exists():
                return candidate
        return cwd

    def find_workflow_traces(self, workflow_name: str, custom_input_dir: Optional[Path] = None) -> List[Path]:
        """
        Find all JSON trace run files for a given workflow.
        Skips batch summary and non-run JSON files.
        """
        candidate_dirs = []
        if custom_input_dir:
            candidate_dirs.append(Path(custom_input_dir).resolve())

        # Standard discovery paths in order of preference
        candidate_dirs.extend([
            self.base_dir / "Ingestion_Traces_final" / "Ingestion_Traces" / workflow_name,
            self.base_dir / "Target-MAS-Dataset-main" / "runs" / workflow_name,
            self.base_dir / "runs" / workflow_name,
        ])

        found_files: List[Path] = []
        for d in candidate_dirs:
            if d.exists() and d.is_dir():
                json_files = sorted(d.glob("*.json"))
                # Filter to trace run files (files containing 'agent_executions' and 'run_id')
                for f in json_files:
                    if f.name.startswith("batch_") or f.name.endswith("_summary.json"):
                        continue
                    try:
                        with open(f, "r", encoding="utf-8") as fp:
                            content = json.load(fp)
                        if isinstance(content, dict) and "agent_executions" in content and "run_id" in content:
                            found_files.append(f)
                    except Exception as e:
                        logger.warning(f"Skipping unreadable or corrupt JSON file {f.name}: {e}")
                if found_files:
                    logger.info(f"Found {len(found_files)} trace files in: {d}")
                    break

        return found_files

    def extract_row(self, workflow_data: Dict[str, Any], agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts and aligns workflow-level and agent-level fields into a single record.
        Strictly excludes 'output', 'assigned_key_identifier', and other unlisted keys.
        Missing or null values are preserved cleanly without fabrication.
        """
        row: Dict[str, Any] = {}

        if self.naming_style == "disambiguated":
            # Workflow-level
            row["run_id"] = workflow_data.get("run_id")
            row["workflow_name"] = workflow_data.get("workflow_name")
            row["model"] = workflow_data.get("model")
            row["workflow_status"] = workflow_data.get("status")
            row["workflow_duration_seconds"] = workflow_data.get("duration_seconds")
            row["agent_execution_count"] = workflow_data.get("agent_execution_count")
            row["total_tokens"] = workflow_data.get("total_tokens")
            row["workflow_prompt_tokens"] = workflow_data.get("prompt_tokens")
            row["workflow_completion_tokens"] = workflow_data.get("completion_tokens")
            row["estimated_cost_usd"] = workflow_data.get("estimated_cost_usd")
            row["error_message"] = workflow_data.get("error_message")
            row["timestamp"] = workflow_data.get("timestamp")

            # Agent-level
            row["agent_status"] = agent_data.get("status")
            row["agent_duration_seconds"] = agent_data.get("duration_seconds")
            row["tokens"] = agent_data.get("tokens")
            row["agent_prompt_tokens"] = agent_data.get("prompt_tokens")
            row["agent_completion_tokens"] = agent_data.get("completion_tokens")
            row["notes"] = agent_data.get("notes")
            row["order"] = agent_data.get("order")
            row["agent_id"] = agent_data.get("agent_id")

        else:  # agent_prefixed
            # Workflow-level
            row["run_id"] = workflow_data.get("run_id")
            row["workflow_name"] = workflow_data.get("workflow_name")
            row["model"] = workflow_data.get("model")
            row["status"] = workflow_data.get("status")
            row["duration_seconds"] = workflow_data.get("duration_seconds")
            row["agent_execution_count"] = workflow_data.get("agent_execution_count")
            row["total_tokens"] = workflow_data.get("total_tokens")
            row["prompt_tokens"] = workflow_data.get("prompt_tokens")
            row["completion_tokens"] = workflow_data.get("completion_tokens")
            row["estimated_cost_usd"] = workflow_data.get("estimated_cost_usd")
            row["error_message"] = workflow_data.get("error_message")
            row["timestamp"] = workflow_data.get("timestamp")

            # Agent-level
            row["agent_status"] = agent_data.get("status")
            row["agent_duration_seconds"] = agent_data.get("duration_seconds")
            row["tokens"] = agent_data.get("tokens")
            row["agent_prompt_tokens"] = agent_data.get("prompt_tokens")
            row["agent_completion_tokens"] = agent_data.get("completion_tokens")
            row["notes"] = agent_data.get("notes")
            row["order"] = agent_data.get("order")
            row["agent_id"] = agent_data.get("agent_id")

        return row

    def convert_workflow(
        self,
        workflow_name: str,
        custom_input_dir: Optional[Path] = None,
        custom_output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Converts all JSON traces for a given workflow into a single consolidated CSV file.

        :param workflow_name: Identifier of the workflow (e.g. 'customer_support').
        :param custom_input_dir: Optional custom input directory.
        :param custom_output_dir: Optional custom output directory.
        :return: Path to the generated CSV file.
        """
        trace_files = self.find_workflow_traces(workflow_name, custom_input_dir)
        if not trace_files:
            raise FileNotFoundError(f"No valid JSON trace files found for workflow '{workflow_name}'.")

        # Determine output directory: worklow_traces_csv/<workflow_name>/
        if custom_output_dir:
            out_subdir = Path(custom_output_dir).resolve()
        else:
            out_subdir = self.base_dir / self.output_dir_name / workflow_name

        out_subdir.mkdir(parents=True, exist_ok=True)
        csv_file_path = out_subdir / f"{workflow_name}.csv"

        records: List[Dict[str, Any]] = []
        total_runs_processed = 0

        for trace_file in trace_files:
            with open(trace_file, "r", encoding="utf-8") as fp:
                trace_data = json.load(fp)

            # Workflow metadata
            wf_meta = {k: trace_data.get(k) for k in WORKFLOW_FIELDS}

            # Agent executions list (dynamically adapts to any number of agents: 6, 8, 10, etc.)
            agent_executions = trace_data.get("agent_executions", [])
            for agent in agent_executions:
                row = self.extract_row(wf_meta, agent)
                records.append(row)

            total_runs_processed += 1

        # Write to CSV cleanly
        with open(csv_file_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.columns)
            writer.writeheader()
            for rec in records:
                # None values write as empty string by default in csv.DictWriter
                writer.writerow(rec)

        logger.info(
            f"Successfully generated CSV for '{workflow_name}': "
            f"{len(records)} agent rows across {total_runs_processed} runs -> {csv_file_path}"
        )

        return csv_file_path

    def convert_all(self, custom_input_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Discovers all available workflows and converts their traces to CSV.
        """
        results: Dict[str, Path] = {}
        base_traces = self.base_dir / "Ingestion_Traces_final" / "Ingestion_Traces"

        if not base_traces.exists():
            base_traces = self.base_dir / "Target-MAS-Dataset-main" / "runs"

        if base_traces.exists():
            for folder in sorted(base_traces.iterdir()):
                if folder.is_dir() and not folder.name.startswith("."):
                    try:
                        csv_path = self.convert_workflow(folder.name, custom_input_dir=custom_input_dir)
                        results[folder.name] = csv_path
                    except FileNotFoundError:
                        continue
                    except Exception as exc:
                        logger.error(f"Error converting workflow '{folder.name}': {exc}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert multi-agent workflow execution JSON traces to structured CSV format."
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default="customer_support",
        help="Workflow name to convert (default: customer_support).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all available workflows in the traces directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Optional custom input directory containing the trace JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional custom output directory to store the generated CSV file.",
    )
    parser.add_argument(
        "--naming-style",
        type=str,
        choices=["disambiguated", "agent_prefixed"],
        default="disambiguated",
        help="Column header naming style: 'disambiguated' (default) or 'agent_prefixed'.",
    )

    args = parser.parse_args()

    converter = TraceToCSVConverter(naming_style=args.naming_style)

    input_dir = Path(args.input_dir) if args.input_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.all:
        logger.info("Converting all workflows...")
        results = converter.convert_all(custom_input_dir=input_dir)
        print(f"\n[Summary] Converted {len(results)} workflows:")
        for wf, path in results.items():
            print(f"  - {wf}: {path}")
    else:
        logger.info(f"Converting workflow: {args.workflow}")
        csv_path = converter.convert_workflow(
            workflow_name=args.workflow,
            custom_input_dir=input_dir,
            custom_output_dir=output_dir,
        )
        print(f"\n[Success] Output CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

