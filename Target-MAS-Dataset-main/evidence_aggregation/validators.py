"""
Evidence Validation Module.

Validates the integrity, mathematical consistency, schema compliance,
and evidence provenance of generated workflow_evidence.json files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evidence_aggregation.schema import WorkflowEvidenceDocument


class EvidenceValidator:
    """Validates unified workflow evidence artifacts."""

    def __init__(self, evidence_dir: Path):
        self.evidence_dir = Path(evidence_dir)

    def validate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Validate a single workflow_evidence.json file against all requirements."""
        errors: List[str] = []
        if not file_path.exists():
            return False, [f"File does not exist: {file_path}"]

        try:
            content = file_path.read_text(encoding="utf-8")
            raw_data = json.loads(content)
        except Exception as e:
            return False, [f"Invalid JSON in {file_path}: {str(e)}"]

        # 1. Pydantic Schema Validation
        try:
            doc = WorkflowEvidenceDocument.model_validate(raw_data)
        except Exception as e:
            errors.append(f"Schema validation failed: {str(e)}")
            return False, errors

        # 2. Trace Accounting Check
        if doc.metadata.traces_discovered == 0:
            errors.append("Zero traces discovered.")
        if doc.metadata.traces_processed != doc.metadata.traces_discovered:
            errors.append(
                f"Mismatch: {doc.metadata.traces_discovered} discovered but {doc.metadata.traces_processed} processed."
            )

        # 3. Input Files Verification
        wf_json_path = Path(doc.metadata.input_files.workflow_json)
        if not wf_json_path.exists():
            errors.append(f"Referenced workflow.json does not exist: {wf_json_path}")

        wf_ctx_path = Path(doc.metadata.input_files.workflow_context_json)
        if not wf_ctx_path.exists():
            errors.append(f"Referenced workflow_context.json does not exist: {wf_ctx_path}")

        # 4. Agent Correlation Check
        declared_agents = {a.agent_id for a in doc.workflow_definition.declared_agents}
        static_agents = {a.agent_id for a in doc.static_implementation_context.agents}
        runtime_agents = set(doc.runtime_observations.agent_metrics.keys())

        reconciled_ids = {r.agent_id for r in doc.reconciliation_and_limitations.agent_reconciliation}
        all_ids = declared_agents | static_agents | runtime_agents
        if reconciled_ids != all_ids:
            errors.append(
                f"Reconciliation incomplete: reconciled {len(reconciled_ids)} vs total {len(all_ids)} agents."
            )

        # 5. Metric Mathematical Reproduction Check
        runs = doc.runtime_observations.runs_summary
        if runs:
            # Check sum and mean of duration
            run_durs = [r.duration_seconds for r in runs]
            expected_mean_dur = round(sum(run_durs) / len(run_durs), 4)
            if abs(doc.runtime_observations.workflow_latency.mean_seconds - expected_mean_dur) > 0.01:
                errors.append(
                    f"Workflow latency mean mismatch: reported {doc.runtime_observations.workflow_latency.mean_seconds} vs calculated {expected_mean_dur}"
                )

            # Check sum of tokens
            run_tokens = [r.total_tokens for r in runs]
            expected_tot_tokens = sum(run_tokens)
            if doc.runtime_observations.workflow_tokens.total_tokens != expected_tot_tokens:
                errors.append(
                    f"Workflow total tokens mismatch: reported {doc.runtime_observations.workflow_tokens.total_tokens} vs sum {expected_tot_tokens}"
                )

            # Check sum of costs
            run_costs = [r.estimated_cost_usd for r in runs]
            expected_tot_cost = round(sum(run_costs), 5)
            if abs(doc.runtime_observations.workflow_cost.total_cost_usd - expected_tot_cost) > 0.001:
                errors.append(
                    f"Workflow total cost mismatch: reported {doc.runtime_observations.workflow_cost.total_cost_usd} vs sum {expected_tot_cost}"
                )

        # 6. Check that missing values are not represented as zero
        for aid, am in doc.runtime_observations.agent_metrics.items():
            if am.cost and am.cost.get("total_cost_usd") == 0:
                errors.append(
                    f"Agent {aid} cost represented as 0 instead of None for unavailable agent-level pricing."
                )

        # 7. Graph Node / Edge Integrity Check
        node_ids = {n.id for n in doc.workflow_graph.nodes}
        for edge in doc.workflow_graph.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge source '{edge.source}' not found in graph nodes.")
            if edge.target not in node_ids:
                errors.append(f"Edge target '{edge.target}' not found in graph nodes.")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_all(self) -> Dict[str, Tuple[bool, List[str]]]:
        """Validate all workflow_evidence.json files found in evidence_dir."""
        results: Dict[str, Tuple[bool, List[str]]] = {}
        for wf_dir in sorted(self.evidence_dir.iterdir()):
            if not wf_dir.is_dir():
                continue
            ev_file = wf_dir / "workflow_evidence.json"
            if ev_file.exists():
                results[wf_dir.name] = self.validate_file(ev_file)
            else:
                results[wf_dir.name] = (False, [f"Missing workflow_evidence.json in {wf_dir}"])
        return results

