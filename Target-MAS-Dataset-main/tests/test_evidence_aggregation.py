"""
Unit and Integration Tests for Unified Evidence Aggregation.

Validates that:
1. All 10 workflows are discovered and aggregated.
2. workflow_evidence.json conforms to the Pydantic schema.
3. All available runtime traces are accounted for without dropping or fabricating.
4. Calculated aggregates match mathematical sums/means of underlying observations.
5. Missing values are represented as None, not mixed with zero.
6. Graph relationships preserve provenance.
"""

import json
from pathlib import Path
import pytest

from evidence_aggregation.aggregator import EvidenceAggregator
from evidence_aggregation.validators import EvidenceValidator
from evidence_aggregation.schema import WorkflowEvidenceDocument


@pytest.fixture(scope="module")
def aggregator():
    return EvidenceAggregator()


@pytest.fixture(scope="module")
def generated_evidence(aggregator):
    return aggregator.generate_all()


def test_workflow_discovery(aggregator):
    """Verify that all 10 target workflows are discovered."""
    workflows = aggregator.discover_workflows()
    assert len(workflows) == 10
    expected = {
        "contract_review",
        "customer_support",
        "expense_reimbursement",
        "incident_triage",
        "inventory_replenishment",
        "invoice_processing",
        "market_research",
        "procurement_supplier_selection",
        "recruitment_screening",
        "sales_lead_qualification",
    }
    assert set(workflows) == expected


def test_evidence_generation_and_validation(aggregator, generated_evidence):
    """Verify that workflow_evidence.json is generated and valid for all 10 workflows."""
    assert len(generated_evidence) == 10
    validator = EvidenceValidator(aggregator.output_dir)
    results = validator.validate_all()

    for wf_name, (is_valid, errors) in results.items():
        assert is_valid, f"Workflow '{wf_name}' validation failed: {errors}"


def test_trace_accounting(generated_evidence):
    """Verify that all 30 traces are accounted for in each workflow."""
    for wf_name, ev_path in generated_evidence.items():
        data = json.loads(ev_path.read_text(encoding="utf-8"))
        meta = data["metadata"]
        runtime = data["runtime_observations"]

        assert meta["traces_discovered"] == 30, f"{wf_name} did not discover 30 traces"
        assert meta["traces_processed"] == 30, f"{wf_name} did not process 30 traces"
        assert meta["traces_failed"] == 0, f"{wf_name} had failed traces"
        assert runtime["total_runs"] == 30, f"{wf_name} total_runs mismatch"
        assert runtime["successful_runs"] == 30, f"{wf_name} successful_runs mismatch"


def test_mathematical_metrics_reproduction(generated_evidence):
    """Verify that reported workflow aggregates equal the exact mathematical calculation of run samples."""
    for wf_name, ev_path in generated_evidence.items():
        data = json.loads(ev_path.read_text(encoding="utf-8"))
        runtime = data["runtime_observations"]
        runs = runtime["runs_summary"]

        # 1. Total tokens must equal sum of run tokens
        sum_tokens = sum(r["total_tokens"] for r in runs)
        assert runtime["workflow_tokens"]["total_tokens"] == sum_tokens

        # 2. Total cost must equal sum of run costs
        sum_cost = round(sum(r["estimated_cost_usd"] for r in runs), 5)
        assert abs(runtime["workflow_cost"]["total_cost_usd"] - sum_cost) < 0.001

        # 3. Mean duration must equal sum of durations / count
        mean_dur = round(sum(r["duration_seconds"] for r in runs) / len(runs), 4)
        assert abs(runtime["workflow_latency"]["mean_seconds"] - mean_dur) < 0.01


def test_missing_values_not_zero(generated_evidence):
    """Verify that unavailable metrics (e.g. agent-level cost) are None, not 0."""
    for wf_name, ev_path in generated_evidence.items():
        data = json.loads(ev_path.read_text(encoding="utf-8"))
        agent_metrics = data["runtime_observations"]["agent_metrics"]
        for aid, am in agent_metrics.items():
            cost_info = am.get("cost", {})
            # total_cost_usd must be None because individual agent cost is not tracked in traces
            assert cost_info.get("total_cost_usd") is None, f"{wf_name}:{aid} cost was not None"


def test_reconciliation_integrity(generated_evidence):
    """Verify that reconciliation covers all declared, static, and runtime agents."""
    for wf_name, ev_path in generated_evidence.items():
        data = json.loads(ev_path.read_text(encoding="utf-8"))
        reconciliation = data["reconciliation_and_limitations"]
        agent_recs = reconciliation["agent_reconciliation"]

        dec_agents = {a["agent_id"] for a in data["workflow_definition"]["declared_agents"]}
        stat_agents = {a["agent_id"] for a in data["static_implementation_context"]["agents"]}
        run_agents = set(data["runtime_observations"]["agent_metrics"].keys())
        expected_all = dec_agents | stat_agents | run_agents

        reconciled_ids = {r["agent_id"] for r in agent_recs}
        assert reconciled_ids == expected_all


def test_graph_provenance_and_connectivity(generated_evidence):
    """Verify that graph nodes and edges have valid provenance and no broken node references."""
    for wf_name, ev_path in generated_evidence.items():
        data = json.loads(ev_path.read_text(encoding="utf-8"))
        graph = data["workflow_graph"]
        node_ids = {n["id"] for n in graph["nodes"]}

        for edge in graph["edges"]:
            assert edge["source"] in node_ids, f"Broken source {edge['source']} in {wf_name}"
            assert edge["target"] in node_ids, f"Broken target {edge['target']} in {wf_name}"
            assert edge["provenance"] in {"declared", "statically_observed", "runtime_observed"}

