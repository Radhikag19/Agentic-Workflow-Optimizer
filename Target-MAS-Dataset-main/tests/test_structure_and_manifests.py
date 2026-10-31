"""
Tests verifying workflow structure, agent counts, unique IDs, and workflow.json manifests across all 10 workflows.
"""

import json
from pathlib import Path
import pytest

from common.settings import MANDATORY_MODEL, PROJECT_ROOT
from common.api_key_manager import AGENT_WORKFLOW_MAP

EXPECTED_WORKFLOW_AGENT_COUNTS = {
    "invoice_processing": 8,
    "customer_support": 6,
    "procurement_supplier_selection": 9,
    "sales_lead_qualification": 7,
    "recruitment_screening": 8,
    "inventory_replenishment": 6,
    "contract_review": 10,
    "incident_triage": 7,
    "expense_reimbursement": 8,
    "market_research": 9,
}


def test_total_78_agents_registered():
    """Verify exactly 78 agents registered across all 10 workflows."""
    assert len(AGENT_WORKFLOW_MAP) == 78, f"Expected 78 agents, got {len(AGENT_WORKFLOW_MAP)}"
    agent_ids = list(AGENT_WORKFLOW_MAP.keys())
    assert len(set(agent_ids)) == 78, "Agent IDs must all be unique"


def test_agent_counts_per_workflow():
    """Verify each of the 10 workflows has its exact required agent count."""
    for wf, count in EXPECTED_WORKFLOW_AGENT_COUNTS.items():
        wf_agents = [aid for aid, w in AGENT_WORKFLOW_MAP.items() if w == wf]
        assert len(wf_agents) == count, f"Workflow '{wf}' expected {count} agents, got {len(wf_agents)}"


def test_workflow_manifests_exist_and_valid():
    """Verify all 10 workflow.json files exist, are valid JSON, and match requirements."""
    agents_dir = PROJECT_ROOT / "Agents"
    for wf, expected_count in EXPECTED_WORKFLOW_AGENT_COUNTS.items():
        manifest_path = agents_dir / wf / "workflow.json"
        assert manifest_path.exists(), f"Missing manifest at {manifest_path}"

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data.get("workflow_name") == wf
        assert data.get("fixed_model") == MANDATORY_MODEL
        assert data.get("total_agents") == expected_count
        assert len(data.get("agents", [])) == expected_count
        assert len(data.get("execution_order", [])) == expected_count
        assert "optimization_opportunities" in data
