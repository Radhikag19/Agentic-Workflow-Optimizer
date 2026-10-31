"""
Tests for TraceProfiler AST Ingestion & Workflow Analysis Pipeline.

Validates that static AST analysis covers workflows under Agents/,
achieves agent definition resolution, extracts execution flows,
and generates validated workflow contexts without crashing.
"""

import tempfile
import unittest
from pathlib import Path

from common.settings import PROJECT_ROOT
from ingestion.ast_parser import ASTParser
from ingestion.workflow_analyzer import WorkflowAnalyzer
from ingestion.orchestrator import IngestionOrchestrator

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


class TestASTWorkflowAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orchestrator = IngestionOrchestrator()
        cls.agents_dir = PROJECT_ROOT / "Agents"
        cls.discovered = cls.orchestrator.discover_workflows(cls.agents_dir)

    def test_discover_all_10_workflows(self):
        """Verify exactly 10 workflow directories are discovered."""
        self.assertEqual(len(self.discovered), 10, f"Expected 10 workflows, got {len(self.discovered)}")
        names = {d.name for d in self.discovered}
        self.assertEqual(names, set(EXPECTED_WORKFLOW_AGENT_COUNTS.keys()))

    def test_workflow_analysis_sample(self):
        """Verify single workflow analysis on invoice_processing produces valid results."""
        target_dir = self.agents_dir / "invoice_processing"
        self.assertTrue(target_dir.exists())

        report = self.orchestrator.process_workflow(target_dir, write_output=False)
        self.assertIn(report.status, ("completed", "partial"))
        self.assertEqual(report.candidate_agent_files, EXPECTED_WORKFLOW_AGENT_COUNTS["invoice_processing"])
        self.assertGreater(report.identified_agents, 0)
        self.assertGreater(report.serialized_bytes, 0)
        self.assertGreater(report.estimated_tokens, 0)

    def test_workflow_orchestrator_batch(self):
        """Verify batch orchestrator processes discovered workflows."""
        summary = self.orchestrator.process_all_workflows(self.agents_dir, write_output=False)
        self.assertEqual(summary.total_workflows, 10)
        self.assertEqual(summary.failed_workflows, 0)
        self.assertGreaterEqual(summary.successful_workflows, 8)


if __name__ == "__main__":
    unittest.main()
