"""
Unit tests for code evidence selection, line-range preservation, and helper tracing.
"""

import tempfile
import unittest
from pathlib import Path

from ingestion.ast_parser import ASTParser
from ingestion.agent_extractor import AgentExtractor
from ingestion.workflow_analyzer import WorkflowAnalyzer


class TestCodeEvidence(unittest.TestCase):
    def setUp(self):
        self.parser = ASTParser()
        self.agent_extractor = AgentExtractor(self.parser.cache)
        self.analyzer = WorkflowAnalyzer(self.parser)

    def test_evidence_selection_and_helper_tracing(self):
        """Verify AST line-range preservation and helper function classification."""
        code = """
from crewai import Agent, Task

def deterministic_math_check(data):
    # Important arithmetic calculation
    return data.get("qty", 0) * data.get("price", 0)

def create_agent(llm=None):
    return Agent(role="Auditor", goal="Audit math", backstory="Accountant", llm=llm)

def create_task(agent, invoice_data):
    math_result = deterministic_math_check(invoice_data)
    return Task(description=f"Validate {math_result}", expected_output="JSON", agent=agent)
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write(code)
            f_path = Path(f.name)

        try:
            mod = self.parser.parse_file(f_path)
            agents = self.agent_extractor.extract_from_module(mod)

            self.assertEqual(len(agents), 1)
            agent = agents[0]

            # Line range preserved
            self.assertGreaterEqual(agent.start_line, 1)
            self.assertGreaterEqual(agent.end_line, agent.start_line)

            # Helper function deterministic_math_check is captured
            self.assertIn("deterministic_math_check", agent.helper_functions)

            # Task definition captured
            self.assertEqual(len(agent.tasks), 1)
            task = agent.tasks[0]
            self.assertEqual(task.expected_output, "JSON")
            self.assertGreaterEqual(task.start_line, 1)
        finally:
            f_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
