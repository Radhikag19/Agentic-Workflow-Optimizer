"""
Unit tests for AgentExtractor and framework pattern recognition.
"""

import tempfile
import unittest
from pathlib import Path

from ingestion.ast_parser import ASTParser
from ingestion.agent_extractor import AgentExtractor


class TestAgentIdentification(unittest.TestCase):
    def setUp(self):
        self.parser = ASTParser()
        self.extractor = AgentExtractor(self.parser.cache)

    def test_crewai_agent_constructor_detection(self):
        """Verify CrewAI Agent(...) constructor extraction with role, goal, backstory."""
        code = """
from crewai import Agent

AGENT_ID = "invoice_extraction_agent"

def create_agent(llm=None):
    return Agent(
        role="Senior Extraction Specialist",
        goal="Accurately extract all structured invoice fields.",
        backstory="An expert document auditor parsing accounts payable.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write(code)
            f_path = Path(f.name)

        try:
            mod = self.parser.parse_file(f_path)
            agents = self.extractor.extract_from_module(mod)

            self.assertEqual(len(agents), 1)
            agent = agents[0]
            self.assertEqual(agent.role, "Senior Extraction Specialist")
            self.assertIn("structured invoice", agent.goal)
            self.assertIn("document auditor", agent.backstory)
            self.assertEqual(agent.definition_type, "function")
            self.assertGreaterEqual(agent.start_line, 1)
        finally:
            f_path.unlink(missing_ok=True)

    def test_no_false_positive_agents(self):
        """Verify helper classes or functions with 'agent' in their name are not falsely classified as agents."""
        code = """
class AgentDatabaseHelper:
    '''Utility for querying agent logs.'''
    def get_agent_record(self, agent_id):
        return {"id": agent_id, "status": "active"}

def run_agent_monitor():
    return "Monitoring active agents"
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write(code)
            f_path = Path(f.name)

        try:
            mod = self.parser.parse_file(f_path)
            agents = self.extractor.extract_from_module(mod)

            # Neither AgentDatabaseHelper nor run_agent_monitor construct an agent
            self.assertEqual(len(agents), 0, "Non-agent classes/functions must not be classified as agents")
        finally:
            f_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
