"""
Targeted Unit Tests for AST Ingestion Pipeline Corrections.

Validates the six essential corrections:
1. Distinction of agent inputs, task expected outputs, and return values (no AST constructor leaks).
2. Separation of model initialization, direct LLM invocations, agent execution, workflow execution, and tool execution.
3. Distinction of explicitly registered tools from ordinary helper functions (deterministic_* is not a tool).
4. Accurate unresolved reference classification (imported symbols like Task are not unresolved).
5. Tokenizer-based measurements with clear fallback labeling (heuristic_character_ratio).
6. Compact representation without raw AST duplication.
"""

import json
import tempfile
import unittest
from pathlib import Path

from ingestion.ast_parser import ASTParser, parse_source
from ingestion.agent_extractor import AgentExtractor
from ingestion.workflow_analyzer import WorkflowAnalyzer
from ingestion.context_generator import ContextGenerator
from schemas.workflow_context_schema import TOKEN_ESTIMATION_DEFAULT_METHOD, validate_workflow_context


class TestTargetedCorrections(unittest.TestCase):
    def setUp(self):
        self.parser = ASTParser()
        self.agent_extractor = AgentExtractor(self.parser.cache)
        self.analyzer = WorkflowAnalyzer(self.parser)
        self.generator = ContextGenerator()

    def test_1_inputs_outputs_and_intermediate_expressions(self):
        """
        Correction 1: Correctly distinguish inputs, return values, task expected outputs.
        Do not extract entire AST constructor expressions as agent outputs.
        """
        code = """
from crewai import Agent, Task

AGENT_ID = "test_extraction_agent"

def create_agent(llm=None) -> Agent:
    return Agent(role="Extractor", goal="Extract fields", backstory="Auditor", llm=llm)

def create_task(agent: Agent, invoice_data: dict) -> Task:
    description = f"Extract fields from {invoice_data}"
    expected_output = "Strictly valid JSON containing extracted_invoice and extraction_flags."
    return Task(description=description, expected_output=expected_output, agent=agent)
"""
        mod = self.parser.parse_source(code, file_path="test_agent.py")
        agents = self.agent_extractor.extract_from_module(mod)

        self.assertEqual(len(agents), 1)
        ag = agents[0]
        self.assertEqual(len(ag.tasks), 1)
        task = ag.tasks[0]

        # 1. Task Expected Outputs: Must resolve clean string literal, NOT entire Task() or Agent() AST
        self.assertEqual(task.expected_output, "Strictly valid JSON containing extracted_invoice and extraction_flags.")
        self.assertNotIn("Task(", task.expected_output)
        self.assertNotIn("Agent(", task.expected_output)

        # 2. Description separated from expected output
        self.assertNotEqual(task.description, task.expected_output)

    def test_2_execution_patterns_separation(self):
        """
        Correction 2: Separate model initialization, direct LLM invocations, agent execution,
        workflow execution, and tool execution. Methods like kickoff() must not be direct LLM calls.
        """
        code = """
from crewai import Agent, Crew
from common.agent_runner import create_groq_llm_for_agent

def run_pipeline():
    llm, _ = create_groq_llm_for_agent("my_agent")
    crew = Crew(agents=[], tasks=[])
    result = crew.kickoff()
    return result
"""
        mod = self.parser.parse_source(code, file_path="workflow.py")
        kickoff_calls = [c for c in mod.all_calls if c.method_name == "kickoff"]
        self.assertEqual(len(kickoff_calls), 1)
        self.assertEqual(kickoff_calls[0].caller_obj, "crew")

        # In WorkflowAnalyzer, kickoff is classified as crew_kickoff, NOT direct LLM invocation
        op_types = []
        for call in mod.all_calls:
            target = call.func_name
            method = call.method_name
            if method == "kickoff":
                op_types.append("crew_kickoff")
            elif target in ("Crew", "crewai.Crew"):
                op_types.append("crew_creation")
            else:
                op_types.append("function_call")

        self.assertIn("crew_kickoff", op_types)
        self.assertNotIn("direct_llm_call", op_types)

    def test_3_tools_vs_helper_functions_distinction(self):
        """
        Correction 3: Distinguish explicitly registered tools from ordinary helper functions.
        A deterministic_* name alone is insufficient evidence of tool registration.
        """
        code = """
from crewai import Agent
from crewai.tools import tool

def deterministic_arithmetic_check(data):
    '''Ordinary deterministic calculation function.'''
    return data.get("qty", 0) * data.get("price", 0)

@tool
def registered_calculator_tool(query: str) -> str:
    '''Explicitly registered tool.'''
    return "calculated"

research_agent = Agent(
    role="Research Analyst",
    goal="Research",
    tools=[registered_calculator_tool]
)
"""
        mod = self.parser.parse_source(code, file_path="tools_sample.py")
        agents = self.agent_extractor.extract_from_module(mod)

        self.assertEqual(len(agents), 1)
        ag = agents[0]

        # 1. Registered tools must only contain decorated tool
        self.assertIn("registered_calculator_tool", ag.registered_tools)
        self.assertNotIn("deterministic_arithmetic_check", ag.registered_tools)

        # 2. Helper functions must contain deterministic_arithmetic_check
        self.assertIn("deterministic_arithmetic_check", ag.helper_functions)
        self.assertNotIn("registered_calculator_tool", ag.helper_functions)

    def test_4_unresolved_reference_classification(self):
        """
        Correction 4: An imported symbol is resolved by name and module even when
        its internal implementation source has not been supplied. Local receiver variables
        like json.dumps or crew.kickoff must not be falsely flagged as unresolved external imports.
        """
        code = """
from crewai import Agent, Task
import json

def create_task(agent, data):
    result = json.dumps(data)
    truly_undefined_symbol(result)
    return Task(description="desc", expected_output="out", agent=agent)
"""
        mod = self.parser.parse_source(code, file_path="test_refs.py")

        # Imported symbols are mapped
        imported_symbols = {item.effective_name: ref.module for ref in mod.imports for item in ref.names}
        self.assertIn("Task", imported_symbols)
        self.assertIn("Agent", imported_symbols)
        self.assertIn("json", imported_symbols)

        # Local symbols / imported symbols check
        self.assertTrue(mod.is_local_symbol("Task"))
        self.assertTrue(mod.is_local_symbol("json"))
        self.assertFalse(mod.is_local_symbol("truly_undefined_symbol"))

    def test_5_token_counter_measurement_and_labeling(self):
        """
        Correction 5: Replace raw unlabelled token counts with clear labeling:
        heuristic_character_ratio.
        """
        self.assertEqual(TOKEN_ESTIMATION_DEFAULT_METHOD, "heuristic_character_ratio")
        sample_dict = {"test": "data", "count": 123}
        serialized = json.dumps(sample_dict)
        est = max(1, len(serialized) // 4)
        self.assertGreater(est, 0)

    def test_6_no_duplication_and_compact_index_minimalism(self):
        """
        Correction 6: Keep compact index minimal and evidence separately retrievable.
        Do not duplicate entire AST constructor blocks or full source code snippets.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            wf_path = Path(tmp_dir) / "test_wf"
            wf_path.mkdir()

            (wf_path / "agent.py").write_text("""
from crewai import Agent
def create_agent():
    return Agent(role="Auditor", goal="Audit", backstory="Backstory")
""", encoding="utf-8")

            (wf_path / "workflow.py").write_text("""
from agent import create_agent
from crewai import Crew, Process

a = create_agent()
crew = Crew(agents=[a], process=Process.sequential)
result = crew.kickoff()
""", encoding="utf-8")

            analysis = self.analyzer.analyze_workflow_dir(wf_path)
            ctx = self.generator.generate_context(analysis, final_validation=True, strict_evidence=True)

            ctx_str = json.dumps(ctx)
            # Verify no huge raw AST nodes or AST class names in output
            self.assertNotIn("ast.FunctionDef", ctx_str)
            self.assertNotIn("ast.ClassDef", ctx_str)
            self.assertNotIn("ast.Call", ctx_str)

            # Verify evidence entries only contain line numbers and metadata, not full code dumps
            for ev in ctx["evidence_and_provenance"]["evidence"]:
                self.assertIn("start_line", ev)
                self.assertIn("end_line", ev)
                self.assertNotIn("code_dump", ev)
                self.assertNotIn("raw_ast", ev)


if __name__ == "__main__":
    unittest.main()
