"""
Unit tests for WorkflowContext preparation, schema validation, and token measurement.
"""

import json
import unittest

from schemas.workflow_context_schema import (
    WorkflowContext,
    create_empty_workflow_context,
    validate_workflow_context,
    TOKEN_ESTIMATION_DEFAULT_METHOD,
)
from ingestion.context_generator import ContextGenerator


class TestContextPreparation(unittest.TestCase):
    def test_workflow_context_lookups_and_structure(self):
        """Verify deterministic dictionary structure and fields without vector DBs or embeddings."""
        ctx_dict = create_empty_workflow_context(
            workflow_id="test_wf_01",
            workflow_name="Test Workflow",
            source_directory="Agents/test",
            workflow_entrypoint="workflow.py",
            analysis_status="completed",
        )

        ctx_dict["agent_inventory"]["agents"].append({
            "agent_id": "test_agent_1",
            "source_file": "agent_1.py",
            "definition_status": "static",
            "invocation_status": "invoked",
            "evidence_refs": ["ev_1"],
        })
        ctx_dict["agent_inventory"]["discovered_agent_files"] = 1
        ctx_dict["agent_inventory"]["identified_agent_definitions"] = 1
        ctx_dict["agent_inventory"]["workflow_invoked_agents"] = 1

        ctx_dict["agent_details"].append({
            "agent_id": "test_agent_1",
            "definition": {
                "type": "function",
                "name": "create_agent",
                "source_file": "agent_1.py",
                "start_line": 10,
                "end_line": 20,
            },
            "role": "Extractor",
            "goal": "Extract invoice fields",
            "backstory": "Auditor",
            "llm": {"provider": "Groq", "model": "llama-3.3-70b-versatile", "resolution_status": "static"},
            "tasks": [],
            "registered_tools": [],
            "helper_functions": [],
            "explicit_code_invocations": [],
            "evidence_refs": ["ev_1"],
        })

        ctx_dict["evidence_and_provenance"]["evidence"].append({
            "evidence_id": "ev_1",
            "source_file": "agent_1.py",
            "start_line": 10,
            "end_line": 20,
            "entity_id": "agent:test_agent_1",
            "evidence_type": "agent_definition",
        })

        # Summary sync
        ctx_dict["analysis_summary"]["total_agents"] = 1
        ctx_dict["analysis_summary"]["invoked_agents"] = 1
        ctx_dict["analysis_summary"]["uninvoked_agents"] = 0
        ctx_dict["analysis_summary"]["total_tasks"] = 0
        ctx_dict["analysis_summary"]["total_dependencies"] = 0
        ctx_dict["analysis_summary"]["unresolved_relationships"] = 0

        # Validate
        validate_workflow_context(ctx_dict, final_validation=True, strict_evidence=True)

        wf_obj = WorkflowContext.from_dict(ctx_dict)
        self.assertEqual(wf_obj.workflow_metadata["workflow_id"], "test_wf_01")
        self.assertEqual(len(wf_obj.agent_details), 1)
        self.assertEqual(wf_obj.agent_details[0]["role"], "Extractor")

    def test_token_measurement_and_serialization(self):
        """Verify token estimation method and standard json serialization."""
        gen = ContextGenerator()
        ctx_dict = create_empty_workflow_context("wf_tok", "WF Tok", "dir")
        json_str = gen.serialize_to_json(ctx_dict)

        self.assertGreater(len(json_str), 0)
        deserialized = json.loads(json_str)
        self.assertEqual(
            deserialized["analysis_summary"]["output_metrics"]["token_estimation_method"],
            TOKEN_ESTIMATION_DEFAULT_METHOD,
        )


if __name__ == "__main__":
    unittest.main()
