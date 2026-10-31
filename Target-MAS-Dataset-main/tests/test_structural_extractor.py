"""
Unit tests for ASTParser and structural extraction elements.
"""

import tempfile
import unittest
from pathlib import Path

from ingestion.ast_parser import ASTParser, safe_eval_literal


class TestStructuralExtractor(unittest.TestCase):
    def setUp(self):
        self.parser = ASTParser()

    def test_safe_read_and_sha256(self):
        """Verify safe text reading and SHA256 computation."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write("x = 42\nprint(x)\n")
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            self.assertEqual(parsed.parse_status, "parsed")
            self.assertIsNotNone(parsed.sha256)
            self.assertEqual(len(parsed.sha256), 64)
            self.assertIn("x", parsed.local_variables)
        finally:
            f_path.unlink(missing_ok=True)

    def test_syntax_error_resilience(self):
        """Verify syntax errors are recorded without halting the pipeline."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write("def malformed_syntax(\n  # incomplete\n")
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            self.assertEqual(parsed.parse_status, "failed")
            self.assertIn("SyntaxError", parsed.error_message or "")
        finally:
            f_path.unlink(missing_ok=True)

    def test_structural_extraction_elements(self):
        """Verify imports, classes, functions, and top constants are properly extracted."""
        code = """
import os
from common.settings import MANDATORY_MODEL
from external_unknown import mystery_fn

AGENT_ID = "test_sample_agent"

def deterministic_math(a, b=10):
    '''Calculate total sum.'''
    return a + b

class SampleProcessor:
    def process(self, data):
        return mystery_fn(data)
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".py") as f:
            f.write(code)
            f_path = Path(f.name)

        try:
            parsed = self.parser.parse_file(f_path)
            self.assertEqual(parsed.parse_status, "parsed")

            # Check constants
            agent_id_assign = next(a for a in parsed.top_level_assignments if "AGENT_ID" in a.targets)
            stat, val, _ = safe_eval_literal(agent_id_assign.value_node)
            self.assertTrue(stat)
            self.assertEqual(val, "test_sample_agent")

            # Check imports
            import_modules = [imp.module for imp in parsed.imports if imp.is_from]
            self.assertIn("external_unknown", import_modules)
            self.assertIn("common.settings", import_modules)

            # Check functions
            self.assertIn("deterministic_math", parsed.functions)
            math_fn = parsed.functions["deterministic_math"]
            self.assertEqual(math_fn.args, ["a", "b"])
            self.assertEqual(math_fn.docstring, "Calculate total sum.")

            # Check classes
            self.assertIn("SampleProcessor", parsed.classes)
            cls_def = parsed.classes["SampleProcessor"]
            self.assertIn("process", cls_def.methods)
        finally:
            f_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
