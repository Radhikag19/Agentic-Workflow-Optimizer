"""
Unit tests for the JSON-to-CSV Trace Conversion Pipeline.
"""

import csv
import json
import sys
from pathlib import Path

# Add project root and inner project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

import pytest
from tools.convert_traces_to_csv import TraceToCSVConverter, WORKFLOW_FIELDS, AGENT_FIELDS


def test_converter_customer_support():
    converter = TraceToCSVConverter()
    csv_path = converter.convert_workflow("customer_support")

    assert csv_path.exists(), f"Output file does not exist: {csv_path}"
    assert "customer_support" in csv_path.parts

    with open(csv_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    # 30 runs * 6 agents = 180 records
    assert len(rows) == 180, f"Expected 180 rows, found {len(rows)}"

    fieldnames = reader.fieldnames or []

    # Verify no unlisted or internal fields were extracted
    assert "output" not in fieldnames
    assert "assigned_key_identifier" not in fieldnames
    assert "final_output" not in fieldnames

    # Verify core workflow fields retained
    assert "run_id" in fieldnames
    assert "workflow_name" in fieldnames
    assert "model" in fieldnames

    # Check first row values
    r0 = rows[0]
    assert r0["workflow_name"] == "customer_support"
    assert r0["run_id"] == "live_20261031_132411_trace01"
    assert r0["agent_id"] == "ticket_classification_agent"
    assert r0["order"] == "1"
    assert r0["agent_status"] == "SUCCESS"
    assert r0["workflow_status"] == "SUCCESS"
    # Null field handling
    assert r0["error_message"] == ""


def test_converter_custom_dummy(tmp_path):
    dummy_trace = {
        "run_id": "test_run_999",
        "workflow_name": "dummy_wf",
        "model": "openai/gpt-oss-120b",
        "status": "SUCCESS",
        "duration_seconds": 12.34,
        "agent_execution_count": 2,
        "total_tokens": 1000,
        "prompt_tokens": 600,
        "completion_tokens": 400,
        "estimated_cost_usd": 0.001,
        "error_message": None,
        "timestamp": "2026-11-03T00:00:00Z",
        "agent_executions": [
            {
                "status": "SUCCESS",
                "assigned_key_identifier": "KEY_1",
                "duration_seconds": 5.0,
                "tokens": 500,
                "prompt_tokens": 300,
                "completion_tokens": 200,
                "output": {"secret": "hidden"},
                "notes": "dummy note 1",
                "order": 1,
                "agent_id": "agent_alpha",
            },
            {
                "status": "SUCCESS",
                "assigned_key_identifier": "KEY_2",
                "duration_seconds": 7.34,
                "tokens": 500,
                "prompt_tokens": 300,
                "completion_tokens": 200,
                "output": {"secret": "hidden_again"},
                "notes": "",
                "order": 2,
                "agent_id": "agent_beta",
            },
        ],
    }

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    with open(in_dir / "trace_01.json", "w", encoding="utf-8") as f:
        json.dump(dummy_trace, f)

    converter = TraceToCSVConverter(base_dir=tmp_path)
    csv_file = converter.convert_workflow(
        workflow_name="dummy_wf",
        custom_input_dir=in_dir,
        custom_output_dir=out_dir,
    )

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    assert len(records) == 2
    assert records[0]["agent_id"] == "agent_alpha"
    assert records[0]["notes"] == "dummy note 1"
    assert records[1]["agent_id"] == "agent_beta"
    assert records[1]["notes"] == ""
    assert "output" not in reader.fieldnames
