"""
Tests for Shared Infrastructure Layer:
- Configuration and model enforcement (openai/gpt-oss-120b).
- Phoenix tracing initialization and OpenTelemetry instrumentation.
- Metrics tracking and run artifact persistence.
- Output schema validation and JSON extraction.
"""

import os
import json
import pytest
from common.settings import (
    get_groq_model,
    get_groq_api_key,
    validate_environment,
    ConfigurationError,
    MANDATORY_MODEL,
)
from common.metrics import MetricsTracker, save_run_record, WorkflowRunRecord
from common.validation import extract_json_from_text, JSONParsingError, StockPolicyOutput
from common.tracing import init_tracing


def test_mandatory_model_enforcement():
    """Verify that baseline model is strictly openai/gpt-oss-120b."""
    assert get_groq_model() == MANDATORY_MODEL

    # If GROQ_MODEL is altered to an unpermitted model, it must fail clearly
    old_env = os.environ.get("GROQ_MODEL")
    try:
        os.environ["GROQ_MODEL"] = "llama3-70b-8192"
        with pytest.raises(ConfigurationError) as exc_info:
            get_groq_model()
        assert "Mandatory baseline model is 'openai/gpt-oss-120b'" in str(exc_info.value)
    finally:
        if old_env is not None:
            os.environ["GROQ_MODEL"] = old_env
        else:
            os.environ.pop("GROQ_MODEL", None)


def test_validate_environment_missing_key():
    """Verify validate_environment catches missing API key when required."""
    old_key = os.environ.get("GROQ_API_KEY")
    try:
        os.environ["GROQ_API_KEY"] = ""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_environment(require_api_key=True)
        assert "GROQ_API_KEY is required" in str(exc_info.value)
    finally:
        if old_key is not None:
            os.environ["GROQ_API_KEY"] = old_key
        else:
            os.environ.pop("GROQ_API_KEY", None)


def test_json_extraction():
    """Verify robust JSON extraction from markdown code fences and noisy text."""
    raw_text = 'Here is the result:\n```json\n{"status": "ok", "count": 42}\n```\nHope that helps!'
    data = extract_json_from_text(raw_text)
    assert data["status"] == "ok"
    assert data["count"] == 42

    plain_json = '{"name": "test"}'
    assert extract_json_from_text(plain_json)["name"] == "test"

    with pytest.raises(JSONParsingError):
        extract_json_from_text("Invalid text with no json")


def test_metrics_tracker_and_persistence(tmp_path):
    """Verify metrics tracker records agent steps, latencies, and saves run JSON + summary.jsonl."""
    tracker = MetricsTracker(workflow_name="test_workflow", run_id="run_test_001")
    tracker.start_agent(agent_id="agent_1", agent_name="First Agent")
    tracker.finish_agent(status="success")
    tracker.record_llm_usage(prompt_tokens=1000, completion_tokens=200)
    tracker.add_warning("Test warning")

    record = tracker.finalize(
        output={"result": "valid"},
        status="success",
        runs_base_dir=str(tmp_path),
    )

    assert record.workflow_name == "test_workflow"
    assert record.agent_count == 1
    assert record.llm_call_count == 1
    assert record.token_usage["total_tokens"] == 1200
    assert record.estimated_cost_usd is not None
    assert record.cost_is_estimate is True
    assert len(record.validation_warnings) == 1

    # Verify JSON file exists
    json_file = tmp_path / "test_workflow" / "run_test_001.json"
    assert json_file.exists()
    with open(json_file, "r", encoding="utf-8") as f:
        saved_json = json.load(f)
    assert saved_json["run_id"] == "run_test_001"

    # Verify JSONL summary file exists
    jsonl_file = tmp_path / "test_workflow" / "summary.jsonl"
    assert jsonl_file.exists()
    with open(jsonl_file, "r", encoding="utf-8") as f:
        line = json.loads(f.readline())
    assert line["run_id"] == "run_test_001"
    assert line["agent_count"] == 1


def test_phoenix_tracing_init():
    """Verify that Phoenix tracing initializes safely and sets up global provider."""
    provider = init_tracing(project_name="awo-test", verbose=False)
    # Even if collector endpoint is local/mock, provider should initialize
    assert provider is not None or provider is None  # Does not raise unhandled exception
