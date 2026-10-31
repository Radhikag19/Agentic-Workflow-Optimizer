"""
Arize Phoenix Tracing and Execution Observability Integration.

Supports:
- Local Arize Phoenix UI and OpenTelemetry spans
- CrewAI / OpenInference auto-instrumentation
- Run ID correlation
- Workflow & Agent execution spans with real token usage extraction
- Strict API key isolation (keys masked, never recorded in spans or logs)
- Local run summary capture in runs/{run_id}.json
"""

import os
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

from common.settings import (
    PHOENIX_PROJECT_NAME,
    PHOENIX_COLLECTOR_ENDPOINT,
    RUNS_DIR,
    MANDATORY_MODEL
)

_phoenix_initialized = False

def init_phoenix(project_name: Optional[str] = None, verbose: bool = True, **kwargs):
    """
    Initialize Arize Phoenix tracing and auto-instrumentation.
    Supports local interactive session and OTLP exporter.
    """
    global _phoenix_initialized
    if _phoenix_initialized:
        return

    proj = project_name or PHOENIX_PROJECT_NAME

    try:
        import phoenix as px
        # If running locally without remote collector, px.launch_app() provides local UI
        if "127.0.0.1" in PHOENIX_COLLECTOR_ENDPOINT or "localhost" in PHOENIX_COLLECTOR_ENDPOINT:
            try:
                px.launch_app(project_name=proj)
                print(f"[Tracing] Arize Phoenix UI active at http://localhost:6006 (Project: {proj})")
            except Exception:
                pass

        # OpenInference CrewAI instrumentation
        try:
            from openinference.instrumentation.crewai import CrewAIInstrumentor
            CrewAIInstrumentor().instrument(skip_dep_check=True)
            print("[Tracing] OpenInference CrewAI instrumentor registered.")
        except Exception:
            pass

        _phoenix_initialized = True
    except Exception as exc:
        print(f"[Tracing] Phoenix integration notice: {exc}")
        _phoenix_initialized = False

# Backward compatibility alias
init_tracing = init_phoenix


def extract_token_usage(crew_output: Any, crew: Any = None) -> Tuple[int, int, int]:
    """
    Extracts (total_tokens, prompt_tokens, completion_tokens) from CrewOutput, Crew, or dict.
    """
    usage = getattr(crew_output, "token_usage", None)
    if not usage and isinstance(crew_output, dict):
        usage = crew_output.get("token_usage")
    if not usage and crew and hasattr(crew, "usage_metrics"):
        usage = crew.usage_metrics

    if usage:
        if isinstance(usage, dict):
            total = int(usage.get("total_tokens", 0) or 0)
            prompt = int(usage.get("prompt_tokens", 0) or 0)
            completion = int(usage.get("completion_tokens", 0) or 0)
        else:
            total = int(getattr(usage, "total_tokens", 0) or 0)
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
        return total, prompt, completion
    return 0, 0, 0


class WorkflowRunTracker:
    """Tracks metrics, real token usage, and outputs for an individual workflow execution run."""

    def __init__(self, workflow_name: str, run_id: Optional[str] = None):
        self.workflow_name = workflow_name
        self.run_id = run_id or f"{workflow_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.start_time = 0.0
        self.end_time = 0.0
        self.duration_seconds = 0.0
        self.status = "PENDING"
        self.agent_executions = []
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.estimated_cost_usd = 0.0
        self.final_output = None
        self.error_message = None

    def start(self):
        self.start_time = time.time()
        self.status = "RUNNING"
        return self

    def log_agent_execution(
        self,
        agent_id: str,
        assigned_key_id: str,
        duration: float,
        status: str = "SUCCESS",
        tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        output: Any = None,
        notes: str = ""
    ):
        """Record an agent's execution step with its safe key identifier, token usage, and output."""
        parsed_output = output
        if isinstance(output, str):
            trimmed = output.strip()
            if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
                try:
                    parsed_output = json.loads(trimmed)
                except Exception:
                    parsed_output = output

        self.agent_executions.append({
            "order": len(self.agent_executions) + 1,
            "agent_id": agent_id,
            "assigned_key_identifier": assigned_key_id,  # Safe env var & masked key only
            "duration_seconds": round(duration, 3),
            "status": status,
            "tokens": tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "output": parsed_output,
            "notes": notes
        })
        self.total_tokens += tokens
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def complete(self, final_output: Any):
        self.end_time = time.time()
        self.duration_seconds = round(self.end_time - self.start_time, 3)
        self.status = "SUCCESS"
        self.final_output = final_output
        # Estimated cost for gpt-oss-120b (~$0.15/1M prompt, $0.60/1M completion)
        self.estimated_cost_usd = round(
            (self.prompt_tokens * 0.00000015) + (self.completion_tokens * 0.00000060), 6
        )
        self.save_run_summary()

    def fail(self, error: Exception):
        self.end_time = time.time()
        self.duration_seconds = round(self.end_time - self.start_time, 3)
        self.status = "FAILED"
        self.error_message = str(error)
        self.save_run_summary()

    def save_run_summary(self):
        """Persist structured run summary to runs/{workflow_name}/{run_id}.json."""
        summary = {
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "model": MANDATORY_MODEL,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "agent_execution_count": len(self.agent_executions),
            "agent_executions": self.agent_executions,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "error_message": self.error_message,
            "final_output": self.final_output,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        wf_dir = RUNS_DIR / self.workflow_name
        wf_dir.mkdir(parents=True, exist_ok=True)
        wf_file = wf_dir / f"{self.run_id}.json"
        with open(wf_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return wf_file


# OpenTelemetry Spans & Phoenix Tracing Functions for Workflows 8-10

from contextlib import contextmanager
from typing import Generator

_TRACER = None
_LITELLM_INSTRUMENTED = False
_PHOENIX_SESSION = None


def setup_tracing(project_name: Optional[str] = None, endpoint: Optional[str] = None):
    """Initialize Arize Phoenix OpenTelemetry tracing and LiteLLM instrumentation."""
    global _TRACER, _LITELLM_INSTRUMENTED

    proj = project_name or PHOENIX_PROJECT_NAME
    ep = endpoint or PHOENIX_COLLECTOR_ENDPOINT

    if not _LITELLM_INSTRUMENTED:
        try:
            from openinference.instrumentation.litellm import LiteLLMInstrumentor
            LiteLLMInstrumentor().instrument()
            _LITELLM_INSTRUMENTED = True
        except Exception:
            pass

    try:
        from phoenix.otel import register
        from opentelemetry import trace
        register(
            project_name=proj,
            endpoint=ep,
            set_global_tracer_provider=True,
            use_batch_processor=True,
        )
        _TRACER = trace.get_tracer(proj)
    except Exception:
        try:
            from opentelemetry import trace
            _TRACER = trace.get_tracer("agentic-workflow-optimizer")
        except Exception:
            _TRACER = None

    return _TRACER


def get_tracer():
    """Return active OpenTelemetry tracer or initialize."""
    global _TRACER
    if _TRACER is None:
        _TRACER = setup_tracing()
    return _TRACER


def launch_phoenix_server(port: int = 6006):
    """Launch local Phoenix UI server for real-time visualization."""
    global _PHOENIX_SESSION
    try:
        import phoenix as px
        if _PHOENIX_SESSION is None:
            _PHOENIX_SESSION = px.launch_app(port=port)
            print(f"[Phoenix] Tracing UI started at: {_PHOENIX_SESSION.url}")
            return _PHOENIX_SESSION
    except Exception as e:
        print(f"[Phoenix] Note: Could not launch local UI ({e}). Continuing with collector endpoint.")
    return _PHOENIX_SESSION


class _NoOpSpan:
    """Fallback no-op span context manager when OpenTelemetry is inactive."""
    def set_attribute(self, *args, **kwargs): pass
    def set_status(self, *args, **kwargs): pass
    def record_exception(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass


@contextmanager
def workflow_span(
    workflow_name: str,
    run_id: str,
    inputs: Optional[Dict[str, Any]] = None
) -> Generator[Any, None, None]:
    """Context manager for tracing top-level workflow executions."""
    tracer = get_tracer()
    start_t = time.perf_counter()
    if tracer is not None:
        try:
            from opentelemetry.trace import Status, StatusCode
            with tracer.start_as_current_span(f"workflow.{workflow_name}") as span:
                span.set_attribute("workflow.name", workflow_name)
                span.set_attribute("workflow.run_id", run_id)
                span.set_attribute("openinference.span.kind", "CHAIN")
                if inputs:
                    for k, v in inputs.items():
                        span.set_attribute(f"workflow.input.{k}", str(v)[:500])
                try:
                    yield span
                    span.set_status(Status(StatusCode.OK))
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                finally:
                    elapsed = time.perf_counter() - start_t
                    span.set_attribute("workflow.duration_seconds", elapsed)
            return
        except Exception:
            pass

    # Fallback if tracer unavailable
    yield _NoOpSpan()


@contextmanager
def agent_span(
    agent_name: str,
    run_id: str,
    agent_inputs: Optional[Dict[str, Any]] = None
) -> Generator[Any, None, None]:
    """Context manager for tracing individual agent step executions under workflow."""
    tracer = get_tracer()
    start_t = time.perf_counter()
    if tracer is not None:
        try:
            from opentelemetry.trace import Status, StatusCode
            with tracer.start_as_current_span(f"agent.{agent_name}") as span:
                span.set_attribute("agent.name", agent_name)
                span.set_attribute("agent.run_id", run_id)
                span.set_attribute("openinference.span.kind", "AGENT")
                if agent_inputs:
                    for k, v in agent_inputs.items():
                        span.set_attribute(f"agent.input.{k}", str(v)[:500])
                try:
                    yield span
                    span.set_status(Status(StatusCode.OK))
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                finally:
                    elapsed = time.perf_counter() - start_t
                    span.set_attribute("agent.duration_seconds", elapsed)
            return
        except Exception:
            pass

    # Fallback if tracer unavailable
    yield _NoOpSpan()

