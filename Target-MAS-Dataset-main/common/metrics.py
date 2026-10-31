"""
Shared Metrics and Run Logging for AWO Multi-Agent Workflows.

Collects real execution metrics:
- Workflow name and unique run ID
- Start time, end time, total latency
- Per-agent execution durations and order
- LLM call counts and token usage (when reported)
- Estimated API cost (labeled as estimate, never fabricated)
- Execution status (success/failure)
- Final structured output and validation warnings
- Saves JSON file per run and appends to JSONL run history
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Approximate reference pricing for gpt-oss-120b on Groq per 1M tokens ($0.59 input / $0.79 output approx, or None if unknown)
GROQ_GPT_OSS_120B_INPUT_PER_1M = 0.59
GROQ_GPT_OSS_120B_OUTPUT_PER_1M = 0.79


class AgentMetric(BaseModel):
    agent_id: str
    agent_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    status: str = "success"
    error: Optional[str] = None
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    assigned_key_identifier: Optional[str] = None
    output: Optional[Any] = None


class WorkflowRunRecord(BaseModel):
    workflow_name: str
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: str
    end_time: str
    total_duration_seconds: float
    agent_count: int
    agent_execution_order: List[str]
    agent_metrics: List[AgentMetric] = Field(default_factory=list)
    llm_call_count: int = 0
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    estimated_cost_usd: Optional[float] = None
    cost_is_estimate: bool = True
    status: str = "success"  # "success" or "failed"
    validation_warnings: List[str] = Field(default_factory=list)
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MetricsTracker:
    """Tracks execution times, agent steps, and writes run artifacts."""

    def __init__(self, workflow_name: str, run_id: Optional[str] = None):
        self.workflow_name = workflow_name
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.start_timestamp = time.time()
        self.start_iso = datetime.now(timezone.utc).isoformat()
        self.agent_metrics: List[AgentMetric] = []
        self.agent_order: List[str] = []
        self.current_agent: Optional[str] = None
        self.current_agent_start: Optional[float] = None
        self.current_agent_start_iso: Optional[str] = None
        self.llm_call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.validation_warnings: List[str] = []

    def start_agent(self, agent_id: str, agent_name: str) -> None:
        """Mark the start of an agent stage."""
        self.current_agent = agent_id
        self.current_agent_name = agent_name
        self.current_agent_start = time.time()
        self.current_agent_start_iso = datetime.now(timezone.utc).isoformat()
        self.agent_order.append(agent_id)

    def finish_agent(
        self,
        status: str = "success",
        error: Optional[str] = None,
        tokens: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        output: Optional[Any] = None,
        assigned_key_identifier: Optional[str] = None,
    ) -> None:
        """Mark the completion of current agent stage."""
        if self.current_agent and self.current_agent_start is not None:
            end_t = time.time()
            end_iso = datetime.now(timezone.utc).isoformat()
            duration = round(end_t - self.current_agent_start, 3)
            self.agent_metrics.append(
                AgentMetric(
                    agent_id=self.current_agent,
                    agent_name=getattr(self, "current_agent_name", self.current_agent),
                    start_time=self.current_agent_start_iso or self.start_iso,
                    end_time=end_iso,
                    duration_seconds=duration,
                    status=status,
                    error=error,
                    tokens=tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    output=output,
                    assigned_key_identifier=assigned_key_identifier,
                )
            )
            self.current_agent = None
            self.current_agent_start = None

    def record_llm_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Record real token usage reported by LLM calls."""
        self.llm_call_count += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def add_warning(self, warning: str) -> None:
        """Add a validation warning to the run."""
        self.validation_warnings.append(warning)

    def finalize(
        self,
        output: Optional[Dict[str, Any]] = None,
        status: str = "success",
        error: Optional[str] = None,
        runs_base_dir: str = "runs",
    ) -> WorkflowRunRecord:
        """Finalize metrics and persist to disk in runs/<workflow_name>/."""
        end_timestamp = time.time()
        end_iso = datetime.now(timezone.utc).isoformat()
        total_duration = round(end_timestamp - self.start_timestamp, 3)

        # Token usage dict
        token_usage: Dict[str, Any] = {}
        estimated_cost: Optional[float] = None
        if self.prompt_tokens > 0 or self.completion_tokens > 0:
            total_tokens = self.prompt_tokens + self.completion_tokens
            token_usage = {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": total_tokens,
            }
            cost_calc = (
                (self.prompt_tokens / 1_000_000.0) * GROQ_GPT_OSS_120B_INPUT_PER_1M
                + (self.completion_tokens / 1_000_000.0) * GROQ_GPT_OSS_120B_OUTPUT_PER_1M
            )
            estimated_cost = round(cost_calc, 6)

        record = WorkflowRunRecord(
            workflow_name=self.workflow_name,
            run_id=self.run_id,
            start_time=self.start_iso,
            end_time=end_iso,
            total_duration_seconds=total_duration,
            agent_count=len(self.agent_order),
            agent_execution_order=self.agent_order,
            agent_metrics=self.agent_metrics,
            llm_call_count=self.llm_call_count,
            token_usage=token_usage,
            estimated_cost_usd=estimated_cost,
            cost_is_estimate=True,
            status=status,
            validation_warnings=self.validation_warnings,
            output=output,
            error=error,
        )

        # Persist to runs/<workflow_name>/
        save_run_record(record, runs_base_dir=runs_base_dir)
        return record


def save_run_record(record: WorkflowRunRecord, runs_base_dir: str = "runs") -> str:
    """Save run JSON and append to summary.jsonl in workflow dir."""
    target_dir = os.path.join(runs_base_dir, record.workflow_name)
    os.makedirs(target_dir, exist_ok=True)

    json_path = os.path.join(target_dir, f"{record.run_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record.model_dump(), indent=2, ensure_ascii=False))

    jsonl_path = os.path.join(target_dir, "summary.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        summary_line = {
            "run_id": record.run_id,
            "workflow_name": record.workflow_name,
            "start_time": record.start_time,
            "end_time": record.end_time,
            "total_duration_seconds": record.total_duration_seconds,
            "agent_count": record.agent_count,
            "status": record.status,
            "llm_call_count": record.llm_call_count,
            "token_usage": record.token_usage,
            "estimated_cost_usd": record.estimated_cost_usd,
            "validation_warnings_count": len(record.validation_warnings),
            "has_error": bool(record.error),
        }
        f.write(json.dumps(summary_line, ensure_ascii=False) + "\n")

    return json_path
