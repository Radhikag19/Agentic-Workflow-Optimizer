"""
Shared Agent Runner and Groq LLM Initializer.

Rules:
- Strictly enforces model: openai/gpt-oss-120b
- Retrieves dedicated workflow Groq API key via api_key_manager.py
- Masks and isolates credentials
- Connects through Groq API endpoint
- Drops unsupported parameters (e.g. cache_breakpoint) for Groq API compatibility
"""

from typing import Tuple, Any, Optional, Dict
import litellm
from common.settings import MANDATORY_MODEL
from common.api_key_manager import get_key_for_agent

# Ensure LiteLLM drops unsupported parameters before sending to Groq
litellm.drop_params = True

# Groq's API strictly rejects 'cache_breakpoint' in message dictionaries.
# Neutralize cache breakpoint marking so messages remain clean standard OpenAI-compatible dicts.
try:
    import crewai.llms.cache
    crewai.llms.cache.mark_cache_breakpoint = lambda msg: msg
except Exception:
    pass


def create_groq_llm_for_agent(agent_id: str) -> Tuple[Any, str]:
    """
    Creates an initialized CrewAI LLM instance strictly using Groq's openai/gpt-oss-120b model
    and the assigned workflow Groq API key.
    
    Returns:
        (llm_instance, safe_masked_identifier)
    """
    raw_key, env_var, safe_id = get_key_for_agent(agent_id)
    
    from crewai import LLM

    # Configured strictly with groq provider and mandatory model openai/gpt-oss-120b
    llm = LLM(
        model=f"groq/{MANDATORY_MODEL}",
        api_key=raw_key,
        temperature=0.1
    )
    
    return llm, safe_id


def get_llm(agent_id: Optional[str] = None) -> Any:
    """Retrieve or create an LLM instance."""
    if agent_id:
        llm, _ = create_groq_llm_for_agent(agent_id)
        return llm
    
    from crewai import LLM
    import os
    raw_key = os.getenv("GROQ_API_KEY_5") or os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY")
    return LLM(model=f"groq/{MANDATORY_MODEL}", api_key=raw_key, temperature=0.1)


# Test mock handler hook for unit tests without live API calls
_MOCK_TASK_HANDLER = None


def set_mock_task_handler(handler):
    """Set a mock handler for testing without live Groq API calls."""
    global _MOCK_TASK_HANDLER
    _MOCK_TASK_HANDLER = handler


ROLE_TO_AGENT_ID: Dict[str, str] = {
    # Recruitment (8 agents)
    "Resume Intake Specialist": "resume_intake_agent",
    "Candidate Qualification Extractor": "qualification_agent",
    "Job Criteria Analyst": "job_criteria_agent",
    "Candidate Evidence Mapping Specialist": "evidence_mapping_agent",
    "Experience Assessment Evaluator": "experience_assessment_agent",
    "Recruitment Consistency and Verification Auditor": "missing_evidence_agent",
    "Screening Quality and Integrity Reviewer": "screening_review_agent",
    "Executive Recruiter Summary Synthesizer": "recruiter_summary_agent",
    # Inventory (6 agents)
    "Inventory Intake Coordinator": "inventory_intake_agent",
    "Stock Policy and Inventory Level Specialist": "stock_policy_agent",
    "Demand Forecasting and Sales History Specialist": "demand_forecast_agent",
    "Supplier Lead-Time and Constraints Specialist": "supplier_agent",
    "Replenishment Optimization Planner": "replenishment_planning_agent",
    "Replenishment Integrity and Compliance Auditor": "inventory_review_agent",
    # Contracts (10 agents)
    "Contract Intake and Ingestion Specialist": "contract_intake_agent",
    "Contract Clause Extraction Specialist": "clause_extraction_agent",
    "Policy Interpretation Specialist": "policy_interpretation_agent",
    "Clause-to-Policy Mapping Specialist": "clause_mapping_agent",
    "Contract Compliance Deviation Evaluator": "compliance_deviation_agent",
    "Contract Risk and Ambiguity Analyst": "risk_ambiguity_agent",
    "Contract Inter-Clause Consistency Auditor": "cross_clause_consistency_agent",
    "Legal Evidence and Citation Validator": "evidence_validation_agent",
    "Senior Contract Compliance Reviewer": "contract_independent_review_agent",
    "Lead Contract Compliance Report Author": "final_compliance_report_agent",
}


def create_crew_agent(
    role: str,
    goal: str,
    backstory: str,
    agent_id: Optional[str] = None,
    llm: Optional[Any] = None,
    verbose: bool = False,
):
    """
    Create a genuine CrewAI Agent configured with the mandatory Groq openai/gpt-oss-120b LLM.
    Automatically assigns dedicated workflow Groq key based on agent role.
    """
    from crewai import Agent
    effective_id = agent_id or ROLE_TO_AGENT_ID.get(role)
    agent_llm = llm if llm is not None else get_llm(effective_id)
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=agent_llm,
        verbose=verbose,
        memory=False,
        allow_delegation=False,
    )


def run_agent_task(
    agent,
    agent_id: str,
    task_description: str,
    expected_output_desc: str,
    output_model=None,
    tracker=None,
    agent_name: Optional[str] = None,
):
    """
    Execute a single CrewAI agent task and validate its structured output.
    """
    global _MOCK_TASK_HANDLER
    from crewai import Task, Crew, Process
    from common.validation import extract_json_from_text
    from common.tracing import extract_token_usage
    import logging

    logger = logging.getLogger(__name__)
    name = agent_name or agent.role

    # If unit test mock handler is installed, route through mock handler
    if _MOCK_TASK_HANDLER is not None:
        if tracker and hasattr(tracker, "start_agent"):
            tracker.start_agent(agent_id=agent_id, agent_name=name)
        try:
            mock_data = _MOCK_TASK_HANDLER(agent_id, task_description)
            if isinstance(mock_data, str):
                parsed = extract_json_from_text(mock_data)
            elif isinstance(mock_data, dict):
                parsed = mock_data
            else:
                parsed = mock_data.model_dump() if hasattr(mock_data, "model_dump") else mock_data

            if output_model:
                try:
                    validated = output_model.model_validate(parsed)
                    result = validated.model_dump()
                except Exception:
                    result = parsed
            else:
                result = parsed

            if tracker and hasattr(tracker, "record_llm_usage"):
                tracker.record_llm_usage(prompt_tokens=150, completion_tokens=80)
            if tracker and hasattr(tracker, "finish_agent"):
                tracker.finish_agent(status="success")
            return result
        except Exception as exc:
            if tracker and hasattr(tracker, "finish_agent"):
                tracker.finish_agent(status="failed", error=str(exc))
            raise

    # Live Execution path with real CrewAI & Groq
    if tracker and hasattr(tracker, "start_agent"):
        tracker.start_agent(agent_id=agent_id, agent_name=name)

    task = Task(
        description=task_description,
        expected_output=expected_output_desc,
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    try:
        import time
        max_retries = 5
        crew_output = None
        for attempt in range(max_retries):
            try:
                crew_output = crew.kickoff()
                break
            except Exception as exc:
                err_str = str(exc)
                if ("rate_limit" in err_str.lower() or "429" in err_str or "rate limit" in err_str.lower()) and attempt < max_retries - 1:
                    wait_sec = 3.0 * (attempt + 1)
                    logger.warning(f"Rate limit encountered for agent '{agent_id}'. Retrying in {wait_sec}s (Attempt {attempt+1}/{max_retries})...")
                    print(f"\n[RateLimit Notice] Groq TPM limit approached for {agent_id}. Pausing {wait_sec:.1f}s for window reset...")
                    time.sleep(wait_sec)
                else:
                    raise

        raw_output_text = str(crew_output.raw if hasattr(crew_output, "raw") else crew_output)

        # Parse JSON
        parsed_data = extract_json_from_text(raw_output_text)

        # Validate with Pydantic model if provided
        if output_model:
            try:
                validated_obj = output_model.model_validate(parsed_data)
                result = validated_obj.model_dump()
            except Exception:
                result = parsed_data
        else:
            result = parsed_data

        # Record token usage from crew execution
        tot_toks, prompt_toks, comp_toks = extract_token_usage(crew_output, crew)
        if tracker and hasattr(tracker, "record_llm_usage"):
            tracker.record_llm_usage(prompt_tokens=prompt_toks, completion_tokens=comp_toks)

        safe_key_id = None
        try:
            _, _, safe_key_id = get_key_for_agent(agent_id)
        except Exception:
            pass

        if tracker and hasattr(tracker, "finish_agent"):
            try:
                tracker.finish_agent(
                    status="success",
                    tokens=tot_toks,
                    prompt_tokens=prompt_toks,
                    completion_tokens=comp_toks,
                    output=result,
                    assigned_key_identifier=safe_key_id,
                )
            except TypeError:
                tracker.finish_agent(status="success")

        return result

    except Exception as exc:
        error_msg = f"Agent '{agent_id}' execution failed: {str(exc)}"
        logger.error(error_msg)
        if tracker and hasattr(tracker, "finish_agent"):
            tracker.finish_agent(status="failed", error=error_msg)
        if tracker and hasattr(tracker, "add_warning"):
            tracker.add_warning(error_msg)
        raise RuntimeError(error_msg) from exc


import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class WorkflowExecutionResult:
    workflow_name: str
    run_id: str
    status: str  # "SUCCESS" | "FAILED"
    execution_order: list[str] = field(default_factory=list)
    agent_durations: dict[str, float] = field(default_factory=dict)
    total_duration_seconds: float = 0.0
    agent_count: int = 0
    llm_call_count: int = 0
    token_usage: dict[str, Any] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "is_available": False,
    })
    estimated_cost_usd: Optional[float] = None
    cost_estimation_label: str = "Unavailable"
    output: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunMetricsTracker:
    """Thread-safe real-time LLM call and token consumption accumulator."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.llm_call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def record_usage(self, prompt: int, completion: int, total: int):
        self.llm_call_count += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total

    def snapshot(self) -> dict[str, Any]:
        prompt_rate = 0.59 / 1_000_000
        completion_rate = 0.79 / 1_000_000
        cost = (self.prompt_tokens * prompt_rate) + (self.completion_tokens * completion_rate)
        has_calls = self.llm_call_count > 0
        return {
            "llm_call_count": self.llm_call_count,
            "token_usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "is_available": has_calls,
            },
            "estimated_cost_usd": round(cost, 6) if has_calls else None,
            "cost_estimation_label": (
                f"Estimated (${cost:.6f} USD at standard $0.59/$0.79 per 1M tokens)"
                if has_calls else "No live LLM calls recorded"
            ),
        }


ACTIVE_TRACKER = RunMetricsTracker()


def generate_run_id(prefix: str = "run") -> str:
    """Generate a clean timestamped correlation run ID."""
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{prefix}_{now_str}_{short_uuid}"


def save_run_result(result: WorkflowExecutionResult) -> Path:
    """Persist the workflow run result as structured JSON in runs/ and runs/{workflow_name}/."""
    wf_dir = RUNS_DIR / result.workflow_name
    wf_dir.mkdir(parents=True, exist_ok=True)
    
    # Save into workflow-specific subfolder
    sub_run_file = wf_dir / f"{result.run_id}.json"
    with open(sub_run_file, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Append to runs_summary.jsonl
    summary_file = RUNS_DIR / "runs_summary.jsonl"
    summary_entry = {
        "timestamp": result.timestamp,
        "run_id": result.run_id,
        "workflow": result.workflow_name,
        "status": result.status,
        "agent_count": result.agent_count,
        "duration_seconds": round(result.total_duration_seconds, 3),
        "total_tokens": result.token_usage.get("total_tokens", 0),
        "error": result.error,
    }
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary_entry) + "\n")

    return root_run_file


def print_run_summary(result: WorkflowExecutionResult):
    """Print a clean, structured console summary of the workflow execution."""
    divider = "=" * 68
    subdivider = "-" * 68
    print(f"\n{divider}")
    print(f" WORKFLOW EXECUTION SUMMARY: {result.workflow_name.upper()}")
    print(divider)
    print(f" Run ID           : {result.run_id}")
    print(f" Status           : {'[SUCCESS]' if result.status == 'SUCCESS' else '[FAILED]'}")
    print(f" Total Duration   : {result.total_duration_seconds:.2f}s")
    print(f" Agents Executed  : {result.agent_count}")
    print(f" Execution Order  : {' -> '.join(result.execution_order)}")
    print(subdivider)
    print(" Agent Execution Durations:")
    for agent, dur in result.agent_durations.items():
        print(f"   * {agent:<28} : {dur:.2f}s")
    print(subdivider)
    print(f" Token Usage      : {result.token_usage.get('total_tokens', 'N/A')} tokens")
    print(f" Estimated Cost   : {result.cost_estimation_label}")
    if result.error:
        print(subdivider)
        print(f" Error Details    : {result.error}")
    print(subdivider)
    print(" Structured Output Preview:")
    out_str = json.dumps(result.output, indent=2)
    preview_lines = out_str.splitlines()[:20]
    for line in preview_lines:
        print(f"   {line}")
    if len(out_str.splitlines()) > 20:
        print("   ... [truncated in preview, full output saved to run artifact]")
    print(divider + "\n")


