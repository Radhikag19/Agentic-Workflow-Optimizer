"""
Global settings, environment configuration, and Groq API key distribution.

Strict constraints enforced:
1. Model: Strictly openai/gpt-oss-120b across all 10 workflows and 78 agents.
2. Groq Keys: Keys 1 to 14 supported with dedicated per-workflow or per-agent routing.
3. Observability: Arize Phoenix tracing endpoint and project definitions.
4. Groq Sanitizer: Strips unsupported properties (like cache_breakpoint) for clean execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

try:
    from dotenv import dotenv_values, load_dotenv
except ImportError:
    def dotenv_values(path):
        env = {}
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env[k.strip()] = v.strip().strip("'\"")
        return env

    def load_dotenv(dotenv_path=None):
        vals = dotenv_values(dotenv_path or ".env")
        for k, v in vals.items():
            if k not in os.environ:
                os.environ[k] = v

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Load .env file
ENV_PATH = PROJECT_ROOT / ".env"
raw_env = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
load_dotenv(dotenv_path=ENV_PATH)

# Supported key pool: Keys 1 through 14
ALLOWED_KEY_NAMES = [f"GROQ_API_KEY_{i}" for i in range(1, 15)]

GROQ_KEYS_POOL: Dict[str, str] = {}
for key_name in ALLOWED_KEY_NAMES:
    val = raw_env.get(key_name) or os.getenv(key_name) or ""
    val = val.strip()
    if val:
        GROQ_KEYS_POOL[key_name] = val

# Mandatory model constraint
MANDATORY_MODEL = "openai/gpt-oss-120b"
configured_model = (os.getenv("GROQ_MODEL") or MANDATORY_MODEL).strip()
if configured_model != MANDATORY_MODEL:
    raise ValueError(
        f"Invalid model configuration: '{configured_model}'. "
        f"All workflows must strictly use '{MANDATORY_MODEL}'."
    )
GROQ_MODEL = MANDATORY_MODEL

# Arize Phoenix Tracing Configuration
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "agentic-workflow-optimizer").strip()
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces").strip()
PHOENIX_API_KEY = (os.getenv("PHOENIX_API_KEY") or "").strip()


class ConfigurationError(ValueError):
    """Raised when configuration validation fails."""
    pass


def get_groq_api_key() -> str:
    """Retrieve default/fallback Groq API key."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    return GROQ_KEYS_POOL.get("GROQ_API_KEY_1", "")


def get_groq_model() -> str:
    """Return mandatory Groq model, verifying against MANDATORY_MODEL."""
    model = (os.getenv("GROQ_MODEL") or MANDATORY_MODEL).strip()
    if model != MANDATORY_MODEL:
        raise ConfigurationError(
            f"Invalid model configuration: '{model}'. "
            f"Mandatory baseline model is '{MANDATORY_MODEL}'."
        )
    return model


def get_phoenix_endpoint() -> str:
    """Return Phoenix collector endpoint."""
    return PHOENIX_COLLECTOR_ENDPOINT


def get_phoenix_api_key() -> Optional[str]:
    """Retrieve Phoenix API key if configured."""
    key = PHOENIX_API_KEY
    return key if key else None


def get_phoenix_project_name() -> str:
    """Return Phoenix project name."""
    return PHOENIX_PROJECT_NAME


def validate_environment(require_api_key: bool = False) -> None:
    """Validate runtime environment and required API keys."""
    model = get_groq_model()
    if model != MANDATORY_MODEL:
        raise ConfigurationError(f"Mandatory model must be {MANDATORY_MODEL}, got {model}")
    if require_api_key:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is required for live LLM execution. "
                "Please configure GROQ_API_KEY in your environment or .env file."
            )


# Explicit per-agent key assignments for Workflows 8, 9, 10 (Keys 8..14)
AGENT_KEY_ASSIGNMENTS: Dict[str, str] = {
    # Workflow 8: IT Incident Triage (7 agents -> 1-to-1 mapping with keys 8..14)
    "intake_agent": "GROQ_API_KEY_8",
    "classification_agent": "GROQ_API_KEY_9",
    "runbook_agent": "GROQ_API_KEY_10",
    "symptom_agent": "GROQ_API_KEY_11",
    "severity_agent": "GROQ_API_KEY_12",
    "remediation_agent": "GROQ_API_KEY_13",
    "incident_review_agent": "GROQ_API_KEY_14",

    # Workflow 9: Expense Reimbursement (8 agents distributed across keys 8..14)
    "receipt_agent": "GROQ_API_KEY_8",
    "normalize_agent": "GROQ_API_KEY_9",
    "category_agent": "GROQ_API_KEY_10",
    "amount_agent": "GROQ_API_KEY_11",
    "policy_agent": "GROQ_API_KEY_12",
    "duplicate_agent": "GROQ_API_KEY_13",
    "exception_agent": "GROQ_API_KEY_14",
    "expense_review_agent": "GROQ_API_KEY_8",

    # Workflow 10: Market Research (9 agents distributed across keys 8..14)
    "planning_agent": "GROQ_API_KEY_8",
    "profile_agent": "GROQ_API_KEY_9",
    "product_agent": "GROQ_API_KEY_10",
    "pricing_agent": "GROQ_API_KEY_11",
    "feedback_agent": "GROQ_API_KEY_12",
    "evidence_agent": "GROQ_API_KEY_13",
    "comparison_agent": "GROQ_API_KEY_14",
    "synthesis_agent": "GROQ_API_KEY_8",
    "fact_check_agent": "GROQ_API_KEY_9",
}


def get_groq_key_for_agent(agent_name: str) -> str:
    """Return the designated Groq API key for a given agent name."""
    # Check explicit per-agent assignment first
    key_name = AGENT_KEY_ASSIGNMENTS.get(agent_name)
    if key_name and key_name in GROQ_KEYS_POOL:
        return GROQ_KEYS_POOL[key_name]
    
    # Try api_key_manager for workflows 1-7
    try:
        from common.api_key_manager import get_key_for_agent as mgr_get_key
        raw_key, _, _ = mgr_get_key(agent_name)
        if raw_key:
            return raw_key
    except Exception:
        pass

    # Fallback to designated key or default key
    if key_name and key_name in os.environ:
        return os.environ[key_name].strip()
    return get_groq_api_key()


def create_agent_llm(agent_name: str, temperature: float = 0.1):
    """Factory creating a CrewAI LLM instance strictly bound to openai/gpt-oss-120b
    and the agent's dedicated Groq API key."""
    import litellm
    from crewai import LLM

    # Enable drop_params in LiteLLM
    litellm.drop_params = True

    # Ensure unsupported parameters like 'cache_breakpoint' from CrewAI are stripped for Groq
    if not getattr(litellm, "_groq_sanitizer_applied", False):
        orig_completion = litellm.completion

        def _sanitized_completion(*args, **kwargs):
            if "messages" in kwargs and isinstance(kwargs["messages"], list):
                for msg in kwargs["messages"]:
                    if isinstance(msg, dict):
                        msg.pop("cache_breakpoint", None)
            
            import time
            max_retries = 6
            resp = None
            for attempt in range(max_retries):
                try:
                    resp = orig_completion(*args, **kwargs)
                    break
                except Exception as exc:
                    err_str = str(exc)
                    if ("rate_limit" in err_str.lower() or "429" in err_str or "rate limit" in err_str.lower()) and attempt < max_retries - 1:
                        wait_sec = 4.0 * (attempt + 1)
                        print(f"\n[RateLimit Notice] Groq rate limit hit. Pausing {wait_sec:.1f}s before retry ({attempt+1}/{max_retries})...")
                        time.sleep(wait_sec)
                    else:
                        raise

            try:
                if hasattr(resp, "usage") and resp.usage:
                    from common.agent_runner import ACTIVE_TRACKER
                    prompt = getattr(resp.usage, "prompt_tokens", 0) or 0
                    completion = getattr(resp.usage, "completion_tokens", 0) or 0
                    total = getattr(resp.usage, "total_tokens", 0) or (prompt + completion)
                    ACTIVE_TRACKER.record_usage(prompt, completion, total)
            except Exception:
                pass
            return resp

        litellm.completion = _sanitized_completion
        litellm._groq_sanitizer_applied = True

    api_key = get_groq_key_for_agent(agent_name)
    if not api_key:
        api_key = "dummy_mock_key_for_offline_testing"

    model_str = f"groq/{GROQ_MODEL}"
    return LLM(
        model=model_str,
        api_key=api_key,
        temperature=temperature,
    )


def get_llm(api_key: Optional[str] = None):
    """Create and return a CrewAI LLM instance configured with mandatory model."""
    from common.agent_runner import get_llm as runner_get_llm
    return runner_get_llm(api_key=api_key or get_groq_api_key())
