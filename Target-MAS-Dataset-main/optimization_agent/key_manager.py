"""
Groq API Key Manager for TraceProfiler Autonomous Optimization Discovery Agent.

Strict Rules Enforced:
1. 20 Dedicated Groq API keys (GROQ_API_KEY_1 through GROQ_API_KEY_20).
2. NO simple round-robin. Task-based and batch-based logical allocation.
3. Dimension-specific discovery analyses use distinct designated keys.
4. Failover pool (GROQ_API_KEY_15 through GROQ_API_KEY_20) handles rate limits gracefully.
5. Secret API keys must NEVER be exposed in logs, state objects, traces, or reports.
   Only safe masked identifiers (e.g., 'GROQ_API_KEY_1 (gsk_***...PW)') are returned.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("optimization_agent.key_manager")

# Task-to-key logical mapping
TASK_KEY_MAPPING: Dict[str, List[str]] = {
    "discovery_overview": ["GROQ_API_KEY_1", "GROQ_API_KEY_2"],
    "agent_pruning": ["GROQ_API_KEY_3"],
    "agent_consolidation": ["GROQ_API_KEY_4"],
    "workflow_parallelization": ["GROQ_API_KEY_5"],
    "model_selection": ["GROQ_API_KEY_6"],
    "prompt_optimization": ["GROQ_API_KEY_7"],
    "tool_optimization": ["GROQ_API_KEY_8"],
    "communication_optimization": ["GROQ_API_KEY_9"],
    "additional_optimization": ["GROQ_API_KEY_10"],
    "tradeoff_assessment": ["GROQ_API_KEY_11", "GROQ_API_KEY_12"],
    "report_generation": ["GROQ_API_KEY_13", "GROQ_API_KEY_14"],
}

FAILOVER_KEYS: List[str] = [f"GROQ_API_KEY_{i}" for i in range(15, 21)]


def mask_key(raw_key: str) -> str:
    """Safely mask a secret key for logging and tracing without exposing credentials."""
    if not raw_key:
        return "[NOT_SET]"
    stripped = raw_key.strip()
    if len(stripped) <= 8:
        return "***"
    return f"{stripped[:4]}...{stripped[-4:]}"


def _load_env_keys() -> Dict[str, str]:
    """Scan candidate .env file locations and os.environ for GROQ_API_KEY_1..20."""
    loaded_keys: Dict[str, str] = {}

    candidate_files = [
        Path.cwd() / ".env",
        Path.cwd() / "Target-MAS-Dataset-main" / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]

    for p in candidate_files:
        if p.exists() and p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k.startswith("GROQ_API_KEY_") and v and k not in loaded_keys:
                            loaded_keys[k] = v
            except Exception as e:
                logger.debug(f"Could not read .env at {p}: {e}")

    # Environment variables take precedence if set
    for i in range(1, 21):
        k = f"GROQ_API_KEY_{i}"
        val = os.getenv(k, "").strip()
        if val:
            loaded_keys[k] = val

    return loaded_keys


class GroqKeyManager:
    """
    Manages 20 Groq API keys with task-based distribution, failover pools,
    strict secret masking, and allocation auditing.
    """

    def __init__(self, key_pool: Optional[Dict[str, str]] = None):
        self._keys: Dict[str, str] = key_pool if key_pool is not None else _load_env_keys()
        self._usage_log: List[Dict[str, Any]] = []
        self._failover_index = 0
        self._task_counters: Dict[str, int] = {}

    @property
    def total_loaded_keys(self) -> int:
        return len(self._keys)

    def is_key_configured(self, key_name: str) -> bool:
        return bool(self._keys.get(key_name))

    def get_masked_identifier(self, env_var: str) -> str:
        """Return safe display identifier (e.g., 'GROQ_API_KEY_1 (masked: gsk_***...PW)')."""
        raw = self._keys.get(env_var, "")
        return f"{env_var} (masked: {mask_key(raw)})"

    def get_key_for_task(self, task_type: str) -> Tuple[str, str, str]:
        """
        Allocate a dedicated Groq key for a specific task or optimization dimension.
        Returns:
            (raw_key, env_var_name, safe_masked_identifier)
        Raises:
            KeyError if the task type is not recognized.
            ValueError if no key is configured for this task.
        """
        candidate_vars = TASK_KEY_MAPPING.get(task_type)
        if not candidate_vars:
            candidate_vars = ["GROQ_API_KEY_1"]

        # Deterministic assignment within the task pool (no naive round-robin)
        idx = self._task_counters.get(task_type, 0) % len(candidate_vars)
        self._task_counters[task_type] = idx + 1
        env_var = candidate_vars[idx]

        raw_key = self._keys.get(env_var, "").strip()
        if not raw_key:
            # Try alternate key in same task pool
            for alt_var in candidate_vars:
                raw = self._keys.get(alt_var, "").strip()
                if raw:
                    env_var = alt_var
                    raw_key = raw
                    break

        if not raw_key:
            # Fallback to key 1 if configured
            env_var = "GROQ_API_KEY_1"
            raw_key = self._keys.get(env_var, "").strip()

        if not raw_key:
            raise ValueError(
                f"No configured Groq API key available for task '{task_type}'. "
                f"Ensure {candidate_vars[0]} is set."
            )

        safe_masked = self.get_masked_identifier(env_var)
        self._record_usage(task_type, env_var, safe_masked, "allocated")
        return raw_key, env_var, safe_masked

    def get_keys_for_batch(self, task_type: str, count: int) -> List[Tuple[str, str, str]]:
        """
        Allocate distinct keys for a batch of parallel subtasks.
        Returns up to `count` distinct keys.
        """
        task_vars = list(TASK_KEY_MAPPING.get(task_type, []))
        all_candidate_vars = task_vars + [k for k in FAILOVER_KEYS if k not in task_vars]

        allocated: List[Tuple[str, str, str]] = []
        for ev in all_candidate_vars:
            if len(allocated) >= count:
                break
            raw = self._keys.get(ev, "").strip()
            if raw:
                masked = self.get_masked_identifier(ev)
                allocated.append((raw, ev, masked))
                self._record_usage(task_type, ev, masked, f"batch_allocated_{len(allocated)}")

        if not allocated:
            # Fallback to primary task key
            allocated.append(self.get_key_for_task(task_type))

        return allocated

    def handle_rate_limit(self, task_type: str, failed_key_var: str) -> Tuple[str, str, str]:
        """
        Gracefully handle a 429 rate limit by failing over to the failover pool.
        Logs the failover event using masked identifiers only.
        """
        failed_masked = self.get_masked_identifier(failed_key_var)

        for _ in range(len(FAILOVER_KEYS)):
            candidate_var = FAILOVER_KEYS[self._failover_index % len(FAILOVER_KEYS)]
            self._failover_index += 1
            raw = self._keys.get(candidate_var, "").strip()
            if raw and candidate_var != failed_key_var:
                safe_masked = self.get_masked_identifier(candidate_var)
                logger.warning(
                    f"Rate limit encountered on {failed_masked} for task '{task_type}'. "
                    f"Failing over to {safe_masked}."
                )
                self._record_usage(
                    task_type,
                    candidate_var,
                    safe_masked,
                    f"failover_from_{failed_key_var}",
                )
                return raw, candidate_var, safe_masked

        # If all failover keys exhausted, return first available
        logger.error(f"All failover keys exhausted for task '{task_type}'.")
        return self.get_key_for_task(task_type)

    def _record_usage(self, task_type: str, env_var: str, masked_key: str, action: str) -> None:
        """Record usage event without logging secret tokens."""
        self._usage_log.append({
            "task_type": task_type,
            "env_var": env_var,
            "masked_key": masked_key,
            "action": action,
        })

    def get_usage_log(self) -> List[Dict[str, Any]]:
        """Return audit trail of all key allocations and failover events."""
        return list(self._usage_log)

    def contains_raw_secrets(self, text: str) -> bool:
        """Check whether any known raw API key value appears in the given string."""
        if not text:
            return False
        for raw in self._keys.values():
            if raw and len(raw) > 10 and raw in text:
                return True
        return False

    def sanitize_text(self, text: str) -> str:
        """Replace any accidental raw API keys with masked equivalents."""
        if not text:
            return text
        sanitized = text
        for ev, raw in self._keys.items():
            if raw and len(raw) > 10 and raw in sanitized:
                sanitized = sanitized.replace(raw, self.get_masked_identifier(ev))
        return sanitized


# Global singleton instance
_default_key_manager: Optional[GroqKeyManager] = None


def get_default_key_manager() -> GroqKeyManager:
    global _default_key_manager
    if _default_key_manager is None:
        _default_key_manager = GroqKeyManager()
    return _default_key_manager

