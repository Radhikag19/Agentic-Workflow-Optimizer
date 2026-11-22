"""
Groq-Only LLM Client for Dimension B (Agent Consolidation).

Uses GroqKeyManager ("agent_consolidation" -> GROQ_API_KEY_4 + failover pool)
exclusively, with structured JSON parsing and secret-safe error logging.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, Optional, Tuple

from optimization_agent.key_manager import GroqKeyManager, get_default_key_manager

logger = logging.getLogger("optimization_agent.consolidation.groq_client")

DEFAULT_GROQ_CONSOLIDATION_MODEL = "openai/gpt-oss-120b"


def _parse_json_response(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse a JSON object from raw LLM output with code-fence stripping."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None, "empty_llm_response"

    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "json_root_not_object"
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed, None
            return None, "json_root_not_object"
        except Exception:
            pass

    return None, "malformed_json_response"


class GroqConsolidationClient:
    """
    Groq-exclusive LLM client backed by GroqKeyManager ('agent_consolidation').
    Supports injectable chat_completion_fn for deterministic mocked unit tests.
    """

    provider_name: str = "groq"

    def __init__(
        self,
        key_manager: Optional[GroqKeyManager] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 20.0,
        chat_completion_fn: Optional[Callable[[str, str, str], str]] = None,
        task_type: str = "agent_consolidation",
    ):
        self.key_manager = key_manager or get_default_key_manager()
        self.model = model or DEFAULT_GROQ_CONSOLIDATION_MODEL
        self.timeout_seconds = timeout_seconds
        self._chat_completion_fn = chat_completion_fn
        self.task_type = task_type

        self._raw_key: str = ""
        self.allocated_env_var: str = ""
        self.allocated_key_masked: str = "[UNALLOCATED]"
        self._allocate_key()

    def _allocate_key(self) -> None:
        try:
            raw, env_var, masked = self.key_manager.get_key_for_task(self.task_type)
            self._raw_key = raw
            self.allocated_env_var = env_var
            self.allocated_key_masked = masked
        except Exception as e:
            safe_err = self.key_manager.sanitize_text(str(e))
            logger.warning(f"Key allocation unavailable for {self.task_type}: {safe_err}")
            self._raw_key = ""
            self.allocated_key_masked = f"{self.task_type} (masked: [NOT_SET])"

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Invoke Groq chat completions expecting a JSON object response."""
        if self._chat_completion_fn is not None:
            try:
                raw_text = self._chat_completion_fn(system_prompt, user_prompt, self.model)
                return _parse_json_response(raw_text)
            except Exception as e:
                safe_err = self.key_manager.sanitize_text(str(e))
                logger.warning(f"Injected Groq completion callable failed: {safe_err}")
                return None, f"llm_call_error:{type(e).__name__}"

        if not self._raw_key:
            return None, "missing_groq_api_key"

        try:
            from groq import Groq
        except ImportError:
            return None, "groq_package_not_installed"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        current_key = self._raw_key
        current_var = self.allocated_env_var

        for attempt in range(3):
            try:
                client = Groq(
                    api_key=current_key,
                    timeout=self.timeout_seconds,
                    max_retries=0,
                )
                try:
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                except Exception as inner_err:
                    # Retry once without response_format if the model rejects JSON mode
                    err_low = str(inner_err).lower()
                    if "400" in err_low or "response_format" in err_low:
                        resp = client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    else:
                        raise

                content = resp.choices[0].message.content or ""
                return _parse_json_response(content)
            except Exception as exc:
                safe_msg = self.key_manager.sanitize_text(str(exc))
                is_rate_limit = "429" in safe_msg or "rate" in safe_msg.lower()
                if is_rate_limit and attempt < 2:
                    logger.warning(
                        f"Groq rate limit hit on attempt {attempt + 1}; rotating failover key."
                    )
                    current_key, current_var, _ = self.key_manager.handle_rate_limit(
                        self.task_type, current_var
                    )
                    continue
                logger.warning(f"Groq API call failed ({type(exc).__name__}): {safe_msg[:200]}")
                return None, f"groq_api_error:{type(exc).__name__}"

        return None, "groq_rate_limit_exhausted"
