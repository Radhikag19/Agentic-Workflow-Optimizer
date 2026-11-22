"""
TypeSafe JEV System One API Client.

Handles authenticated communication with the TypeSafe JEV System One endpoint,
including retry logic for rate-limits/overloads and safe credential management.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("optimization_agent.model_selection.jev_client")

DEFAULT_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_JEV_MODEL = "jev-latest"


def get_jev_api_key() -> str:
    """Retrieve the TypeSafe JEV API key from environment."""
    key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "Missing TYPESAFE_API_KEY environment variable. "
            "Please ensure TYPESAFE_API_KEY is configured in your .env file."
        )
    return key


def call_jev(
    state: Dict[str, Any],
    questions: Dict[str, Any],
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    timeout: int = 90,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call TypeSafe JEV System One API.

    Args:
        state: The contextual state dictionary for the JEV evaluation.
        questions: Dictionary of JEV questions (Score or Choice).
        model: Model identifier (defaults to TYPESAFE_MODEL env or 'jev-latest').
        endpoint: JEV endpoint URL (defaults to TYPESAFE_BASE_URL env or DEFAULT_JEV_ENDPOINT).
        timeout: HTTP request timeout in seconds.
        api_key: Optional explicit API key; defaults to get_jev_api_key().

    Returns:
        JSON response from JEV API containing 'model', 'answers', 'usage'.

    Raises:
        RuntimeError: If the API request fails after retries.
    """
    token = api_key or get_jev_api_key()
    target_endpoint = endpoint or os.getenv("TYPESAFE_BASE_URL", DEFAULT_JEV_ENDPOINT)
    target_model = model or os.getenv("TYPESAFE_MODEL", DEFAULT_JEV_MODEL)

    payload = {
        "model": target_model,
        "state": state,
        "questions": questions,
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    response = None
    for delay in (0, 2, 4, 8):
        if delay > 0:
            logger.warning(f"JEV API rate-limit/overload encountered. Retrying in {delay}s...")
            time.sleep(delay)

        try:
            response = requests.post(
                target_endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code not in (429, 529, 503):
                break
        except requests.RequestException as e:
            if delay == 8:
                raise RuntimeError(f"JEV API network communication error: {e}") from e
            logger.warning(f"Network error on JEV call ({e}). Retrying...")

    if response is None:
        raise RuntimeError("No response received from JEV API.")

    if not response.ok:
        # Mask any potential secrets in error output
        err_snippet = response.text[:1500].replace(token, "[REDACTED_API_KEY]")
        raise RuntimeError(
            f"JEV API request failed with status {response.status_code}: {err_snippet}"
        )

    return response.json()

