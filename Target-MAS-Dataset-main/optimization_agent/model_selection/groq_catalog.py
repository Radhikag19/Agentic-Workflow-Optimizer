"""
Groq Model Catalogue and Live Availability Resolver.

Maintains metadata specifications for candidate models and intersects with
live Groq API models endpoint to guarantee current operational availability.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("optimization_agent.model_selection.groq_catalog")

GROQ_MODELS_ENDPOINT = "https://api.groq.com/openai/v1/models"

# Exact 6-model catalogue from Colab prototype
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "openai/gpt-oss-120b": {
        "model_role": "general_purpose_reasoning",
        "suitable_for_general_agent_tasks": True,
        "context_window": 131072,
        "max_input_tokens": 65536,
        "max_output_tokens": 65536,
        "max_input_is_derived": True,
        "reasoning_capability": "High",
        "instruction_following": "High",
        "tool_calling_support": True,
        "input_price_per_million_usd": 0.15,
        "output_price_per_million_usd": 0.60,
        "metadata_source": "Groq model documentation; max input derived; qualitative ratings are project heuristics",
    },
    "openai/gpt-oss-20b": {
        "model_role": "general_purpose_reasoning",
        "suitable_for_general_agent_tasks": True,
        "context_window": 131072,
        "max_input_tokens": 65536,
        "max_output_tokens": 65536,
        "max_input_is_derived": True,
        "reasoning_capability": "High",
        "instruction_following": "High",
        "tool_calling_support": True,
        "input_price_per_million_usd": 0.075,
        "output_price_per_million_usd": 0.30,
        "metadata_source": "Groq model documentation; max input derived; qualitative ratings are project heuristics",
    },
    "qwen/qwen3.8-27b": {
        "model_role": "general_purpose_reasoning_multimodal",
        "suitable_for_general_agent_tasks": True,
        "context_window": 131072,
        "max_input_tokens": 114688,
        "max_output_tokens": 16384,
        "max_input_is_derived": True,
        "reasoning_capability": "High",
        "instruction_following": "High",
        "tool_calling_support": True,
        "input_price_per_million_usd": 0.80,
        "output_price_per_million_usd": 4.00,
        "metadata_source": "Groq model documentation; max input derived; qualitative ratings are project heuristics",
    },
    "openai/gpt-oss-safeguard-20b": {
        "model_role": "policy_following_safety_classifier",
        "suitable_for_general_agent_tasks": False,
        "context_window": 131072,
        "max_input_tokens": 65536,
        "max_output_tokens": 65536,
        "max_input_is_derived": True,
        "reasoning_capability": "High",
        "instruction_following": "High",
        "tool_calling_support": True,
        "input_price_per_million_usd": 0.075,
        "output_price_per_million_usd": 0.30,
        "specialization": "Safety and policy classification",
        "metadata_source": "Groq model documentation; specialized safety model; max input derived",
    },
    "meta-llama/llama-prompt-guard-2-22m": {
        "model_role": "prompt_injection_classifier",
        "suitable_for_general_agent_tasks": False,
        "context_window": 512,
        "max_input_tokens": None,
        "max_output_tokens": 512,
        "max_input_is_derived": False,
        "reasoning_capability": "Not applicable",
        "instruction_following": "Not applicable",
        "tool_calling_support": False,
        "input_price_per_million_usd": 0.03,
        "output_price_per_million_usd": 0.03,
        "specialization": "Prompt injection detection",
        "metadata_source": "Groq model documentation; specialized prompt-injection classifier",
    },
    "meta-llama/llama-prompt-guard-2-86m": {
        "model_role": "multilingual_prompt_injection_classifier",
        "suitable_for_general_agent_tasks": False,
        "context_window": 512,
        "max_input_tokens": None,
        "max_output_tokens": 512,
        "max_input_is_derived": False,
        "reasoning_capability": "Not applicable",
        "instruction_following": "Not applicable",
        "tool_calling_support": False,
        "input_price_per_million_usd": 0.04,
        "output_price_per_million_usd": 0.04,
        "specialization": "Multilingual prompt-injection detection",
        "metadata_source": "Groq model documentation; specialized prompt-injection classifier",
    },
}


def get_groq_api_key() -> str:
    """Retrieve an active Groq API key from environment."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        # Check GROQ_API_KEY_1..20
        for i in range(1, 21):
            cand = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
            if cand:
                return cand
    if not key:
        raise ValueError(
            "Missing Groq API key. Please configure GROQ_API_KEY or GROQ_API_KEY_1 in your .env file."
        )
    return key


def get_live_groq_models(
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """
    Query the authenticated live Groq models endpoint.

    Returns:
        Mapping of model_id -> dict with active status and context_window.
    """
    token = api_key or get_groq_api_key()
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(GROQ_MODELS_ENDPOINT, headers=headers, timeout=timeout)
    if not response.ok:
        err_msg = response.text[:1000].replace(token, "[REDACTED_API_KEY]")
        raise RuntimeError(
            f"Groq model-list request failed ({response.status_code}): {err_msg}"
        )

    data = response.json().get("data", [])
    return {
        item["id"]: {
            "model_id": item.get("id"),
            "owned_by": item.get("owned_by"),
            "active": item.get("active", True),
            "api_context_window": item.get("context_window"),
        }
        for item in data
        if item.get("id")
    }


def resolve_available_llms(
    live_models: Optional[Dict[str, Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Intersect MODEL_CATALOG with active models returned by Groq API.

    Args:
        live_models: Optional pre-fetched live models map (useful for offline unit testing).
        api_key: Optional explicit Groq API key.

    Returns:
        (available_llms, missing_from_live_api)
    """
    if live_models is None:
        try:
            live_models = get_live_groq_models(api_key=api_key)
        except Exception as e:
            logger.warning(f"Could not reach live Groq API ({e}). Falling back to local catalog.")
            live_models = {}

    available_llms: List[Dict[str, Any]] = []
    missing_from_live_api: List[str] = []

    for model_id, meta in MODEL_CATALOG.items():
        if live_models:
            live_meta = live_models.get(model_id)
            if not live_meta:
                missing_from_live_api.append(model_id)
                continue
            if live_meta.get("active") is False:
                missing_from_live_api.append(f"{model_id} (inactive)")
                continue

            merged = {**live_meta, **meta, "model_id": model_id}
            live_context = live_meta.get("api_context_window")
            if isinstance(live_context, (int, float)) and live_context > 0:
                merged["context_window"] = int(live_context)
            available_llms.append(merged)
        else:
            # When live_models is empty, use catalog metadata directly
            merged = {**meta, "model_id": model_id, "active": True}
            available_llms.append(merged)

    return available_llms, missing_from_live_api

