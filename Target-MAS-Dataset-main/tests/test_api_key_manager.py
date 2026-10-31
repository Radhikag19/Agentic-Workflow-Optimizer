"""
Tests verifying workflow API key isolation, masking, and error reporting.
"""

import os
import pytest
from common.api_key_manager import (
    mask_key,
    get_workflow_for_agent,
    get_key_for_agent,
    get_key_for_workflow,
    AGENT_WORKFLOW_MAP,
    WORKFLOW_KEY_ENV_MAP
)

def test_key_masking_never_reveals_secret():
    """Ensure mask_key never prints secret keys."""
    secret = "gsk_abcd1234efgh5678ijkl"
    masked = mask_key(secret)
    assert secret not in masked
    assert masked == "gsk_...ijkl"
    assert mask_key("") == "[NOT_SET]"


def test_agent_workflow_mapping():
    """Ensure every agent maps to its valid workflow and env var."""
    for agent_id, wf_id in AGENT_WORKFLOW_MAP.items():
        assert get_workflow_for_agent(agent_id) == wf_id
        assert wf_id in WORKFLOW_KEY_ENV_MAP


def test_missing_key_raises_clear_error(monkeypatch):
    """Ensure missing key raises ValueError without silent fallbacks."""
    monkeypatch.delenv("GROQ_API_KEY_1", raising=False)
    monkeypatch.delenv("GROQ_API_KEY_INVOICE_PROCESSING", raising=False)
    
    with pytest.raises(ValueError) as excinfo:
        get_key_for_workflow("invoice_processing")
    
    assert "Missing dedicated Groq API key for workflow 'invoice_processing'" in str(excinfo.value)
