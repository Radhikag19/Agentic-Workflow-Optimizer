"""
Dimension G: Communication & Data Handoff Optimization Subpackage.

Implements the five-stage Python-first, Groq-assisted architecture for detecting
oversized/redundant inter-agent payloads and recommending compact typed schemas
without mutating global workflow state or removing required fields.
"""

from optimization_agent.data_handoff.consumer_mapper import map_consumer_requirements
from optimization_agent.data_handoff.engine import DataHandoffOptimizationEngine
from optimization_agent.data_handoff.groq_semantic_analyzer import (
    HANDOFF_SEMANTIC_SYSTEM_PROMPT,
    MIN_SEMANTIC_CONFIDENCE,
    REQUIRED_GROQ_HANDOFF_FIELDS,
    analyze_handoff_semantically_with_groq,
    validate_groq_semantic_output,
)
from optimization_agent.data_handoff.handoff_extractor import extract_verified_handoffs
from optimization_agent.data_handoff.payload_analyzer import (
    LARGE_FIELD_TOKEN_THRESHOLD,
    LARGE_PAYLOAD_BYTE_THRESHOLD,
    LARGE_PAYLOAD_TOKEN_THRESHOLD,
    analyze_payloads,
)
from optimization_agent.data_handoff.recommendation_engine import build_handoff_recommendation

__all__ = [
    "DataHandoffOptimizationEngine",
    "HANDOFF_SEMANTIC_SYSTEM_PROMPT",
    "LARGE_FIELD_TOKEN_THRESHOLD",
    "LARGE_PAYLOAD_BYTE_THRESHOLD",
    "LARGE_PAYLOAD_TOKEN_THRESHOLD",
    "MIN_SEMANTIC_CONFIDENCE",
    "REQUIRED_GROQ_HANDOFF_FIELDS",
    "analyze_handoff_semantically_with_groq",
    "analyze_payloads",
    "build_handoff_recommendation",
    "extract_verified_handoffs",
    "map_consumer_requirements",
    "validate_groq_semantic_output",
]
