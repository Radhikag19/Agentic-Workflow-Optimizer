"""
Dimension B: Agent Consolidation Discovery Subpackage.

Implements the four-stage, evidence-grounded Agent Consolidation pipeline:
1. Context-window feasibility pre-filter (70% threshold) & post-generation check
2. Dynamic sequential-chain extraction & 50% cumulative token segmentation
3. Groq LLM-based independent-validation responsibility detection
4. Groq LLM-based unified prompt combination & deterministic validation
"""

from optimization_agent.consolidation.engine import AgentConsolidationEngine
from optimization_agent.consolidation.groq_client import GroqConsolidationClient
from optimization_agent.consolidation.prompt_combiner import (
    REQUIRED_COMBINATION_FIELDS,
    check_group_io_compatibility,
    compute_required_group_schemas,
    generate_and_validate_unified_prompt,
    validate_combined_prompt_output,
)
from optimization_agent.consolidation.segmentation import (
    extract_sequential_chains,
    segment_sequential_chain,
)
from optimization_agent.consolidation.token_budget import (
    MAX_CONTEXT_UTILIZATION_RATIO,
    MODEL_CONTEXT_WINDOWS,
    count_tokens,
    evaluate_context_budget,
    extract_agent_static_prompt,
    resolve_model_context_window,
)
from optimization_agent.consolidation.validator_classifier import (
    MIN_CLASSIFICATION_CONFIDENCE,
    VALID_RESPONSIBILITY_TYPES,
    VALIDATOR_TYPES,
    build_agent_classification_payload,
    classify_agents_with_groq,
    validate_classification_output,
)

__all__ = [
    "AgentConsolidationEngine",
    "GroqConsolidationClient",
    "MAX_CONTEXT_UTILIZATION_RATIO",
    "MIN_CLASSIFICATION_CONFIDENCE",
    "MODEL_CONTEXT_WINDOWS",
    "REQUIRED_COMBINATION_FIELDS",
    "VALIDATOR_TYPES",
    "VALID_RESPONSIBILITY_TYPES",
    "build_agent_classification_payload",
    "check_group_io_compatibility",
    "classify_agents_with_groq",
    "compute_required_group_schemas",
    "count_tokens",
    "evaluate_context_budget",
    "extract_agent_static_prompt",
    "extract_sequential_chains",
    "generate_and_validate_unified_prompt",
    "resolve_model_context_window",
    "segment_sequential_chain",
    "validate_classification_output",
    "validate_combined_prompt_output",
]
