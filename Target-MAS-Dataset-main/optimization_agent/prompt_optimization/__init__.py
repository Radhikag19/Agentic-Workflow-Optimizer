"""
Dimension E: Prompt Optimization Subpackage.

Implements evidence-grounded prompt inspection, single-call Groq compression
(prioritizing Dimension B consolidated `unified_prompt` outputs when present),
deterministic semantic/schema validation, and consistent token savings math.
"""

from optimization_agent.prompt_optimization.engine import PromptOptimizationEngine
from optimization_agent.prompt_optimization.inspector import (
    TOKEN_ELIGIBILITY_THRESHOLD,
    inspect_prompt_redundancy,
)
from optimization_agent.prompt_optimization.optimizer import (
    PROMPT_OPTIMIZATION_SYSTEM_PROMPT,
    REQUIRED_OPTIMIZATION_FIELDS,
    build_optimization_request_payload,
    optimize_prompt_with_groq,
    validate_optimized_prompt_output,
)
from optimization_agent.prompt_optimization.token_savings import calculate_token_savings

__all__ = [
    "PROMPT_OPTIMIZATION_SYSTEM_PROMPT",
    "PromptOptimizationEngine",
    "REQUIRED_OPTIMIZATION_FIELDS",
    "TOKEN_ELIGIBILITY_THRESHOLD",
    "build_optimization_request_payload",
    "calculate_token_savings",
    "inspect_prompt_redundancy",
    "optimize_prompt_with_groq",
    "validate_optimized_prompt_output",
]
