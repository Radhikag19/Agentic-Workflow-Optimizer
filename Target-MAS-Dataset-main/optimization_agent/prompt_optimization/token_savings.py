"""
Section 4 (Dimension E): Consistent Token Savings & Consolidation Cumulative Math.

Uses the exact same tokenizer/estimator on original and optimized prompts, labels
heuristic estimates explicitly, and computes non-double-counted cumulative savings
against pre-consolidation combined prompts only when grounded evidence exists.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from optimization_agent.consolidation.token_budget import count_tokens


def calculate_token_savings(
    original_prompt: str,
    optimized_prompt: str,
    model_id: Optional[str] = None,
    pre_consolidation_original_tokens: Optional[int] = None,
    pre_consolidation_tokenization_method: Optional[str] = None,
    originated_from_consolidation: bool = False,
) -> Dict[str, Any]:
    """
    Calculate direct token savings and cumulative consolidation + prompt optimization savings.
    Never mixes tokenizers and never fabricates cumulative savings when original prompt evidence is absent.
    """
    orig_info = count_tokens(original_prompt, model_id)
    opt_info = count_tokens(optimized_prompt, model_id)

    if orig_info["method"] != opt_info["method"] and orig_info["method"] != "empty" and opt_info["method"] != "empty":
        raise ValueError(
            f"Tokenizer mismatch between original ({orig_info['method']}) and optimized ({opt_info['method']})."
        )

    orig_tokens = int(orig_info["tokens"])
    opt_tokens = int(opt_info["tokens"])
    tokens_saved = orig_tokens - opt_tokens
    pct_saved = round((tokens_saved / orig_tokens) * 100.0, 2) if orig_tokens > 0 else 0.0

    is_exact = bool(orig_info["is_exact_tokenizer"] and opt_info["is_exact_tokenizer"])
    measurement_type = "exact_tokenizer" if is_exact else "estimated_heuristic_tokenizer"

    # Check if valid pre-consolidation baseline exists for non-double-counted cumulative math
    valid_pre_cons = (
        originated_from_consolidation
        and isinstance(pre_consolidation_original_tokens, (int, float))
        and not isinstance(pre_consolidation_original_tokens, bool)
        and pre_consolidation_original_tokens > 0
        and (
            pre_consolidation_tokenization_method is None
            or pre_consolidation_tokenization_method == orig_info["method"]
        )
    )

    if valid_pre_cons:
        pre_cons_tok = int(pre_consolidation_original_tokens)
        cons_saved = pre_cons_tok - orig_tokens
        cum_saved = pre_cons_tok - opt_tokens  # == cons_saved + tokens_saved (zero double-counting)
        cum_pct = round((cum_saved / pre_cons_tok) * 100.0, 2)
        cum_status = "available"
    else:
        pre_cons_tok = None
        cons_saved = None
        cum_saved = None
        cum_pct = None
        cum_status = "unavailable_original_prompts" if originated_from_consolidation else "not_applicable_single_agent"

    return {
        "original_prompt_tokens": orig_tokens,
        "optimized_prompt_tokens": opt_tokens,
        "tokens_saved": tokens_saved,
        "token_savings_percentage": pct_saved,
        "tokenization_method": orig_info["method"],
        "is_exact_tokenizer": is_exact,
        "measurement_type": measurement_type,
        "originated_from_consolidation": bool(originated_from_consolidation),
        "pre_consolidation_original_tokens": pre_cons_tok,
        "consolidation_tokens_saved": cons_saved,
        "additional_prompt_opt_tokens_saved": tokens_saved if originated_from_consolidation else None,
        "cumulative_tokens_saved": cum_saved,
        "cumulative_savings_percentage": cum_pct,
        "cumulative_evidence_status": cum_status,
    }
