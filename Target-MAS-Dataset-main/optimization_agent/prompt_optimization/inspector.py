"""
Stage 1 (Dimension E): Prompt Inspection & Eligibility Filtering.

Deterministically measures static prompt length and structural redundancy signals
before deciding whether a prompt warrants a single Groq optimization call.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from optimization_agent.consolidation.token_budget import count_tokens

TOKEN_ELIGIBILITY_THRESHOLD = 1000

_ROLE_PATTERNS = re.compile(
    r"(?:you are (?:a|an)\b|as (?:a|an) [^.\n]{3,50}\b|your role is\b|your responsibility\b|\bbackstory\s*:)",
    re.IGNORECASE,
)
_PREAMBLE_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:#+\s*)?(?:instructions?|task|objective|guidelines?|system prompt|overview)\s*:",
    re.IGNORECASE,
)
_HANDOFF_PATTERNS = re.compile(
    r"(?:hand\s*off\s+to\b|pass\s+(?:this\s+)?(?:json\s+|structured\s+)?output\s+to\s+the\s+next\s+agent|"
    r"pass\s+to\s+(?:the\s+)?downstream\s+agent|intermediate\s+handoff|serialize\s+for\s+the\s+next\s+agent|"
    r"output\s+from\s+(?:the\s+)?previous\s+agent)",
    re.IGNORECASE,
)
_CONSTRAINT_PATTERNS = re.compile(
    r"(?:strictly valid json|return (?:only|strictly)[^.\n]*json|no additional commentary|"
    r"do not fabricate|do not invent|return parsable json)",
    re.IGNORECASE,
)
_ENUM_LIST_PATTERN = re.compile(
    r"""(?:\(|\[)\s*(?:['"][^'"]{2,40}['"]\s*,\s*){2,}['"][^'"]{2,40}['"]\s*(?:\)|\])""",
    re.IGNORECASE,
)


def inspect_prompt_redundancy(
    prompt_text: Optional[str],
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inspect prompt token length and concrete redundancy markers.
    Returns insufficient_evidence without fabricating metrics if prompt text is missing.
    """
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return {
            "evidence_status": "insufficient_evidence",
            "eligible": False,
            "exceeds_1000_tokens": False,
            "flagged_for_repetition": False,
            "prompt_tokens": None,
            "char_count": 0,
            "word_count": 0,
            "tokenization_method": None,
            "is_exact_tokenizer": None,
            "detected_redundancy_signals": [],
            "signal_details": {},
            "eligibility_reason": "missing_prompt_text",
        }

    char_count = len(prompt_text)
    words = prompt_text.split()
    word_count = len(words)
    tok_info = count_tokens(prompt_text, model_id)
    prompt_tokens = int(tok_info["tokens"])

    # 1. Repeated lines / sentences (>= 5 words) & repeated 6-word phrases
    raw_lines = [ln.strip() for ln in prompt_text.splitlines() if len(ln.strip().split()) >= 5]
    norm_lines = [re.sub(r"\s+", " ", ln.lower()) for ln in raw_lines]
    line_counts = Counter(norm_lines)
    repeated_lines = {k: c for k, c in line_counts.items() if c >= 2}
    repeated_line_chars = sum(len(k) * (c - 1) for k, c in repeated_lines.items())

    sentences = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", prompt_text) if len(s.strip().split()) >= 5]
    norm_sentences = [re.sub(r"\s+", " ", s.lower()) for s in sentences]
    sent_counts = Counter(norm_sentences)
    repeated_sentences = {k: c for k, c in sent_counts.items() if c >= 2}

    lower_words = [w.strip(".,;:!?\"'()[]{}").lower() for w in words]
    lower_words = [w for w in lower_words if w]
    ngram_counts: Counter = Counter()
    if len(lower_words) >= 6:
        for i in range(len(lower_words) - 5):
            ngram = " ".join(lower_words[i : i + 6])
            ngram_counts[ngram] += 1
    repeated_ngrams = {k: c for k, c in ngram_counts.items() if c >= 2}
    strong_ngram_rep = len(repeated_ngrams) >= 2 or any(c >= 3 for c in repeated_ngrams.values())

    # 2. Repeated static dictionaries / enum / schema blocks
    enum_matches = [re.sub(r"\s+", " ", m.lower()) for m in _ENUM_LIST_PATTERN.findall(prompt_text)]
    enum_counts = Counter(enum_matches)
    repeated_dicts = {k: c for k, c in enum_counts.items() if c >= 2}

    schema_key_lines = [
        re.sub(r"\s+", " ", ln.strip().lower())
        for ln in prompt_text.splitlines()
        if re.match(r"^\s*[-*]\s*`?[a-zA-Z_][a-zA-Z0-9_]*`?\s*(?:\(|:)", ln)
    ]
    repeated_schema_lines = {k: c for k, c in Counter(schema_key_lines).items() if c >= 2}

    # 3. Duplicate role & responsibility descriptions
    role_matches = _ROLE_PATTERNS.findall(prompt_text)

    # 4. Repetitive system / task preambles
    preamble_matches = _PREAMBLE_PATTERNS.findall(prompt_text)

    # 5. Redundant intermediate handoff instructions
    handoff_matches = _HANDOFF_PATTERNS.findall(prompt_text)

    # 6. Repeated constraints or formatting instructions
    constraint_matches = _CONSTRAINT_PATTERNS.findall(prompt_text)

    detected_signals: List[str] = []
    if repeated_lines or repeated_sentences or strong_ngram_rep:
        detected_signals.append("repeated_instructions_or_phrases")
    if repeated_dicts or repeated_schema_lines:
        detected_signals.append("repeated_static_dictionaries_or_schema")
    if len(role_matches) >= 2:
        detected_signals.append("duplicate_role_or_responsibility_descriptions")
    if len(preamble_matches) >= 2:
        detected_signals.append("repetitive_system_or_task_preambles")
    if len(handoff_matches) >= 1:
        detected_signals.append("redundant_intermediate_handoff_instructions")
    if len(constraint_matches) >= 2:
        detected_signals.append("repeated_constraints_or_formatting")

    duplicate_line_ratio = round(repeated_line_chars / max(char_count, 1), 4)

    exceeds_1000_tokens = prompt_tokens > TOKEN_ELIGIBILITY_THRESHOLD
    flagged_for_repetition = bool(
        len(detected_signals) >= 2
        or duplicate_line_ratio >= 0.20
        or len(repeated_sentences) >= 2
        or len(repeated_dicts) >= 1
        or len(repeated_schema_lines) >= 2
    )
    eligible = exceeds_1000_tokens or flagged_for_repetition

    if exceeds_1000_tokens and flagged_for_repetition:
        reason = "exceeds_1000_tokens_and_substantial_repetition"
    elif exceeds_1000_tokens:
        reason = "exceeds_1000_tokens"
    elif flagged_for_repetition:
        reason = "substantial_repetition_detected"
    else:
        reason = "concise_prompt_below_threshold_no_substantial_repetition"

    return {
        "evidence_status": "sufficient",
        "eligible": eligible,
        "exceeds_1000_tokens": exceeds_1000_tokens,
        "flagged_for_repetition": flagged_for_repetition,
        "prompt_tokens": prompt_tokens,
        "char_count": char_count,
        "word_count": word_count,
        "tokenization_method": tok_info["method"],
        "is_exact_tokenizer": tok_info["is_exact_tokenizer"],
        "detected_redundancy_signals": detected_signals,
        "signal_details": {
            "repeated_line_count": len(repeated_lines),
            "repeated_sentence_count": len(repeated_sentences),
            "repeated_ngram_count": len(repeated_ngrams),
            "repeated_static_dict_count": len(repeated_dicts),
            "repeated_schema_line_count": len(repeated_schema_lines),
            "role_marker_count": len(role_matches),
            "preamble_marker_count": len(preamble_matches),
            "handoff_marker_count": len(handoff_matches),
            "constraint_marker_count": len(constraint_matches),
            "duplicate_line_ratio": duplicate_line_ratio,
        },
        "eligibility_reason": reason,
    }
