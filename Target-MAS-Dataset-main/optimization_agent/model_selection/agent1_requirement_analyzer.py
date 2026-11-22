"""
JEV Agent 1: Workflow Requirement Analyzer.

Classifies intrinsic task complexity, prompt complexity, and workflow-relative
token consumption for a target agent using TypeSafe JEV System One Score questions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from optimization_agent.model_selection.jev_client import call_jev

logger = logging.getLogger("optimization_agent.model_selection.agent1")


def build_agent1_questions(token_low_cut: float, token_high_cut: float) -> Dict[str, Any]:
    """Build the exact 3 JEV Score questions for Agent 1."""
    return {
        "task_complexity": {
            "type": "score",
            "instructions": (
                "Assess intrinsic task complexity from the agent's responsibility, goal, "
                "reasoning steps, decisions, dependencies and expected work product. "
                "Do not rate complexity merely from the agent name or token count."
            ),
            "criteria": [
                "Low: Narrow, predictable task with limited interpretation or decision-making; mostly extraction, validation, formatting or straightforward transformation.",
                "Medium: Several related operations, moderate interpretation, mapping, classification or application of multiple criteria; reasoning is bounded and structured.",
                "High: Multi-step or interdependent reasoning, synthesis across evidence, nuanced evaluation, conflict resolution, judgment or consequential recommendations.",
            ],
        },
        "prompt_complexity": {
            "type": "score",
            "instructions": (
                "Assess prompt complexity from the number and interaction of instructions, "
                "conditional rules, constraints, exceptions, reasoning requirements and output "
                "format strictness. Do not equate prompt length alone with complexity."
            ),
            "criteria": [
                "Low: Straightforward instructions with few constraints and a simple output format.",
                "Medium: Multiple instructions or constraints, some conditional logic and a moderately structured output.",
                "High: Many interdependent instructions, nested conditions or exceptions, detailed reasoning requirements and/or a strict complex output schema.",
            ],
        },
        "token_consumption": {
            "type": "score",
            "instructions": (
                "Classify the selected agent's observed mean total tokens per invocation "
                "relative to token_consumption_reference. Use the agent's measured mean, "
                "not the workflow-wide token total."
            ),
            "criteria": [
                f"Low: At or below approximately {token_low_cut:.0f} tokens per invocation.",
                f"Medium: Above approximately {token_low_cut:.0f} and at or below approximately {token_high_cut:.0f} tokens per invocation.",
                f"High: Above approximately {token_high_cut:.0f} tokens per invocation.",
            ],
        },
    }


# Static questions template for inspection
AGENT1_QUESTIONS = build_agent1_questions(token_low_cut=500.0, token_high_cut=1500.0)


def extract_score_answer(
    answer: Any,
    levels: Sequence[str] = ("Low", "Medium", "High"),
) -> Dict[str, Any]:
    """
    Extract structured classification from a JEV Score answer.

    Handles probability distributions, label keys, numerical indices, and fallback scores.
    """
    if not isinstance(answer, dict):
        raise ValueError(f"Unexpected JEV Score answer format: {answer}")

    probs = answer.get("probabilities") or {}
    if isinstance(probs, dict) and probs:
        best_key = max(probs, key=lambda k: float(probs[k]))
        try:
            idx = int(best_key)
        except (TypeError, ValueError):
            label = str(best_key)
            if label not in levels:
                raise ValueError(f"Unrecognized Score probability key '{best_key}'. Expected one of {levels}")
            idx = levels.index(label)
        confidence = float(probs[best_key])
    else:
        raw_score = answer.get("score")
        if raw_score is None:
            raise ValueError(f"JEV Score answer has neither probabilities nor score: {answer}")
        idx = int(round(float(raw_score)))
        confidence = answer.get("confidence")

    idx = max(0, min(idx, len(levels) - 1))
    return {
        "class": levels[idx],
        "score_position": answer.get("score"),
        "probabilities": probs,
        "top_level_probability": confidence,
    }


def run_agent1_requirement_analyzer(
    agent_context: Dict[str, Any],
    custom_call_jev: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute JEV Agent 1: Workflow Requirement Analyzer.

    Args:
        agent_context: The compact agent-specific evidence context.
        custom_call_jev: Optional injected call_jev function for testing/mocking.

    Returns:
        Structured requirements dict containing task_complexity, prompt_complexity,
        token_consumption classifications, and JEV usage metadata.
    """
    ref = agent_context.get("token_consumption_reference", {})
    low_cut = float(ref.get("low_upper_boundary_approx", 500.0))
    high_cut = float(ref.get("medium_upper_boundary_approx", 1500.0))

    questions = build_agent1_questions(token_low_cut=low_cut, token_high_cut=high_cut)

    caller = custom_call_jev or call_jev
    response = caller(agent_context, questions)

    answers = response.get("answers", {})
    missing = [key for key in questions if key not in answers]
    if missing:
        raise ValueError(
            f"JEV response missing Agent 1 answers {missing}. Raw response: "
            f"{json.dumps(response, ensure_ascii=False)[:3000]}"
        )

    agent_id = agent_context.get("agent_definition", {}).get("agent_id", "unknown_agent")

    return {
        "agent_id": agent_id,
        "jev_model": response.get("model"),
        "task_complexity": extract_score_answer(answers["task_complexity"]),
        "prompt_complexity": extract_score_answer(answers["prompt_complexity"]),
        "token_consumption": extract_score_answer(answers["token_consumption"]),
        "jev_usage": response.get("usage", {}),
    }

