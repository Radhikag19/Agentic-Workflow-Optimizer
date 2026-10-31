"""Amount Agent for Employee Expense Reimbursement.

Executes deterministic Python arithmetic to verify subtotal, tax,
and total consistency, flagging mathematical discrepancies.
"""

from typing import Any, Dict
from crewai import Agent
from common.settings import create_agent_llm


def calculate_deterministic_arithmetic(subtotal: float, tax: float, total: float) -> Dict[str, Any]:
    """Perform exact deterministic financial arithmetic verification."""
    expected_total = round(subtotal + tax, 2)
    difference = round(abs(expected_total - round(total, 2)), 2)
    is_valid = difference <= 0.02  # allow 2-cent rounding tolerance

    return {
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "declared_total": round(total, 2),
        "computed_total": expected_total,
        "discrepancy": difference,
        "is_arithmetically_valid": is_valid,
        "has_negative_values": any(v < 0 for v in [subtotal, tax, total]),
    }


def create_amount_agent() -> Agent:
    return Agent(
        role="Expense Arithmetic Specialist",
        goal=(
            "Verify all arithmetic calculations on the claim: subtotal + tax = total. "
            "Flag rounding errors, negative sums, or declared total mismatches."
        ),
        backstory=(
            "You are a Forensic Financial Auditor who checks numerical calculations to the exact cent, "
            "ensuring books reconcile with mathematical precision."
        ),
        llm=create_agent_llm("amount_agent"),
        verbose=False,
        allow_delegation=False,
    )
