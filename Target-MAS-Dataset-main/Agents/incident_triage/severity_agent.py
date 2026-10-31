"""Severity Agent for IT Incident Triage.

Evaluates operational severity using deterministic impact formulas
combined with LLM justification and context.
"""

from typing import Dict, Any
from crewai import Agent
from common.settings import create_agent_llm


def calculate_deterministic_severity(impact_users: int, service_tier: str, outage_minutes: int) -> Dict[str, Any]:
    """Deterministic severity calculation enforcing strict SLA policy matrices."""
    is_tier_1 = service_tier.lower() == "tier-1"
    
    if (is_tier_1 and impact_users >= 5000) or outage_minutes >= 120:
        sev = "SEV-1"
        rule = "Tier-1 service impacting >= 5,000 users or outage >= 120 minutes."
    elif (is_tier_1 and impact_users >= 500) or impact_users >= 2000 or outage_minutes >= 45:
        sev = "SEV-2"
        rule = "Tier-1 service impacting >= 500 users, or cross-tier impact >= 2,000 users."
    elif impact_users >= 100 or outage_minutes >= 20:
        sev = "SEV-3"
        rule = "Moderate disruption impacting >= 100 users."
    else:
        sev = "SEV-4"
        rule = "Minor disruption impacting < 100 users with standard SLA."

    return {"recommended_severity": sev, "governing_rule": rule}


def create_severity_agent() -> Agent:
    return Agent(
        role="Severity Assessment Specialist",
        goal=(
            "Apply deterministic corporate severity thresholds and evaluate contextual impact, "
            "returning the final severity level (SEV-1 to SEV-4) and complete justification."
        ),
        backstory=(
            "You are an Incident Commander responsible for assigning definitive incident severity "
            "levels, strictly adhering to SLA policies and business continuity guidelines."
        ),
        llm=create_agent_llm("severity_agent"),
        verbose=False,
        allow_delegation=False,
    )
