"""
Agent 6: Next Action Agent (Sales Lead Qualification)
Recommends targeted next steps, sales follow-up, or lead disqualification.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "next_action_agent"

def deterministic_action_rule(qualification_tier: str) -> dict:
    """Provides deterministic baseline recommendation based on score tier."""
    if qualification_tier == "SALES_QUALIFIED_LEAD":
        return {
            "action": "schedule_executive_demo",
            "channel": "enterprise_account_exec",
            "priority": "HIGH"
        }
    elif qualification_tier == "MARKETING_NURTURE":
        return {
            "action": "enroll_in_product_nurture_campaign",
            "channel": "automated_email_drip",
            "priority": "MEDIUM"
        }
    else:
        return {
            "action": "disqualify_lead",
            "channel": "crm_status_closed_disqualified",
            "priority": "LOW"
        }

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Sales Engagement Strategy Lead",
        goal="Prescribe the highest-ROI next action without wasting expensive sales rep bandwidth on unfit leads.",
        backstory="An inside sales director orchestrating outbound sequences and qualification handoffs.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, upstream_evidence: dict) -> Task:
    score_data = upstream_evidence.get("scoring", {}).get("deterministic_score", {})
    tier = score_data.get("qualification_tier", "DISQUALIFIED")
    rule_guidance = deterministic_action_rule(tier)
    description = (
        f"Recommend the specific next commercial action for this sales lead.\n"
        f"Upstream Evidence:\n{json.dumps(upstream_evidence, indent=2)}\n\n"
        f"Rule-Based Routing Action:\n{json.dumps(rule_guidance, indent=2)}\n\n"
        f"Instructions:\n"
        f"1. Choose action: 'schedule_executive_demo', 'request_missing_info', 'marketing_nurture', or 'disqualify_lead'.\n"
        f"2. Explain why this action is optimal given budget, timeline, and fit.\n"
        f"3. Note: Never send real emails or update live external CRMs.\n"
        f"Return strictly valid JSON with keys: 'recommended_action', 'target_team', 'priority', and 'action_justification'."
    )
    expected_output = "Strictly valid JSON containing recommended_action, target_team, priority, and action_justification."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
