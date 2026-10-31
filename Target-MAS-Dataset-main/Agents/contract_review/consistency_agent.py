"""
Agent 7: Cross-Clause Consistency Agent for Contract Review and Compliance.

Responsibilities:
- Compare related clauses for internal contradictions and incompatibilities.
- Examine interactions (termination vs auto-renewal, liability caps vs indemnity exclusions).
- Detect conflicting dates, definitions, or procedural exceptions.
- Anchor all inconsistency findings in concrete clause excerpts.
"""

from typing import Any, Dict, List, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import CrossClauseConsistencyOutput
from common.metrics import MetricsTracker

AGENT_ID = "cross_clause_consistency_agent"
ROLE = "Contract Inter-Clause Consistency Auditor"
GOAL = (
    "Identify mutual incompatibilities, conflicting terms, and chronological contradictions "
    "across separate clauses within the same contract."
)
BACKSTORY = (
    "You are a contract harmony specialist. You scrutinize the interplay between disparate sections—such "
    "as an unrestricted termination right in Section 12 vs an unconditional 2-year lock-in in Section 14—to "
    "uncover conflicting provisions."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Cross-Clause Consistency Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_cross_clause_consistency(
    contract_id: str,
    clause_extraction_data: Dict[str, Any],
    contract_metadata: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Cross-Clause Consistency Agent."""
    active_agent = agent or create_agent()

    clauses = clause_extraction_data.get("clauses", [])
    eff_date = contract_metadata.get("effective_date")
    exp_date = contract_metadata.get("expiration_date")

    task_desc = f"""
Contract ID: {contract_id}
Effective Date: {eff_date}
Expiration Date: {exp_date}
Extracted Clauses:
{clauses}

Audit for internal contradictions:
1. Check date consistency: Does expiration date precede effective date?
2. Check termination vs renewal: Does one clause grant 30-day convenience termination while another states contract CANNOT be terminated prior to term?
3. Check liability vs indemnification: Are liability caps nullified or contradicted by broad indemnity clauses?

Return structured JSON matching:
{{
  "contract_id": "{contract_id}",
  "inconsistencies": [
    {{
      "issue_id": "INC-01",
      "clause_id_a": "CLS-xxxx-TERM",
      "clause_id_b": "CLS-xxxx-REN",
      "description": "Termination clause allows 30-day convenience exit but renewal clause states contract cannot be terminated prior to full term expiration.",
      "interaction_type": "termination_vs_renewal",
      "evidence_a": "Either party may terminate for convenience...",
      "evidence_b": "Agreement automatically renews and CANNOT be terminated..."
    }}
  ]
}}
"""
    expected_output = "Valid JSON matching CrossClauseConsistencyOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=CrossClauseConsistencyOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
