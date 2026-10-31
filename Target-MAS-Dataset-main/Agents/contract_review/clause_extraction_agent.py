"""
Agent 2: Clause Extraction Agent for Contract Review and Compliance.

Responsibilities:
- Extract relevant contract clauses (payment, termination, renewal, liability, data protection).
- Assign stable clause IDs and section references.
- Preserve verbatim clause text excerpts.
- Distinguish absent clauses from extraction failures.
- Extract document once to provide reusable structured outputs.
"""

from typing import Any, Dict, List, Optional
from crewai import Agent
from common.agent_runner import create_crew_agent, run_agent_task
from common.validation import ClauseExtractionOutput
from common.metrics import MetricsTracker

AGENT_ID = "clause_extraction_agent"
ROLE = "Contract Clause Extraction Specialist"
GOAL = (
    "Extract discrete legal clauses from contract text, tagging them with standardized "
    "clause types and stable clause identifiers while preserving source excerpts."
)
BACKSTORY = (
    "You are a legal operations clause specialist. You systematically inventory contractual "
    "provisions, assigning unique IDs and recording precise section citations without modifying the legal text."
)


def create_agent(llm: Optional[Any] = None) -> Agent:
    """Create the CrewAI Clause Extraction Agent."""
    return create_crew_agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        llm=llm,
    )


def run_clause_extraction(
    contract_data: Dict[str, Any],
    agent: Optional[Agent] = None,
    tracker: Optional[MetricsTracker] = None,
) -> Dict[str, Any]:
    """Execute Clause Extraction Agent."""
    active_agent = agent or create_agent()

    c_id = contract_data.get("contract_id", "CTR-UNKNOWN")
    raw_clauses = contract_data.get("clauses", [])

    task_desc = f"""
Contract ID: {c_id}
Contract Text Clauses:
{raw_clauses}

Extract each discrete clause:
- Assign stable clause_id (e.g. CLS-xxxx-PAY, CLS-xxxx-TERM).
- Clause type ('payment_terms', 'data_protection', 'liability_cap', 'termination', 'renewal').
- Section title and section reference.
- Full verbatim clause_text excerpt.
- Identify absent standard clause types if any.

Return structured JSON matching:
{{
  "contract_id": "{c_id}",
  "clauses": [
    {{
      "clause_id": "CLS-xxxx-PAY",
      "clause_type": "payment_terms",
      "clause_title": "Section 3: Invoicing and Payment Terms",
      "clause_text": "...",
      "section_reference": "Section 3.1"
    }}
  ],
  "absent_clause_types": [],
  "extraction_notes": ["Document extracted once for downstream policy comparison"]
}}
"""
    expected_output = "Valid JSON matching ClauseExtractionOutput schema."

    return run_agent_task(
        agent=active_agent,
        agent_id=AGENT_ID,
        task_description=task_desc,
        expected_output_desc=expected_output,
        output_model=ClauseExtractionOutput,
        tracker=tracker,
        agent_name=ROLE,
    )
