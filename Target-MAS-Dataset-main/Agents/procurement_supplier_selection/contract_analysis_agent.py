"""
Agent 5: Contract Analysis Agent (Procurement)
Analyzes contractual terms, warranties, penalties, and commercial clauses.
"""

import json
from pathlib import Path
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent
from common.settings import DATA_DIR

AGENT_ID = "contract_analysis_agent"

def deterministic_contract_lookup(supplier_ids: list) -> dict:
    """Retrieves synthetic contract templates from local files."""
    cont_file = DATA_DIR / "procurement_supplier_selection" / "contracts.json"
    contracts = {}
    if cont_file.exists():
        with open(cont_file, "r", encoding="utf-8") as f:
            for c in json.load(f):
                if c.get("supplier_id") in supplier_ids:
                    contracts[c.get("supplier_id")] = c
    return contracts

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Commercial Contracts Specialist",
        goal="Audit legal terms, payment conditions, delay penalties, and warranty protections.",
        backstory="A corporate contract analyst reviewing vendor terms to protect organizational interests.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, supplier_ids: list) -> Task:
    contract_data = deterministic_contract_lookup(supplier_ids)
    description = (
        f"Analyze contract terms across all competing suppliers.\n"
        f"Contract Terms by Supplier:\n{json.dumps(contract_data, indent=2)}\n\n"
        f"Tasks:\n"
        f"1. Compare payment terms (e.g. Net 30 vs Net 45), warranty length, and delay penalties.\n"
        f"2. Identify any unfavorable or restrictive clauses.\n"
        f"Return strictly valid JSON with keys: 'contract_comparison', 'best_terms_supplier', and 'contract_risks'."
    )
    expected_output = "Strictly valid JSON containing contract_comparison, best_terms_supplier, and contract_risks."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
