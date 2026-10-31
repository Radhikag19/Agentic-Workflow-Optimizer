"""
Agent 6: Cost Comparison Agent (Procurement)
Performs deterministic total cost calculations and price ranking.
"""

import json
from crewai import Agent, Task
from common.agent_runner import create_groq_llm_for_agent

AGENT_ID = "cost_comparison_agent"

def deterministic_cost_ranking(quotations: list) -> list:
    """Calculates total landed cost and ranks suppliers deterministically."""
    ranked = []
    for q in quotations:
        item_tot = float(q.get("item_total", 0))
        ship = float(q.get("shipping_cost", 0))
        tax = float(q.get("tax_amount", 0))
        total_landed = round(item_tot + ship + tax, 2)
        ranked.append({
            "supplier_id": q.get("supplier_id"),
            "supplier_name": q.get("supplier_name"),
            "unit_price": q.get("unit_price"),
            "item_total": item_tot,
            "shipping_cost": ship,
            "tax_amount": tax,
            "total_landed_cost": total_landed
        })
    ranked.sort(key=lambda x: x["total_landed_cost"])
    for idx, r in enumerate(ranked, 1):
        r["cost_rank"] = idx
    return ranked

def create_agent(llm=None) -> Agent:
    if llm is None:
        llm, _ = create_groq_llm_for_agent(AGENT_ID)
    return Agent(
        role="Cost & Financial Analyst",
        goal="Perform rigorous commercial cost comparison and rank suppliers by total evaluated cost.",
        backstory="A financial procurement analyst calculating total cost of ownership across competing supplier bids.",
        llm=llm,
        verbose=False
    )

def create_task(agent: Agent, quotations: list) -> Task:
    cost_ranking = deterministic_cost_ranking(quotations)
    description = (
        f"Analyze the total commercial cost of competing supplier quotations.\n"
        f"Deterministic Landed Cost Ranking from Python:\n{json.dumps(cost_ranking, indent=2)}\n\n"
        f"Tasks:\n"
        f"1. Explain the cost breakdown (unit price vs. shipping vs. tax).\n"
        f"2. Highlight the lowest-cost bidder and the cost variance to other bidders.\n"
        f"Return strictly valid JSON with keys: 'cost_ranking', 'lowest_cost_bidder', 'cost_spread_percentage', and 'commercial_summary'."
    )
    expected_output = "Strictly valid JSON containing cost_ranking, lowest_cost_bidder, cost_spread_percentage, and commercial_summary."
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
