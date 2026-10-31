"""
Workflow Orchestrator: Contract Review and Compliance (Workflow 3).

Executes exactly 10 CrewAI agents in sequential baseline order:
1. Contract Intake Agent (intake_agent.py)
2. Clause Extraction Agent (clause_extraction_agent.py)
3. Policy Interpretation Agent (policy_interpretation_agent.py)
4. Clause-to-Policy Mapping Agent (clause_mapping_agent.py)
5. Compliance Deviation Agent (deviation_agent.py)
6. Risk and Ambiguity Agent (risk_ambiguity_agent.py)
7. Cross-Clause Consistency Agent (consistency_agent.py)
8. Evidence and Citation Validation Agent (evidence_validation_agent.py)
9. Independent Review Agent (review_agent.py)
10. Final Compliance Report Agent (final_report_agent.py)
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional
from common.metrics import MetricsTracker
from common.tracing import init_tracing

# Import all 10 agents
from Agents.contract_review import (
    intake_agent,
    clause_extraction_agent,
    policy_interpretation_agent,
    clause_mapping_agent,
    deviation_agent,
    risk_ambiguity_agent,
    consistency_agent,
    evidence_validation_agent,
    review_agent,
    final_report_agent,
)

WORKFLOW_NAME = "contract_review"
EXPECTED_AGENT_COUNT = 10


def load_sample_contract_input(
    contract_index: int = 0,
    policy_index: int = 0,
    data_dir: str = "data/contract_review",
) -> Dict[str, Any]:
    """Load sample contract and compliance policy from local datasets."""
    contracts_path = os.path.join(data_dir, "contracts.json")
    policies_path = os.path.join(data_dir, "policies.json")

    with open(contracts_path, "r", encoding="utf-8") as f:
        contracts = json.load(f)
    with open(policies_path, "r", encoding="utf-8") as f:
        policies = json.load(f)

    contract = contracts[contract_index % len(contracts)]
    policy = policies[policy_index % len(policies)]

    return {
        "contract": contract,
        "policy": policy,
    }


class ContractReviewWorkflow:
    """Manages baseline execution of the 10-agent contract review workflow."""

    def __init__(self, llm: Optional[Any] = None, enable_tracing: bool = True):
        self.llm = llm
        if enable_tracing:
            init_tracing(project_name="awo-contract-review")

        # Instantiate all 10 CrewAI agents
        self.agent_1_intake = intake_agent.create_agent(llm=self.llm)
        self.agent_2_clause_extraction = clause_extraction_agent.create_agent(llm=self.llm)
        self.agent_3_policy_interpretation = policy_interpretation_agent.create_agent(llm=self.llm)
        self.agent_4_clause_mapping = clause_mapping_agent.create_agent(llm=self.llm)
        self.agent_5_deviation = deviation_agent.create_agent(llm=self.llm)
        self.agent_6_risk_ambiguity = risk_ambiguity_agent.create_agent(llm=self.llm)
        self.agent_7_consistency = consistency_agent.create_agent(llm=self.llm)
        self.agent_8_evidence_validation = evidence_validation_agent.create_agent(llm=self.llm)
        self.agent_9_review = review_agent.create_agent(llm=self.llm)
        self.agent_10_final_report = final_report_agent.create_agent(llm=self.llm)

    def get_agent_count(self) -> int:
        """Return the exact number of agents."""
        return EXPECTED_AGENT_COUNT

    def run(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        runs_base_dir: str = "runs",
    ) -> Dict[str, Any]:
        """
        Execute the 10-agent contract review workflow sequentially.
        """
        active_run_id = run_id or f"ctr_{uuid.uuid4().hex[:8]}"
        tracker = MetricsTracker(workflow_name=WORKFLOW_NAME, run_id=active_run_id)

        if input_data is None:
            input_data = load_sample_contract_input()

        contract_data = input_data.get("contract", {})
        policy_data = input_data.get("policy", {})

        try:
            # Stage 1: Contract Intake Agent
            intake_res = intake_agent.run_contract_intake(
                contract_data=contract_data,
                policy_data=policy_data,
                agent=self.agent_1_intake,
                tracker=tracker,
            )

            # Stage 2: Clause Extraction Agent
            clause_extraction_res = clause_extraction_agent.run_clause_extraction(
                contract_data=contract_data,
                agent=self.agent_2_clause_extraction,
                tracker=tracker,
            )

            # Stage 3: Policy Interpretation Agent
            policy_interp_res = policy_interpretation_agent.run_policy_interpretation(
                policy_data=policy_data,
                agent=self.agent_3_policy_interpretation,
                tracker=tracker,
            )

            # Stage 4: Clause-to-Policy Mapping Agent
            clause_mapping_res = clause_mapping_agent.run_clause_mapping(
                contract_id=intake_res.get("contract_id", ""),
                clause_extraction_data=clause_extraction_res,
                policy_interpretation_data=policy_interp_res,
                agent=self.agent_4_clause_mapping,
                tracker=tracker,
            )

            # Stage 5: Compliance Deviation Agent
            deviation_res = deviation_agent.run_compliance_deviation(
                contract_id=intake_res.get("contract_id", ""),
                clause_mapping_data=clause_mapping_res,
                policy_interpretation_data=policy_interp_res,
                agent=self.agent_5_deviation,
                tracker=tracker,
            )

            # Stage 6: Risk and Ambiguity Agent
            risk_res = risk_ambiguity_agent.run_risk_ambiguity(
                contract_id=intake_res.get("contract_id", ""),
                clause_extraction_data=clause_extraction_res,
                agent=self.agent_6_risk_ambiguity,
                tracker=tracker,
            )

            # Stage 7: Cross-Clause Consistency Agent
            consistency_res = consistency_agent.run_cross_clause_consistency(
                contract_id=intake_res.get("contract_id", ""),
                clause_extraction_data=clause_extraction_res,
                contract_metadata=contract_data,
                agent=self.agent_7_consistency,
                tracker=tracker,
            )

            # Stage 8: Evidence and Citation Validation Agent
            evidence_val_res = evidence_validation_agent.run_evidence_validation(
                contract_id=intake_res.get("contract_id", ""),
                clause_extraction_data=clause_extraction_res,
                policy_interpretation_data=policy_interp_res,
                deviation_data=deviation_res,
                risk_data=risk_res,
                consistency_data=consistency_res,
                agent=self.agent_8_evidence_validation,
                tracker=tracker,
            )

            # Stage 9: Independent Review Agent
            review_res = review_agent.run_independent_review(
                contract_id=intake_res.get("contract_id", ""),
                deviation_data=deviation_res,
                risk_data=risk_res,
                consistency_data=consistency_res,
                evidence_validation_data=evidence_val_res,
                agent=self.agent_9_review,
                tracker=tracker,
            )

            # Stage 10: Final Compliance Report Agent
            final_report = final_report_agent.run_final_report(
                run_id=active_run_id,
                intake_data=intake_res,
                clause_extraction_data=clause_extraction_res,
                policy_interpretation_data=policy_interp_res,
                clause_mapping_data=clause_mapping_res,
                deviation_data=deviation_res,
                risk_data=risk_res,
                consistency_data=consistency_res,
                evidence_validation_data=evidence_val_res,
                independent_review_data=review_res,
                agent=self.agent_10_final_report,
                tracker=tracker,
            )

            # Finalize metrics and persist run record
            record = tracker.finalize(
                output=final_report,
                status="success",
                runs_base_dir=runs_base_dir,
            )
            output_dict = dict(final_report) if isinstance(final_report, dict) else (final_report.model_dump() if hasattr(final_report, "model_dump") else {})
            output_dict.update({
                "run_id": active_run_id,
                "status": "SUCCESS",
                "duration_seconds": record.total_duration_seconds,
                "total_tokens": record.token_usage.get("total_tokens", 0) if record.token_usage else 0,
                "prompt_tokens": record.token_usage.get("prompt_tokens", 0) if record.token_usage else 0,
                "completion_tokens": record.token_usage.get("completion_tokens", 0) if record.token_usage else 0,
                "estimated_cost_usd": record.estimated_cost_usd or 0.0,
                "agent_executions": [m.model_dump() for m in record.agent_metrics],
                "final_output": final_report,
            })
            return output_dict

        except Exception as exc:
            tracker.finalize(
                output=None,
                status="failed",
                error=str(exc),
                runs_base_dir=runs_base_dir,
            )
            raise


def run_contract_review_workflow(
    input_data: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute contract review and compliance workflow."""
    wf = ContractReviewWorkflow(enable_tracing=True)
    return wf.run(input_data=input_data, run_id=run_id)

