"""
Orchestrator Engine for Dimension G: Communication & Data Handoff Optimization.

Runs the 5-stage Python-first, Groq-assisted pipeline:
1. Handoff Extraction (Python)
2. Payload Size & Redundancy Analysis (Python)
3. Downstream Consumer Requirement Mapping (Python)
4. Optional Groq Semantic Analysis (ONLY on semantically ambiguous handoffs)
5. Compact Payload Recommendations & Single-Tokenizer Savings (Python)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from optimization_agent.consolidation.groq_client import GroqConsolidationClient
from optimization_agent.consolidation.token_budget import resolve_model_context_window
from optimization_agent.data_handoff.consumer_mapper import map_consumer_requirements
from optimization_agent.data_handoff.groq_semantic_analyzer import analyze_handoff_semantically_with_groq
from optimization_agent.data_handoff.handoff_extractor import extract_verified_handoffs
from optimization_agent.data_handoff.payload_analyzer import analyze_payloads
from optimization_agent.data_handoff.recommendation_engine import build_handoff_recommendation
from optimization_agent.key_manager import GroqKeyManager, get_default_key_manager
from optimization_agent.state import OptimizationAgentState


class DataHandoffOptimizationEngine:
    """Dimension G Python-first, Groq-assisted data handoff analyzer."""

    def __init__(
        self,
        key_manager: Optional[GroqKeyManager] = None,
        groq_client: Optional[GroqConsolidationClient] = None,
        model: Optional[str] = None,
    ):
        self.key_mgr = key_manager or get_default_key_manager()
        self.groq_client = groq_client
        self.model = model

    def _get_groq_client(self, baseline_model: Optional[str]) -> GroqConsolidationClient:
        if self.groq_client is not None:
            return self.groq_client
        return GroqConsolidationClient(
            key_manager=self.key_mgr,
            model=self.model or baseline_model or "openai/gpt-oss-120b",
            task_type="communication_data_handoff",
        )

    def analyze(self, state: OptimizationAgentState) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute Stages 1 through 5 and return (candidates, dimension_analysis)."""
        _, _, baseline_model = resolve_model_context_window(self.model, dict(state))
        groq_client = self._get_groq_client(baseline_model)
        masked_key = groq_client.allocated_key_masked

        profile_map: Dict[str, Dict[str, Any]] = {
            p["agent_id"]: p
            for p in ((state.get("agent_profiles") or {}).get("profiles") or [])
            if isinstance(p, dict) and p.get("agent_id")
        }
        prompt_metrics_map: Dict[str, Dict[str, Any]] = {
            pm["agent_id"]: pm
            for pm in ((state.get("prompt_metrics") or {}).get("prompt_metrics") or [])
            if isinstance(pm, dict) and pm.get("agent_id")
        }
        exec_evidence = dict(state.get("execution_evidence") or {})

        # Stage 1: Inter-agent handoff extraction (Python)
        ext_res = extract_verified_handoffs(dict(state), model_id=baseline_model)
        handoffs = ext_res["handoffs"]

        # Stage 2: Payload size and redundancy analysis (Python)
        analyzed_handoffs = analyze_payloads(handoffs, model_id=baseline_model)

        runtime_count = 0
        static_count = 0
        large_payload_count = 0
        total_redundant_fields = 0
        total_duplicate_fields = 0
        total_uncertain_fields = 0
        analyzed_without_groq = 0
        requiring_groq = 0
        groq_calls_made = 0

        total_orig_handoff_tokens = 0
        total_prop_handoff_tokens = 0
        has_runtime_savings = False

        handoffs_audit: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []

        for h in analyzed_handoffs:
            if h.get("is_statically_inferred"):
                static_count += 1
            else:
                runtime_count += 1

            pa = h.get("payload_analysis") or {}
            if pa.get("is_large_payload"):
                large_payload_count += 1

            # Stage 3: Downstream consumer requirement mapping (Python)
            c_map = map_consumer_requirements(h, profile_map, prompt_metrics_map, exec_evidence)

            red_fields = c_map.get("potentially_redundant_fields") or []
            dup_fields = c_map.get("duplicate_fields") or []
            unc_fields = c_map.get("uncertain_fields") or []

            total_redundant_fields += len(red_fields)
            total_duplicate_fields += len(dup_fields)
            total_uncertain_fields += len(unc_fields)

            # Stage 4: Optional Groq semantic analysis (ONLY when deterministic Python cannot decide)
            groq_res: Optional[Dict[str, Any]] = None
            if c_map.get("needs_groq_semantic_analysis"):
                requiring_groq += 1
                groq_calls_made += 1
                groq_res = analyze_handoff_semantically_with_groq(h, c_map, profile_map, groq_client)
            else:
                analyzed_without_groq += 1

            # Stage 5: Compact payload recommendation & token savings
            cand = build_handoff_recommendation(
                handoff=h,
                python_mapping=c_map,
                groq_result=groq_res,
                model_id=baseline_model,
                allocated_key=masked_key,
            )

            if cand is not None:
                candidates.append(cand)
                ev = cand["supporting_evidence"]
                if ev.get("original_payload_tokens") is not None and ev.get("proposed_payload_tokens") is not None:
                    has_runtime_savings = True
                    total_orig_handoff_tokens += int(ev["original_payload_tokens"])
                    total_prop_handoff_tokens += int(ev["proposed_payload_tokens"])

            handoffs_audit.append({
                "handoff_id": h["handoff_id"],
                "producer_agent": h["producer_agent"],
                "consumer_agent": h["consumer_agent"],
                "evidence_quality": h["evidence_quality"],
                "payload_source": h["payload_source"],
                "transmitted_fields": h["transmitted_fields"],
                "original_payload_tokens": pa.get("original_payload_tokens"),
                "original_payload_bytes": pa.get("original_payload_bytes"),
                "is_large_payload": pa.get("is_large_payload"),
                "required_fields": c_map.get("required_fields"),
                "used_fields": c_map.get("used_fields"),
                "potentially_redundant_fields": red_fields,
                "duplicate_fields": dup_fields,
                "uncertain_fields": unc_fields,
                "analyzed_with_groq": bool(groq_res is not None),
                "groq_accepted": bool(groq_res and groq_res.get("accepted")),
                "recommendation_generated": cand is not None,
            })

        total_tokens_saved = total_orig_handoff_tokens - total_prop_handoff_tokens if has_runtime_savings else 0
        overall_pct = (
            round((total_tokens_saved / total_orig_handoff_tokens) * 100.0, 2)
            if has_runtime_savings and total_orig_handoff_tokens > 0
            else 0.0
        )

        summary: Dict[str, Any] = {
            "total_verified_handoffs_identified": len(analyzed_handoffs),
            "handoffs_with_runtime_payload_evidence": runtime_count,
            "statically_inferred_handoffs": static_count,
            "large_payloads_detected": large_payload_count,
            "potential_redundant_fields_identified": total_redundant_fields,
            "duplicate_fields_identified": total_duplicate_fields,
            "uncertain_fields_identified": total_uncertain_fields,
            "handoffs_analyzed_without_groq": analyzed_without_groq,
            "handoffs_requiring_groq_semantic_analysis": requiring_groq,
            "groq_calls_made": groq_calls_made,
            "compact_payload_recommendations_generated": len(candidates),
            "original_payload_tokens": total_orig_handoff_tokens if has_runtime_savings else None,
            "proposed_payload_tokens": total_prop_handoff_tokens if has_runtime_savings else None,
            "tokens_saved": total_tokens_saved if has_runtime_savings else None,
            "percentage_reduction": overall_pct if has_runtime_savings else None,
            "dependency_evidence_status": ext_res["dependency_evidence_status"],
            "warnings": ext_res["warnings"],
            "allocated_key": masked_key,
        }

        return candidates, {
            "dimension": "communication_data_handoff",
            "allocated_key": masked_key,
            "candidates_found": len(candidates),
            "summary": summary,
            "handoffs_analyzed": handoffs_audit,
        }
