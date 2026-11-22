"""
Orchestrator Engine for Dimension E: Prompt Optimization.

Coordinates Stage 1 prompt inspection & eligibility filtering, Stage 2 single-call
Groq compression (prioritizing Dimension B consolidated `unified_prompt` outputs),
Stage 3 deterministic post-validation, and non-double-counted token savings math.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from optimization_agent.consolidation.groq_client import GroqConsolidationClient
from optimization_agent.consolidation.token_budget import (
    extract_agent_static_prompt,
    resolve_model_context_window,
)
from optimization_agent.key_manager import GroqKeyManager, get_default_key_manager
from optimization_agent.prompt_optimization.inspector import inspect_prompt_redundancy
from optimization_agent.prompt_optimization.optimizer import optimize_prompt_with_groq
from optimization_agent.prompt_optimization.token_savings import calculate_token_savings
from optimization_agent.state import OptimizationAgentState

_CONSTRAINT_LINE_PATTERN = re.compile(
    r"\b(?:must|never|strictly|do not|only\s+json|no additional commentary|"
    r"required|threshold|limit|within\s+\d+|sla|compliance|verbatim)\b",
    re.IGNORECASE,
)


def _extract_constraints_from_text(prompt_text: str, limit: int = 8) -> List[str]:
    """Extract critical business rule / constraint lines present in prompt text."""
    constraints: List[str] = []
    seen: Set[str] = set()
    for raw_ln in (prompt_text or "").splitlines():
        ln = raw_ln.strip()
        if len(ln) < 12 or not _CONSTRAINT_LINE_PATTERN.search(ln):
            continue
        norm = re.sub(r"\s+", " ", ln.lower())
        if norm not in seen:
            seen.add(norm)
            constraints.append(ln)
            if len(constraints) >= limit:
                break
    return constraints


class PromptOptimizationEngine:
    """Dimension E engine prioritizing consolidated unified prompts before standalone agents."""

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
            task_type="prompt_optimization",
        )

    def _build_targets(
        self,
        state: OptimizationAgentState,
        consolidation_candidates: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        profile_map: Dict[str, Dict[str, Any]] = {
            p["agent_id"]: p
            for p in (state.get("agent_profiles", {}).get("profiles") or [])
            if isinstance(p, dict) and p.get("agent_id")
        }
        pm_map: Dict[str, Dict[str, Any]] = {
            pm["agent_id"]: pm
            for pm in (state.get("prompt_metrics", {}).get("prompt_metrics") or [])
            if isinstance(pm, dict) and pm.get("agent_id")
        }

        cons_list = (
            consolidation_candidates
            if consolidation_candidates is not None
            else [
                c
                for c in (state.get("optimization_candidates") or [])
                if isinstance(c, dict) and c.get("dimension") == "agent_consolidation"
            ]
        )

        targets: List[Dict[str, Any]] = []
        covered_agents: Set[str] = set()

        # Priority 1: Consolidated candidates providing a `unified_prompt`
        for c in cons_list:
            ev = c.get("supporting_evidence") or {}
            comb_res = ev.get("groq_prompt_combination_result") or {}
            unified_prompt = c.get("unified_prompt") or comb_res.get("unified_prompt")
            t_agents = list(c.get("target_agents") or [])
            if not t_agents:
                continue

            cons_resp = ev.get("consolidated_responsibilities") or {}
            req_resp = list(cons_resp.get("preserved_responsibilities") or [])
            if not req_resp:
                for aid in t_agents:
                    r = profile_map.get(aid, {}).get("declared_responsibilities")
                    if isinstance(r, str) and r.strip():
                        req_resp.append(r.strip())

            req_inputs = list(cons_resp.get("preserved_input_schema") or comb_res.get("preserved_input_schema") or [])
            if not req_inputs:
                seen_in: Set[str] = set()
                for aid in t_agents:
                    for inp in (profile_map.get(aid, {}).get("declared_inputs") or []):
                        if isinstance(inp, str) and inp.strip() and inp.lower() not in seen_in:
                            req_inputs.append(inp.strip())
                            seen_in.add(inp.lower())

            req_outputs = list(cons_resp.get("preserved_output_schema") or comb_res.get("preserved_output_schema") or [])
            if not req_outputs:
                last_aid = t_agents[-1]
                req_outputs = [
                    o.strip()
                    for o in (profile_map.get(last_aid, {}).get("declared_outputs") or [])
                    if isinstance(o, str) and o.strip()
                ]

            pre_cons_tokens = ev.get("original_prompt_token_count")
            pre_cons_method = (ev.get("context_budget_pass_fail") or {}).get("tokenization_method")
            target_id = c.get("candidate_id") or f"CONSOLIDATED_{'_'.join(t_agents)}"

            targets.append({
                "target_id": target_id,
                "target_agents": t_agents,
                "originated_from_consolidation": True,
                "source_consolidation_candidate_id": c.get("candidate_id"),
                "prompt_text": unified_prompt if isinstance(unified_prompt, str) else "",
                "prompt_source": "agent_consolidation_unified_prompt" if unified_prompt else "unavailable",
                "role": cons_resp.get("combined_role") or comb_res.get("combined_role"),
                "goal": cons_resp.get("combined_goal") or comb_res.get("combined_goal"),
                "required_responsibilities": req_resp,
                "required_inputs": [str(x).strip() for x in req_inputs if isinstance(x, str) and x.strip()],
                "required_outputs": [str(x).strip() for x in req_outputs if isinstance(x, str) and x.strip()],
                "critical_constraints": _extract_constraints_from_text(unified_prompt or ""),
                "pre_consolidation_original_tokens": pre_cons_tokens,
                "pre_consolidation_tokenization_method": pre_cons_method,
            })
            covered_agents.update(t_agents)

        # Priority 2: Individual agents not already covered by consolidated targets
        ordered_ids: List[str] = []
        seen_ids: Set[str] = set()
        for aid in list(pm_map.keys()) + list(state.get("discovered_agents") or []):
            if aid and aid not in seen_ids and aid not in covered_agents:
                seen_ids.add(aid)
                ordered_ids.append(aid)

        for aid in ordered_ids:
            prof = profile_map.get(aid, {})
            p_text, p_src = extract_agent_static_prompt(aid, pm_map, profile_map)
            # Only rely on real static prompt text or full profile spec when explicit
            pm_entry = pm_map.get(aid) or {}
            if p_src == "mcp_profile_fallback" and pm_entry and not pm_entry.get("prompt_text"):
                # If prompt_metrics explicitly shows 0 char length / empty prompt, keep missing
                if pm_entry.get("static_prompt_char_length", 0) == 0 and not pm_entry.get("sample_prompt_text"):
                    p_text, p_src = "", "unavailable"

            resp_list: List[str] = []
            decl_resp = prof.get("declared_responsibilities")
            if isinstance(decl_resp, str) and decl_resp.strip():
                resp_list.append(decl_resp.strip())
            elif isinstance(prof.get("goal"), str) and prof["goal"].strip():
                resp_list.append(prof["goal"].strip())

            req_in = [str(x).strip() for x in (prof.get("declared_inputs") or []) if isinstance(x, str) and x.strip()]
            req_out = [str(x).strip() for x in (prof.get("declared_outputs") or []) if isinstance(x, str) and x.strip()]

            targets.append({
                "target_id": aid,
                "target_agents": [aid],
                "originated_from_consolidation": False,
                "source_consolidation_candidate_id": None,
                "prompt_text": p_text,
                "prompt_source": p_src,
                "role": prof.get("role"),
                "goal": prof.get("goal"),
                "required_responsibilities": resp_list,
                "required_inputs": req_in,
                "required_outputs": req_out,
                "critical_constraints": _extract_constraints_from_text(p_text),
                "pre_consolidation_original_tokens": None,
                "pre_consolidation_tokenization_method": None,
            })

        return targets

    def analyze(
        self,
        state: OptimizationAgentState,
        consolidation_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute Stage 1 (inspection), Stage 2 (Groq call), and Stage 3 (validation)."""
        _, _, baseline_model = resolve_model_context_window(self.model, dict(state))
        groq_client = self._get_groq_client(baseline_model)
        masked_key = groq_client.allocated_key_masked

        targets = self._build_targets(state, consolidation_candidates)

        inspected_records: List[Dict[str, Any]] = []
        rejected_optimizations: List[Dict[str, Any]] = []
        final_candidates: List[Dict[str, Any]] = []

        exceeds_1000_count = 0
        flagged_rep_count = 0
        groq_calls_made = 0
        rejected_count = 0

        total_orig_tokens = 0
        total_opt_tokens = 0
        total_saved_tokens = 0
        total_cum_saved_tokens = 0
        cum_pre_cons_tokens_sum = 0
        has_any_cumulative = False

        for target in targets:
            insp = inspect_prompt_redundancy(target["prompt_text"], model_id=baseline_model)
            target["inspection"] = insp

            if insp.get("exceeds_1000_tokens"):
                exceeds_1000_count += 1
            if insp.get("flagged_for_repetition"):
                flagged_rep_count += 1

            inspected_records.append({
                "target_id": target["target_id"],
                "target_agents": target["target_agents"],
                "originated_from_consolidation": target["originated_from_consolidation"],
                "prompt_source": target["prompt_source"],
                **insp,
            })

            if not insp.get("eligible"):
                continue

            groq_calls_made += 1
            opt_res = optimize_prompt_with_groq(target, groq_client)
            if not opt_res.get("accepted"):
                rejected_count += 1
                rejected_optimizations.append({
                    "target_id": target["target_id"],
                    "target_agents": target["target_agents"],
                    "originated_from_consolidation": target["originated_from_consolidation"],
                    "rejection_reason": opt_res.get("rejection_reason"),
                    "original_prompt_retained": True,
                })
                continue

            llm_json = opt_res["result"]
            savings = calculate_token_savings(
                original_prompt=target["prompt_text"],
                optimized_prompt=llm_json["optimized_prompt"],
                model_id=baseline_model,
                pre_consolidation_original_tokens=target["pre_consolidation_original_tokens"],
                pre_consolidation_tokenization_method=target["pre_consolidation_tokenization_method"],
                originated_from_consolidation=target["originated_from_consolidation"],
            )

            if savings["tokens_saved"] <= 0:
                rejected_count += 1
                rejected_optimizations.append({
                    "target_id": target["target_id"],
                    "target_agents": target["target_agents"],
                    "originated_from_consolidation": target["originated_from_consolidation"],
                    "rejection_reason": f"no_token_reduction_achieved_delta={savings['tokens_saved']}",
                    "original_prompt_retained": True,
                })
                continue

            total_orig_tokens += savings["original_prompt_tokens"]
            total_opt_tokens += savings["optimized_prompt_tokens"]
            total_saved_tokens += savings["tokens_saved"]
            if savings["cumulative_evidence_status"] == "available":
                has_any_cumulative = True
                cum_pre_cons_tokens_sum += savings["pre_consolidation_original_tokens"]
                total_cum_saved_tokens += savings["cumulative_tokens_saved"]

            cid = f"PROMPT-OPT-{target['target_id'].upper()}"
            origin_label = (
                "Consolidated Unified Prompt"
                if target["originated_from_consolidation"]
                else f"Agent '{target['target_id']}'"
            )

            candidate_record: Dict[str, Any] = {
                "candidate_id": cid,
                "dimension": "prompt_optimization",
                "title": f"Compress Redundant Instructions in {origin_label}",
                "target_agents": list(target["target_agents"]),
                "observed_problem": (
                    f"{origin_label} requires {savings['original_prompt_tokens']} prompt tokens "
                    f"({savings['measurement_type']} via {savings['tokenization_method']}) and exhibits "
                    f"detected redundancy signals: {insp.get('detected_redundancy_signals') or ['exceeds_1000_tokens']}."
                ),
                "supporting_evidence": {
                    "mcp_stage_source": ["stage_4_profiles", "stage_5_prompts", "stage_7_tokens"]
                    + (["dimension_b_consolidation"] if target["originated_from_consolidation"] else []),
                    "original_prompt_text": target["prompt_text"],
                    "original_prompt_token_count": savings["original_prompt_tokens"],
                    "optimized_prompt_text": llm_json["optimized_prompt"],
                    "optimized_prompt_token_count": savings["optimized_prompt_tokens"],
                    "tokens_saved": savings["tokens_saved"],
                    "percentage_savings": savings["token_savings_percentage"],
                    "tokenization_method": savings["tokenization_method"],
                    "is_exact_tokenizer": savings["is_exact_tokenizer"],
                    "measurement_type": savings["measurement_type"],
                    "detected_redundancies": llm_json.get("detected_redundancies", []),
                    "compression_actions": llm_json.get("compression_actions", []),
                    "preserved_responsibilities": llm_json.get("preserved_responsibilities", []),
                    "preserved_constraints": llm_json.get("preserved_constraints", []),
                    "input_output_schema_preservation": {
                        "preserved_input_schema": llm_json.get("preserved_input_schema", []),
                        "preserved_output_schema": llm_json.get("preserved_output_schema", []),
                        "all_required_fields_verified": True,
                    },
                    "semantic_preservation_assessment": llm_json.get("semantic_preservation_assessment"),
                    "optimization_rationale": llm_json.get("optimization_rationale"),
                    "originated_from_agent_consolidation": target["originated_from_consolidation"],
                    "source_consolidation_candidate_id": target["source_consolidation_candidate_id"],
                    "cumulative_savings_breakdown": {
                        "pre_consolidation_original_tokens": savings["pre_consolidation_original_tokens"],
                        "consolidated_prompt_tokens_before_opt": (
                            savings["original_prompt_tokens"]
                            if target["originated_from_consolidation"]
                            else None
                        ),
                        "optimized_prompt_tokens": savings["optimized_prompt_tokens"],
                        "savings_from_agent_consolidation": savings["consolidation_tokens_saved"],
                        "additional_savings_from_prompt_optimization": (
                            savings["additional_prompt_opt_tokens_saved"]
                        ),
                        "cumulative_tokens_saved": savings["cumulative_tokens_saved"],
                        "cumulative_savings_percentage": savings["cumulative_savings_percentage"],
                        "cumulative_evidence_status": savings["cumulative_evidence_status"],
                    },
                },
                "optimized_prompt": llm_json["optimized_prompt"],
                "proposed_solution": (
                    f"Adopt the compressed prompt (preserving all original responsibilities, schemas, "
                    f"and constraints) while retaining the original prompt for regression validation."
                ),
                "expected_benefit": {
                    "original_prompt_tokens": savings["original_prompt_tokens"],
                    "optimized_prompt_tokens": savings["optimized_prompt_tokens"],
                    "tokens_saved": savings["tokens_saved"],
                    "token_savings_pct": savings["token_savings_percentage"],
                    "cumulative_tokens_saved": savings["cumulative_tokens_saved"],
                    "cumulative_savings_pct": savings["cumulative_savings_percentage"],
                    "measurement_type": savings["measurement_type"],
                    "tokenization_method": savings["tokenization_method"],
                    "is_estimate": not savings["is_exact_tokenizer"],
                    "quality_impact": "Requires shadow regression validation against baseline traces.",
                    "target_hypothesis_note": "20%-35% prompt compression is a target hypothesis; reported savings reflect actual measured prompt token counts.",
                },
                "tradeoffs_and_risks": {
                    "latency_tradeoff": "Lower prompt prefill volume; net runtime speedup requires empirical verification.",
                    "token_cost_tradeoff": f"Reduces static prompt footprint by {savings['tokens_saved']} tokens per invocation.",
                    "output_quality_risk": "Moderate (prompt wording edits require regression verification).",
                    "implementation_complexity": "Low",
                },
                "tier": "Tier 2",
                "requires_validation": True,
                "allocated_key": masked_key,
            }
            final_candidates.append(candidate_record)

        additional_pct = (
            round((total_saved_tokens / total_orig_tokens) * 100.0, 2)
            if total_orig_tokens > 0
            else 0.0
        )
        cum_pct = (
            round((total_cum_saved_tokens / cum_pre_cons_tokens_sum) * 100.0, 2)
            if has_any_cumulative and cum_pre_cons_tokens_sum > 0
            else None
        )

        summary: Dict[str, Any] = {
            "total_prompts_inspected": len(inspected_records),
            "prompts_exceeding_1000_tokens": exceeds_1000_count,
            "prompts_flagged_for_repetition": flagged_rep_count,
            "groq_optimization_calls_made": groq_calls_made,
            "prompts_successfully_optimized": len(final_candidates),
            "rejected_by_post_optimization_validation": rejected_count,
            "original_token_count": total_orig_tokens,
            "optimized_token_count": total_opt_tokens,
            "additional_tokens_saved": total_saved_tokens,
            "additional_token_savings_percentage": additional_pct,
            "cumulative_savings": (
                {
                    "pre_consolidation_original_tokens": cum_pre_cons_tokens_sum,
                    "cumulative_tokens_saved": total_cum_saved_tokens,
                    "cumulative_savings_percentage": cum_pct,
                }
                if has_any_cumulative
                else None
            ),
            "allocated_key": masked_key,
        }

        return final_candidates, {
            "dimension": "prompt_optimization",
            "allocated_key": masked_key,
            "candidates_found": len(final_candidates),
            "summary": summary,
            "inspected_prompts": inspected_records,
            "rejected_optimizations": rejected_optimizations,
        }
