"""
Orchestrator Engine for Redesigned Dimension B (Agent Consolidation).

Executes the four mandatory stages strictly in order:
1. Context-window feasibility pre-filter (70% ceiling)
2. Dynamic sequential-chain extraction & 50% cumulative token segmentation
3. Groq LLM-based independent-validation responsibility detection
4. Groq LLM-based unified prompt combination & deterministic post-validation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from optimization_agent.consolidation.groq_client import GroqConsolidationClient
from optimization_agent.consolidation.prompt_combiner import (
    check_group_io_compatibility,
    generate_and_validate_unified_prompt,
)
from optimization_agent.consolidation.segmentation import (
    extract_sequential_chains,
    segment_sequential_chain,
)
from optimization_agent.consolidation.token_budget import (
    evaluate_context_budget,
    resolve_model_context_window,
)
from optimization_agent.consolidation.validator_classifier import classify_agents_with_groq
from optimization_agent.key_manager import GroqKeyManager, get_default_key_manager
from optimization_agent.state import OptimizationAgentState


class AgentConsolidationEngine:
    """Four-stage Dimension B discovery engine integrated with AWO MCP state."""

    def __init__(
        self,
        key_manager: Optional[GroqKeyManager] = None,
        groq_client: Optional[GroqConsolidationClient] = None,
        model: Optional[str] = None,
    ):
        self.key_mgr = key_manager or get_default_key_manager()
        self.model = model
        self.groq_client = groq_client

    def _get_groq_client(self, baseline_model: Optional[str]) -> GroqConsolidationClient:
        if self.groq_client is not None:
            return self.groq_client
        return GroqConsolidationClient(
            key_manager=self.key_mgr,
            model=self.model or baseline_model or "openai/gpt-oss-120b",
        )

    def analyze(self, state: OptimizationAgentState) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run Stages 1 through 4 and return (candidates, consolidation_dim_analysis)."""
        discovered_agents = list(state.get("discovered_agents") or [])
        workflow_graph = dict(state.get("workflow_graph") or {})
        agent_profiles = dict(state.get("agent_profiles") or {})
        prompt_metrics_data = dict(state.get("prompt_metrics") or {})
        latency_metrics_data = dict(state.get("latency_metrics") or {})
        token_metrics_data = dict(state.get("token_metrics") or {})
        execution_evidence = dict(state.get("execution_evidence") or {})
        agent_run_comparisons = dict(state.get("agent_run_comparisons") or {})

        profile_map: Dict[str, Dict[str, Any]] = {
            p["agent_id"]: p
            for p in (agent_profiles.get("profiles") or [])
            if isinstance(p, dict) and p.get("agent_id")
        }
        prompt_metrics_map: Dict[str, Dict[str, Any]] = {
            pm["agent_id"]: pm
            for pm in (prompt_metrics_data.get("prompt_metrics") or [])
            if isinstance(pm, dict) and pm.get("agent_id")
        }
        latency_map: Dict[str, Dict[str, Any]] = dict(
            latency_metrics_data.get("agent_latency_metrics") or {}
        )
        token_map: Dict[str, Dict[str, Any]] = dict(
            token_metrics_data.get("agent_token_metrics") or {}
        )

        # Resolve baseline model metadata from MCP state / target_model
        _, _, baseline_model = resolve_model_context_window(self.model, dict(state))
        groq_client = self._get_groq_client(baseline_model)
        masked_key = groq_client.allocated_key_masked

        # Dynamic sequential chain extraction (dependency graph backed)
        chain_info = extract_sequential_chains(
            workflow_graph=workflow_graph,
            discovered_agents=discovered_agents,
            execution_evidence=execution_evidence,
            agent_profiles=agent_profiles,
        )
        chains: List[List[str]] = chain_info["chains"]
        dep_parents_map: Dict[str, Set[str]] = chain_info["dep_parents_map"]
        dep_children_map: Dict[str, Set[str]] = chain_info["dep_children_map"]

        original_agent_count = len(discovered_agents)
        initial_theoretical_search_space_size = (
            (2**original_agent_count) - original_agent_count - 1
            if original_agent_count >= 2
            else 0
        )

        # Enumerate unique contiguous subchains of length >= 2 (order preserved)
        c0_records: List[Dict[str, Any]] = []
        seen_subchains: Set[Tuple[str, ...]] = set()
        for ch_idx, chain in enumerate(chains, 1):
            chain_id = f"chain_{ch_idx}"
            for start in range(len(chain)):
                for end in range(start + 2, len(chain) + 1):
                    sub = chain[start:end]
                    sub_key = tuple(sub)
                    if sub_key in seen_subchains:
                        continue
                    seen_subchains.add(sub_key)
                    c0_records.append({
                        "chain_id": chain_id,
                        "original_chain": list(chain),
                        "agent_ids": list(sub),
                    })

        eliminations_audit: List[Dict[str, Any]] = []

        # =====================================================================
        # STAGE 1: Context-window feasibility pre-filter (70% ceiling)
        # =====================================================================
        c1_records: List[Dict[str, Any]] = []
        elim_context_count = 0

        for rec in c0_records:
            pre_budget = evaluate_context_budget(
                agent_ids=rec["agent_ids"],
                model_id=baseline_model,
                prompt_metrics_map=prompt_metrics_map,
                token_metrics_map=token_map,
                profile_map=profile_map,
                dep_parents_map=dep_parents_map,
                unified_prompt_text=None,
                state=dict(state),
            )
            rec["pre_budget"] = pre_budget
            if pre_budget.get("passed"):
                c1_records.append(rec)
            else:
                elim_context_count += 1
                eliminations_audit.append({
                    "stage": "stage_1_context_window_pre_filter",
                    "candidate_agents": rec["agent_ids"],
                    "reason": pre_budget.get("rejection_reason"),
                    "evidence": {
                        "evidence_status": pre_budget.get("evidence_status"),
                        "utilization_pct": pre_budget.get("utilization_pct"),
                        "total_projected_tokens": pre_budget.get("total_projected_tokens"),
                        "budget_limit_tokens": pre_budget.get("budget_limit_tokens"),
                        "context_window_capacity": pre_budget.get("context_window_capacity"),
                    },
                })

        # =====================================================================
        # STAGE 2: Sequential token-contribution segmentation (50% rule)
        # =====================================================================
        segmentation_results: List[Dict[str, Any]] = []
        chain_group_lookup: Dict[str, List[Dict[str, Any]]] = {}
        for ch_idx, chain in enumerate(chains, 1):
            chain_id = f"chain_{ch_idx}"
            seg = segment_sequential_chain(chain, token_map, chain_id=chain_id)
            segmentation_results.append(seg)
            chain_group_lookup[chain_id] = seg["groups"]

        c2_records: List[Dict[str, Any]] = []
        elim_segmentation_count = 0

        for rec in c1_records:
            groups = chain_group_lookup.get(rec["chain_id"], [])
            cand_agents = rec["agent_ids"]
            matched_group: Optional[Dict[str, Any]] = None

            for grp in groups:
                g_agents = grp["agent_ids"]
                if len(g_agents) < 2 or grp.get("is_isolated_singleton"):
                    continue
                # Must be a contiguous subslice strictly within one segmented group
                for pos in range(0, len(g_agents) - len(cand_agents) + 1):
                    if g_agents[pos : pos + len(cand_agents)] == cand_agents:
                        matched_group = grp
                        break
                if matched_group is not None:
                    break

            if matched_group is not None:
                rec["segmentation_group"] = matched_group
                c2_records.append(rec)
            else:
                elim_segmentation_count += 1
                # Determine explicit audit reason
                singleton_hits = [
                    grp["agent_ids"][0]
                    for grp in groups
                    if grp.get("is_isolated_singleton") and grp["agent_ids"][0] in cand_agents
                ]
                if singleton_hits:
                    seg_reason = (
                        f"Eliminated by 50% segmentation rule: agent(s) {singleton_hits} "
                        f"isolated as singleton (>50% chain token contribution or missing metrics)."
                    )
                else:
                    seg_reason = (
                        "Eliminated by 50% segmentation rule: candidate crosses a cumulative 50% "
                        "segmentation boundary across distinct search groups."
                    )
                eliminations_audit.append({
                    "stage": "stage_2_token_segmentation",
                    "candidate_agents": cand_agents,
                    "reason": seg_reason,
                    "evidence": {
                        "chain_id": rec["chain_id"],
                        "groups_in_chain": [
                            {
                                "group_id": g["group_id"],
                                "agent_ids": g["agent_ids"],
                                "pct_of_chain": g.get("group_contribution_pct_of_chain"),
                                "reason": g.get("segmentation_reason"),
                            }
                            for g in groups
                        ],
                    },
                })

        # =====================================================================
        # STAGE 3: Groq LLM independent-validation responsibility detection
        # =====================================================================
        remaining_agent_ids: List[str] = []
        seen_rem: Set[str] = set()
        for rec in c2_records:
            for aid in rec["agent_ids"]:
                if aid not in seen_rem:
                    seen_rem.add(aid)
                    remaining_agent_ids.append(aid)

        classifications: Dict[str, Dict[str, Any]] = {}
        if remaining_agent_ids:
            classifications = classify_agents_with_groq(
                agent_ids=remaining_agent_ids,
                groq_client=groq_client,
                profile_map=profile_map,
                prompt_metrics_map=prompt_metrics_map,
                dep_parents_map=dep_parents_map,
                dep_children_map=dep_children_map,
                execution_evidence=execution_evidence,
                agent_run_comparisons=agent_run_comparisons,
            )

        independent_validators_identified = [
            aid
            for aid, c in classifications.items()
            if c.get("exclusion_category") == "independent_validator"
        ]
        conservatively_excluded_agents = [
            aid
            for aid, c in classifications.items()
            if c.get("exclusion_category") == "conservatively_excluded_uncertain"
        ]

        c3_records: List[Dict[str, Any]] = []
        elim_validation_count = 0

        for rec in c2_records:
            excluded_in_cand = [
                aid
                for aid in rec["agent_ids"]
                if classifications.get(aid, {}).get("excluded_from_consolidation", True)
            ]
            if not excluded_in_cand:
                c3_records.append(rec)
            else:
                elim_validation_count += 1
                eliminations_audit.append({
                    "stage": "stage_3_validation_responsibility_exclusion",
                    "candidate_agents": rec["agent_ids"],
                    "reason": f"Excluded because agent(s) {excluded_in_cand} must remain separate.",
                    "evidence": {
                        aid: {
                            "responsibility_type": classifications[aid].get("responsibility_type"),
                            "is_independent_validator": classifications[aid].get("is_independent_validator"),
                            "exclusion_category": classifications[aid].get("exclusion_category"),
                            "reasoning": classifications[aid].get("reasoning"),
                            "evidence": classifications[aid].get("evidence"),
                        }
                        for aid in excluded_in_cand
                    },
                })

        # =====================================================================
        # STAGE 4: Groq LLM prompt combination & post-generation check
        # =====================================================================
        final_candidates: List[Dict[str, Any]] = []
        seen_final_groups: Set[Tuple[str, ...]] = set()
        elim_incompatible_count = 0
        elim_llm_schema_count = 0
        prompt_combinations_generated = 0
        elim_post_token_count = 0

        seg_lookup_by_chain = {s["chain_id"]: s for s in segmentation_results}

        for rec in c3_records:
            cand_agents = rec["agent_ids"]
            group_key = tuple(cand_agents)
            if group_key in seen_final_groups:
                continue
            seen_final_groups.add(group_key)

            compat_ok, compat_reason = check_group_io_compatibility(
                group_agents=cand_agents,
                dep_parents_map=dep_parents_map,
                dep_children_map=dep_children_map,
                profile_map=profile_map,
                classifications=classifications,
            )
            if not compat_ok:
                elim_incompatible_count += 1
                eliminations_audit.append({
                    "stage": "stage_4_compatibility_check",
                    "candidate_agents": cand_agents,
                    "reason": compat_reason,
                    "evidence": {},
                })
                continue

            group_id = f"CONSOLIDATE-{'-'.join(a.upper() for a in cand_agents)}"
            comb_out = generate_and_validate_unified_prompt(
                group_id=group_id,
                group_agents=cand_agents,
                groq_client=groq_client,
                model_id=baseline_model,
                profile_map=profile_map,
                prompt_metrics_map=prompt_metrics_map,
                token_metrics_map=token_map,
                dep_parents_map=dep_parents_map,
                dep_children_map=dep_children_map,
                state=dict(state),
            )

            if comb_out.get("prompt_generated"):
                prompt_combinations_generated += 1

            status = comb_out.get("status")
            if status != "accepted":
                if status == "rejected_incompatible":
                    elim_incompatible_count += 1
                elif status == "rejected_post_tokenization":
                    elim_post_token_count += 1
                else:
                    elim_llm_schema_count += 1

                eliminations_audit.append({
                    "stage": f"stage_4_{status}",
                    "candidate_agents": cand_agents,
                    "reason": comb_out.get("rejection_reason"),
                    "evidence": comb_out.get("post_budget") or {},
                })
                continue

            comb_json = comb_out["combination_result"]
            post_budget = comb_out["post_budget"]
            pre_budget = rec["pre_budget"]
            seg_res = seg_lookup_by_chain[rec["chain_id"]]
            seg_group = rec["segmentation_group"]

            # Empirical baseline stats for target agents (no fabricated savings percentages)
            per_agent_lat = {a: round(float(latency_map.get(a, {}).get("mean_seconds", 0.0)), 4) for a in cand_agents}
            combined_baseline_latency = round(sum(per_agent_lat.values()), 4)
            per_agent_in_tok = {
                a: seg_res["agent_metrics"][a]["mean_input_tokens"] for a in cand_agents
            }
            per_agent_out_tok = {
                a: seg_res["agent_metrics"][a]["mean_output_tokens"] for a in cand_agents
            }
            per_agent_contrib_idx = {
                a: seg_res["agent_metrics"][a]["contribution_index"] for a in cand_agents
            }
            per_agent_obs = {
                a: seg_res["agent_metrics"][a]["observation_count"] for a in cand_agents
            }
            combined_baseline_total_tokens = round(
                sum(per_agent_in_tok.values()) + sum(per_agent_out_tok.values()), 2
            )

            orig_prompt_tokens = post_budget.get("original_combined_prompt_tokens") or 0
            unified_prompt_tokens = post_budget.get("prompt_tokens_P") or 0
            static_prompt_token_delta = orig_prompt_tokens - unified_prompt_tokens
            est_prompt_savings_pct = (
                round((static_prompt_token_delta / orig_prompt_tokens) * 100.0, 2)
                if orig_prompt_tokens > 0
                else None
            )

            # Collect full validation classification & exclusion evidence
            chain_validation_map = {
                a: classifications[a]
                for a in cand_agents
                if a in classifications
            }
            exclusion_evidence_for_workflow = [
                {
                    "agent_id": aid,
                    "exclusion_category": cls.get("exclusion_category"),
                    "responsibility_type": cls.get("responsibility_type"),
                    "reasoning": cls.get("reasoning"),
                    "evidence": cls.get("evidence"),
                }
                for aid, cls in classifications.items()
                if cls.get("excluded_from_consolidation")
            ]

            original_responsibilities = {
                a: {
                    "role": profile_map.get(a, {}).get("role"),
                    "goal": profile_map.get(a, {}).get("goal"),
                    "declared_responsibilities": profile_map.get(a, {}).get("declared_responsibilities"),
                }
                for a in cand_agents
            }

            provenance_sources = sorted({
                "stage_2_overview",
                "stage_3_graph",
                "stage_4_profiles",
                "stage_5_prompts",
                "stage_6_latency",
                "stage_7_tokens",
                "stage_9_comparison",
                "stage_10_evidence",
                "groq_stage3_classification",
                "groq_stage4_prompt_combination",
            })

            candidate_record: Dict[str, Any] = {
                "candidate_id": group_id,
                "dimension": "agent_consolidation",
                "title": (
                    f"Consolidate {' -> '.join(cand_agents)} into Unified Agent "
                    f"({comb_json.get('combined_agent_name')})"
                ),
                "target_agents": list(cand_agents),
                "observed_problem": (
                    f"Sequential agents {' -> '.join(cand_agents)} execute as {len(cand_agents)} "
                    f"separate LLM dispatch steps along verified dependency chain {rec['chain_id']} "
                    f"(baseline combined mean latency: {combined_baseline_latency:.2f}s; "
                    f"combined mean input+output tokens: {combined_baseline_total_tokens:.0f}), "
                    f"incurring intermediate serialization and {orig_prompt_tokens} combined static prompt tokens."
                ),
                "supporting_evidence": {
                    "mcp_stage_source": ["stage_3_graph", "stage_4_profiles", "stage_5_prompts", "stage_6_latency", "stage_7_tokens", "stage_10_evidence"],
                    "original_sequential_chain": list(rec["original_chain"]),
                    "candidate_group": list(cand_agents),
                    "agent_execution_order": list(cand_agents),
                    "per_agent_input_tokens": per_agent_in_tok,
                    "per_agent_output_tokens": per_agent_out_tok,
                    "per_agent_observation_counts": per_agent_obs,
                    "per_agent_contribution_index": per_agent_contrib_idx,
                    "group_contribution": {
                        "group_id": seg_group["group_id"],
                        "group_input_tokens": seg_group["group_input_tokens"],
                        "group_output_tokens": seg_group["group_output_tokens"],
                        "group_contribution_tokens": seg_group["group_contribution_tokens"],
                        "group_contribution_pct_of_chain": seg_group["group_contribution_pct_of_chain"],
                        "total_chain_contribution": seg_res["total_chain_contribution"],
                    },
                    "segmentation_boundary_and_reason": {
                        "boundary_after_agent": seg_group.get("boundary_after_agent"),
                        "cumulative_chain_pct_start": seg_group.get("cumulative_chain_pct_start"),
                        "cumulative_chain_pct_end": seg_group.get("cumulative_chain_pct_end"),
                        "segmentation_reason": seg_group.get("segmentation_reason"),
                    },
                    "independent_validation_classification": chain_validation_map,
                    "validation_exclusion_evidence": exclusion_evidence_for_workflow,
                    "original_prompt_token_count": orig_prompt_tokens,
                    "unified_prompt_token_count": unified_prompt_tokens,
                    "context_window_capacity": post_budget.get("context_window_capacity"),
                    "context_utilization_percentage": post_budget.get("utilization_pct"),
                    "context_budget_pass_fail": {
                        "pre_filter_passed": bool(pre_budget.get("passed")),
                        "pre_filter_utilization_pct": pre_budget.get("utilization_pct"),
                        "post_generation_passed": bool(post_budget.get("passed")),
                        "post_generation_utilization_pct": post_budget.get("utilization_pct"),
                        "budget_threshold_pct": post_budget.get("max_utilization_pct"),
                        "tokenization_method": post_budget.get("tokenization_method"),
                        "is_exact_tokenizer": post_budget.get("is_exact_tokenizer"),
                    },
                    "groq_prompt_combination_result": comb_json,
                    "original_agent_responsibilities": original_responsibilities,
                    "consolidated_responsibilities": {
                        "combined_agent_name": comb_json.get("combined_agent_name"),
                        "combined_role": comb_json.get("combined_role"),
                        "combined_goal": comb_json.get("combined_goal"),
                        "preserved_responsibilities": comb_json.get("preserved_responsibilities"),
                        "merged_responsibilities": comb_json.get("merged_responsibilities"),
                        "removed_redundancies": comb_json.get("removed_redundancies"),
                        "preserved_input_schema": comb_json.get("preserved_input_schema"),
                        "preserved_output_schema": comb_json.get("preserved_output_schema"),
                    },
                    "evidence_provenance": provenance_sources,
                },
                "unified_prompt": comb_json["unified_prompt"],
                "proposed_solution": (
                    f"Consolidate sequential execution agents ({' -> '.join(cand_agents)}) into a single "
                    f"unified agent '{comb_json.get('combined_agent_name')}' using the generated schema-preserving "
                    f"unified prompt (requiring regression validation against baseline traces prior to rollout)."
                ),
                "expected_benefit": {
                    "is_estimate": True,
                    "empirical_evaluation_pending": True,
                    "eliminated_agent_handoffs_count": len(cand_agents) - 1,
                    "baseline_combined_latency_seconds": combined_baseline_latency,
                    "baseline_combined_tokens": combined_baseline_total_tokens,
                    "measured_static_prompt_token_reduction": static_prompt_token_delta,
                    "estimated_static_prompt_token_reduction_pct": est_prompt_savings_pct,
                    "estimation_basis": (
                        "Static prompt delta computed from actual prompt tokenizations; runtime latency, "
                        "end-to-end token savings, and quality preservation remain unverified estimates "
                        "pending controlled shadow regression testing."
                    ),
                },
                "tradeoffs_and_risks": {
                    "latency_tradeoff": "Eliminates intermediate dispatch round-trips; net end-to-end latency requires empirical validation.",
                    "token_cost_tradeoff": "Reduces duplicated static instruction prefill; output token impact requires shadow validation.",
                    "output_quality_risk": "Moderate (merging execution boundaries can alter instruction adherence if unvalidated).",
                    "implementation_complexity": "Medium",
                },
                "tier": "Tier 2",
                "requires_validation": True,
                "allocated_key": masked_key,
            }
            final_candidates.append(candidate_record)

        after_compat_count = len(c3_records) - elim_incompatible_count - elim_llm_schema_count
        funnel_summary = (
            f"Initial sequential candidates ({len(c0_records)}) -> "
            f"Context filtering ({len(c1_records)}) -> "
            f"Token segmentation ({len(c2_records)}) -> "
            f"Validation exclusions ({len(c3_records)}) -> "
            f"Prompt compatibility ({max(0, after_compat_count)}) -> "
            f"Final candidates ({len(final_candidates)})"
        )

        summary_metrics: Dict[str, Any] = {
            "original_agent_count": original_agent_count,
            "initial_theoretical_search_space_size": initial_theoretical_search_space_size,
            "initial_sequential_candidate_count": len(c0_records),
            "sequential_chains_identified": len(chains),
            "eliminated_by_context_window_filtering": elim_context_count,
            "eliminated_by_50pct_segmentation_rule": elim_segmentation_count,
            "independent_validation_agents_identified": len(independent_validators_identified),
            "independent_validation_agent_ids": independent_validators_identified,
            "conservatively_excluded_agent_ids": conservatively_excluded_agents,
            "eliminated_by_validation_exclusion": elim_validation_count,
            "rejected_for_incompatible_responsibilities": elim_incompatible_count,
            "rejected_for_llm_or_schema_errors": elim_llm_schema_count,
            "prompt_combinations_generated": prompt_combinations_generated,
            "rejected_after_final_prompt_tokenization": elim_post_token_count,
            "final_consolidation_opportunities": len(final_candidates),
            "search_space_reduction_summary": funnel_summary,
            "dependency_evidence_status": chain_info["dependency_evidence_status"],
            "warnings": chain_info["warnings"],
        }

        dim_analysis: Dict[str, Any] = {
            "dimension": "agent_consolidation",
            "allocated_key": masked_key,
            "candidates_found": len(final_candidates),
            "summary": summary_metrics,
            "sequential_chains": chains,
            "segmentation_groups": segmentation_results,
            "validation_classifications": list(classifications.values()),
            "eliminations_audit": eliminations_audit,
        }

        return final_candidates, dim_analysis
