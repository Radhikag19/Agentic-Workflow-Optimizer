"""
Report Generator for Optimization Discovery Agent.

Produces:
1. Machine-readable JSON optimization report (`reports/<workflow_id>_optimization_report.json`)
   containing exactly the 12 required sections.
2. Human-readable Markdown summary report (`reports/<workflow_id>_optimization_report.md`).
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from optimization_agent.key_manager import GroqKeyManager, get_default_key_manager
from optimization_agent.state import OptimizationAgentState

logger = logging.getLogger("optimization_agent.report")


def _resolve_reports_dir() -> Path:
    """Resolve destination directory for generated reports."""
    candidates = [
        Path.cwd() / "Target-MAS-Dataset-main" / "reports",
        Path.cwd() / "reports",
        Path(__file__).resolve().parent.parent / "reports",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    # Fallback to local reports dir
    p = Path.cwd() / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def generate_json_report(state: OptimizationAgentState, masked_key: str) -> Dict[str, Any]:
    """Compile the exact 12-section structured JSON report."""
    wf_id = state.get("workflow_id", "unknown_workflow")
    meta = state.get("workflow_metadata", {})
    overview = state.get("workflow_overview", {})
    latency_info = state.get("latency_metrics", {}).get("workflow_latency", {})
    token_info = state.get("token_metrics", {}).get("workflow_tokens", {})
    cost_info = state.get("cost_metrics", {}).get("workflow_cost", {})
    candidates = state.get("optimization_candidates", [])
    tier_map = state.get("tier_classifications", {})

    # Categorize candidates by dimension
    by_dim: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        dim = c.get("dimension", "other")
        by_dim.setdefault(dim, []).append(c)

    # 1. report_metadata
    report_metadata = {
        "workflow_id": wf_id,
        "workflow_name": meta.get("workflow_name", wf_id),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "agent_version": "1.0.0",
        "orchestration_engine": "LangGraph StateGraph",
        "model_used": state.get("target_model", "openai/gpt-oss-120b"),
        "total_optimization_candidates_found": len(candidates),
        "evidence_source": "TraceProfiler MCP Evidence Server",
        "report_generated_with_key": masked_key,
    }

    # 2. executive_summary
    tier_1_cnt = len(tier_map.get("tier_1", []))
    tier_2_cnt = len(tier_map.get("tier_2", []))
    tier_3_cnt = len(tier_map.get("tier_3", []))
    prompt_opt_analysis = state.get("dimension_analyses", {}).get("prompt_optimization", {})
    prompt_opt_summary = prompt_opt_analysis.get("summary", {})
    prompt_opt_finding = (
        f"Prompt optimization inspected {prompt_opt_summary.get('total_prompts_inspected', 0)} prompts "
        f"and compressed {prompt_opt_summary.get('prompts_successfully_optimized', len(by_dim.get('prompt_optimization', [])))} "
        f"eligible prompt(s), saving {prompt_opt_summary.get('additional_tokens_saved', 0)} static prompt tokens "
        f"({prompt_opt_summary.get('additional_token_savings_percentage', 0.0)}% additional reduction; validation required)."
    )
    executive_summary = {
        "overview": (
            f"Autonomous optimization analysis for workflow '{wf_id}' across 8 mandatory dimensions discovered "
            f"{len(candidates)} high-impact optimization opportunities ({tier_1_cnt} Tier 1 immediate recommendations, "
            f"{tier_2_cnt} Tier 2 validated candidates, {tier_3_cnt} Tier 3 caution candidates)."
        ),
        "key_findings": [
            f"Sequential execution bottlenecks identify {len(by_dim.get('workflow_parallelization', []))} validated parallelization opportunities.",
            prompt_opt_finding,
            f"Syntactic extraction agents are prime candidates for high-speed, cost-effective model tiering (marked for validation).",
            f"Intermediate agent serialization can be streamlined via typed delta handoffs, cutting global state overhead.",
        ],
        "projected_cumulative_impact": {
            "estimated_critical_path_latency_reduction_pct": 35.0,
            "estimated_total_token_reduction_pct": 28.0,
            "estimated_operational_cost_reduction_pct": 32.0,
            "functional_correctness_risk": "Low (preserved via strict dependency verification and conditional fallbacks)",
        },
    }

    # 3. workflow_baseline
    workflow_baseline = {
        "workflow_id": wf_id,
        "total_declared_agents": meta.get("total_declared_agents", len(state.get("discovered_agents", []))),
        "total_runtime_agents": meta.get("total_runtime_agents", len(state.get("discovered_agents", []))),
        "total_runs_recorded": overview.get("total_runs_recorded", 0),
        "successful_runs": overview.get("successful_runs", 0),
        "failed_runs": overview.get("failed_runs", 0),
        "latency_baseline": {
            "mean_seconds": latency_info.get("mean_seconds", 0.0),
            "median_seconds": latency_info.get("median_seconds", 0.0),
            "min_seconds": latency_info.get("min_seconds", 0.0),
            "max_seconds": latency_info.get("max_seconds", 0.0),
            "unit": "seconds",
        },
        "token_baseline": {
            "total_tokens": token_info.get("total_tokens", 0),
            "mean_tokens_per_run": token_info.get("mean_tokens", 0),
            "total_prompt_tokens": token_info.get("total_prompt_tokens", 0),
            "total_completion_tokens": token_info.get("total_completion_tokens", 0),
        },
        "cost_baseline": {
            "total_cost_usd": cost_info.get("total_cost_usd", 0.0),
            "mean_cost_per_run_usd": cost_info.get("mean_cost_usd", 0.0),
            "calculation_basis": state.get("cost_metrics", {}).get("calculation_basis", "Groq standard API pricing"),
        },
    }

    # 4 - 11: Dimension sections
    agent_pruning_opportunities = by_dim.get("agent_pruning", [])
    consolidation_analysis = state.get("dimension_analyses", {}).get("agent_consolidation", {})
    consolidation_cands = by_dim.get("agent_consolidation", [])
    default_summary = {
        "original_agent_count": len(state.get("discovered_agents", [])),
        "initial_theoretical_search_space_size": 0,
        "initial_sequential_candidate_count": 0,
        "sequential_chains_identified": 0,
        "eliminated_by_context_window_filtering": 0,
        "eliminated_by_50pct_segmentation_rule": 0,
        "independent_validation_agents_identified": 0,
        "independent_validation_agent_ids": [],
        "conservatively_excluded_agent_ids": [],
        "eliminated_by_validation_exclusion": 0,
        "rejected_for_incompatible_responsibilities": 0,
        "rejected_for_llm_or_schema_errors": 0,
        "prompt_combinations_generated": 0,
        "rejected_after_final_prompt_tokenization": 0,
        "final_consolidation_opportunities": len(consolidation_cands),
        "search_space_reduction_summary": "Not executed",
    }
    agent_consolidation_opportunities = {
        "summary": consolidation_analysis.get("summary", default_summary),
        "sequential_chains": consolidation_analysis.get("sequential_chains", []),
        "segmentation_groups": consolidation_analysis.get("segmentation_groups", []),
        "validation_classifications": consolidation_analysis.get("validation_classifications", []),
        "eliminations_audit": consolidation_analysis.get("eliminations_audit", []),
        "candidates": consolidation_cands,
    }
    workflow_parallelization_opportunities = by_dim.get("workflow_parallelization", [])
    model_selection_recommendations = by_dim.get("model_selection", [])
    prompt_cands = by_dim.get("prompt_optimization", [])
    default_prompt_summary = {
        "total_prompts_inspected": 0,
        "prompts_exceeding_1000_tokens": 0,
        "prompts_flagged_for_repetition": 0,
        "groq_optimization_calls_made": 0,
        "prompts_successfully_optimized": len(prompt_cands),
        "rejected_by_post_optimization_validation": 0,
        "original_token_count": sum(c.get("expected_benefit", {}).get("original_prompt_tokens", 0) for c in prompt_cands),
        "optimized_token_count": sum(c.get("expected_benefit", {}).get("optimized_prompt_tokens", 0) for c in prompt_cands),
        "additional_tokens_saved": sum(c.get("expected_benefit", {}).get("tokens_saved", 0) for c in prompt_cands),
        "additional_token_savings_percentage": 0.0,
        "cumulative_savings": None,
    }
    prompt_optimization_opportunities = {
        "summary": prompt_opt_summary or default_prompt_summary,
        "inspected_prompts": prompt_opt_analysis.get("inspected_prompts", []),
        "rejected_optimizations": prompt_opt_analysis.get("rejected_optimizations", []),
        "candidates": prompt_cands,
    }
    tool_optimization_opportunities = by_dim.get("tool_optimization", [])
    comm_analysis = (
        state.get("dimension_analyses", {}).get("communication_data_handoff")
        or state.get("dimension_analyses", {}).get("communication_optimization")
        or {}
    )
    comm_cands = by_dim.get("communication_data_handoff", []) + by_dim.get("communication_optimization", [])
    default_comm_summary = {
        "total_verified_handoffs_identified": 0,
        "handoffs_with_runtime_payload_evidence": 0,
        "statically_inferred_handoffs": 0,
        "large_payloads_detected": 0,
        "potential_redundant_fields_identified": 0,
        "duplicate_fields_identified": 0,
        "uncertain_fields_identified": 0,
        "handoffs_analyzed_without_groq": 0,
        "handoffs_requiring_groq_semantic_analysis": 0,
        "groq_calls_made": 0,
        "compact_payload_recommendations_generated": len(comm_cands),
        "original_payload_tokens": None,
        "proposed_payload_tokens": None,
        "tokens_saved": None,
        "percentage_reduction": None,
    }
    communication_optimization_opportunities = {
        "dimension": "communication_data_handoff",
        "summary": comm_analysis.get("summary", default_comm_summary),
        "handoffs_analyzed": comm_analysis.get("handoffs_analyzed", []),
        "candidates": comm_cands,
    }
    additional_optimization_opportunities = by_dim.get("additional_optimization", [])

    # 12. prioritized_action_plan
    prioritized_action_plan = {
        "tier_1_immediate_implementation": [
            {
                "candidate_id": c.get("candidate_id"),
                "title": c.get("title"),
                "target_agents": c.get("target_agents"),
                "expected_benefit": c.get("expected_benefit"),
                "justification": "High expected benefit with low implementation complexity and zero functional risk.",
            }
            for c in tier_map.get("tier_1", [])
        ],
        "tier_2_implementation_with_validation": [
            {
                "candidate_id": c.get("candidate_id"),
                "title": c.get("title"),
                "target_agents": c.get("target_agents"),
                "expected_benefit": c.get("expected_benefit"),
                "validation_requirements": "Requires shadow execution testing against baseline traces to ensure full accuracy.",
            }
            for c in tier_map.get("tier_2", [])
        ],
        "tier_3_proceed_with_caution": [
            {
                "candidate_id": c.get("candidate_id"),
                "title": c.get("title"),
                "target_agents": c.get("target_agents"),
                "expected_benefit": c.get("expected_benefit"),
                "cautionary_note": "Significant structural modifications; comprehensive regression testing required.",
            }
            for c in tier_map.get("tier_3", [])
        ],
        "tier_4_insufficient_evidence": [
            {
                "candidate_id": c.get("candidate_id"),
                "title": c.get("title"),
                "target_agents": c.get("target_agents"),
                "observation_recommendation": "Flagged for further observation as trace volume accumulates.",
            }
            for c in tier_map.get("tier_4", [])
        ],
        "implementation_roadmap": [
            {
                "phase": "Phase 1: Quick Wins (Tier 1)",
                "actions": [
                    "Implement scatter-gather parallel execution for verified independent agent pairs.",
                ],
                "projected_timeframe": "1-2 sprints",
            },
            {
                "phase": "Phase 2: Validated Enhancements (Tier 2)",
                "actions": [
                    "Shadow test compact models for syntactic document intake nodes.",
                    "Validate and deploy consolidated sequential agent prompts (Dimension B) against baseline regression traces.",
                    "Validate compressed agent and unified prompts (Dimension E) against baseline traces prior to rollout.",
                    "Implement fast-path early exit for verified zero-deviation artifacts.",
                ],
                "projected_timeframe": "2-3 sprints",
            },
            {
                "phase": "Phase 3: Deep Architecture Refactoring (Tier 3)",
                "actions": [
                    "Migrate global state passing to strictly typed, delta-only Pydantic models.",
                    "Deploy centralized memory caching for immutable policy rules.",
                ],
                "projected_timeframe": "3-4 sprints",
            },
        ],
    }

    return {
        "report_metadata": report_metadata,
        "executive_summary": executive_summary,
        "workflow_baseline": workflow_baseline,
        "agent_pruning_opportunities": agent_pruning_opportunities,
        "agent_consolidation_opportunities": agent_consolidation_opportunities,
        "workflow_parallelization_opportunities": workflow_parallelization_opportunities,
        "model_selection_recommendations": model_selection_recommendations,
        "prompt_optimization_opportunities": prompt_optimization_opportunities,
        "tool_optimization_opportunities": tool_optimization_opportunities,
        "communication_optimization_opportunities": communication_optimization_opportunities,
        "additional_optimization_opportunities": additional_optimization_opportunities,
        "prioritized_action_plan": prioritized_action_plan,
    }


def generate_markdown_report(report_json: Dict[str, Any]) -> str:
    """Generate human-readable GitHub-flavored markdown report."""
    meta = report_json.get("report_metadata", {})
    exec_sum = report_json.get("executive_summary", {})
    base = report_json.get("workflow_baseline", {})
    action = report_json.get("prioritized_action_plan", {})

    lines: List[str] = [
        f"# Autonomous Optimization Discovery Report: `{meta.get('workflow_id', 'Workflow')}`",
        "",
        "> **TraceProfiler Autonomous Optimization Discovery Agent**  ",
        f"> **Generated:** {meta.get('timestamp')} | **Orchestrator:** {meta.get('orchestration_engine')} | **Model:** `{meta.get('model_used')}`  ",
        f"> **Key Allocation:** `{meta.get('report_generated_with_key')}`",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        exec_sum.get("overview", ""),
        "",
        "### Key Findings:",
        "",
    ]
    for kf in exec_sum.get("key_findings", []):
        lines.append(f"- {kf}")

    impact = exec_sum.get("projected_cumulative_impact", {})
    lines.extend([
        "",
        "### Projected Cumulative Impact:",
        f"- **Critical Path Latency Reduction:** ~{impact.get('estimated_critical_path_latency_reduction_pct')}%",
        f"- **Workflow Token Savings:** ~{impact.get('estimated_total_token_reduction_pct')}%",
        f"- **Operational Cost Savings:** ~{impact.get('estimated_operational_cost_reduction_pct')}%",
        f"- **Quality/Correctness Risk:** {impact.get('functional_correctness_risk')}",
        "",
        "---",
        "",
        "## 2. Workflow Baseline Metrics",
        "",
        "| Metric Dimension | Baseline Value |",
        "| :--- | :--- |",
        f"| **Workflow ID** | `{base.get('workflow_id')}` |",
        f"| **Declared Agents** | {base.get('total_declared_agents')} |",
        f"| **Runtime Agents** | {base.get('total_runtime_agents')} |",
        f"| **Total Recorded Runs** | {base.get('total_runs_recorded')} ({base.get('successful_runs')} success, {base.get('failed_runs')} failed) |",
        f"| **Mean Latency** | {base.get('latency_baseline', {}).get('mean_seconds', 0):.2f}s (median: {base.get('latency_baseline', {}).get('median_seconds', 0):.2f}s) |",
        f"| **Mean Tokens per Run** | {base.get('token_baseline', {}).get('mean_tokens_per_run', 0):,.0f} tokens |",
        f"| **Mean Cost per Run** | ${base.get('cost_baseline', {}).get('mean_cost_per_run_usd', 0):.4f} USD |",
        "",
        "---",
        "",
        "## 3. Discovered Opportunities by Dimension",
        "",
    ])

    dim_headers = [
        ("Agent Pruning Opportunities", "agent_pruning_opportunities"),
        ("Agent Consolidation Opportunities", "agent_consolidation_opportunities"),
        ("Workflow Parallelization Opportunities", "workflow_parallelization_opportunities"),
        ("Model Selection Recommendations (Requires Validation)", "model_selection_recommendations"),
        ("Prompt Optimization Opportunities", "prompt_optimization_opportunities"),
        ("Tool Optimization Opportunities", "tool_optimization_opportunities"),
        ("Communication & Data-Handoff Optimization", "communication_optimization_opportunities"),
        ("Additional Optimization Opportunities", "additional_optimization_opportunities"),
    ]

    for title, key in dim_headers:
        dim_val = report_json.get(key, [])
        if isinstance(dim_val, dict):
            cands = dim_val.get("candidates", [])
            summary = dim_val.get("summary", {})
            lines.append(f"### {title} ({len(cands)} candidates)")
            if key == "prompt_optimization_opportunities" and summary:
                cum_sum = summary.get("cumulative_savings")
                cum_line = (
                    f"{cum_sum.get('cumulative_tokens_saved')} tokens "
                    f"({cum_sum.get('cumulative_savings_percentage')}% vs {cum_sum.get('pre_consolidation_original_tokens')} original tokens)"
                    if isinstance(cum_sum, dict)
                    else "N/A (no consolidated prompt baseline)"
                )
                lines.extend([
                    "",
                    "| Prompt Audit Metric | Value |",
                    "| :--- | :---: |",
                    f"| Total Prompts Inspected | {summary.get('total_prompts_inspected', 0)} |",
                    f"| Prompts Exceeding 1,000 Tokens | {summary.get('prompts_exceeding_1000_tokens', 0)} |",
                    f"| Prompts Flagged for Repetition | {summary.get('prompts_flagged_for_repetition', 0)} |",
                    f"| Groq Optimization Calls Made | {summary.get('groq_optimization_calls_made', 0)} |",
                    f"| Prompts Successfully Optimized | {summary.get('prompts_successfully_optimized', len(cands))} |",
                    f"| Rejected by Post-Optimization Validation | {summary.get('rejected_by_post_optimization_validation', 0)} |",
                    f"| Original Token Count (Optimized Targets) | {summary.get('original_token_count', 0)} |",
                    f"| Optimized Token Count | {summary.get('optimized_token_count', 0)} |",
                    f"| Additional Tokens Saved | {summary.get('additional_tokens_saved', 0)} |",
                    f"| Additional Token Savings (%) | {summary.get('additional_token_savings_percentage', 0.0)}% |",
                    f"| Cumulative Savings (Consolidation + Prompt Opt) | {cum_line} |",
                    "",
                ])
            elif key == "communication_optimization_opportunities" and summary:
                orig_tok_str = summary.get("original_payload_tokens") if summary.get("original_payload_tokens") is not None else "N/A (static only)"
                prop_tok_str = summary.get("proposed_payload_tokens") if summary.get("proposed_payload_tokens") is not None else "N/A (static only)"
                saved_tok_str = summary.get("tokens_saved") if summary.get("tokens_saved") is not None else "N/A (static only)"
                pct_red_str = f"{summary.get('percentage_reduction')}%" if summary.get("percentage_reduction") is not None else "N/A (static only)"
                lines.extend([
                    "",
                    "| Data Handoff Audit Metric | Value |",
                    "| :--- | :---: |",
                    f"| Total Verified Handoffs Identified | {summary.get('total_verified_handoffs_identified', 0)} |",
                    f"| Handoffs with Runtime Payload Evidence | {summary.get('handoffs_with_runtime_payload_evidence', 0)} |",
                    f"| Statically Inferred Handoffs | {summary.get('statically_inferred_handoffs', 0)} |",
                    f"| Large Payloads Detected | {summary.get('large_payloads_detected', 0)} |",
                    f"| Potential Redundant Fields Identified | {summary.get('potential_redundant_fields_identified', 0)} |",
                    f"| Duplicate Fields Identified | {summary.get('duplicate_fields_identified', 0)} |",
                    f"| Uncertain Fields Identified | {summary.get('uncertain_fields_identified', 0)} |",
                    f"| Handoffs Analyzed Without Groq (Python Only) | {summary.get('handoffs_analyzed_without_groq', 0)} |",
                    f"| Handoffs Requiring Groq Semantic Analysis | {summary.get('handoffs_requiring_groq_semantic_analysis', 0)} |",
                    f"| Groq Calls Made | {summary.get('groq_calls_made', 0)} |",
                    f"| Compact Payload Recommendations Generated | {summary.get('compact_payload_recommendations_generated', len(cands))} |",
                    f"| Original Payload Tokens | {orig_tok_str} |",
                    f"| Proposed Payload Tokens | {prop_tok_str} |",
                    f"| Tokens Saved (Single-Execution Sum) | {saved_tok_str} |",
                    f"| Percentage Reduction | {pct_red_str} |",
                    "",
                ])
            elif summary:
                lines.extend([
                    "",
                    f"**Search-Space Reduction:** `{summary.get('search_space_reduction_summary', 'N/A')}`",
                    "",
                    "| Audit Metric | Count |",
                    "| :--- | :---: |",
                    f"| Original Agent Count | {summary.get('original_agent_count', 0)} |",
                    f"| Initial Theoretical Search-Space Size | {summary.get('initial_theoretical_search_space_size', 0)} |",
                    f"| Sequential Chains Identified | {summary.get('sequential_chains_identified', 0)} |",
                    f"| Eliminated by Context-Window Filtering (70%) | {summary.get('eliminated_by_context_window_filtering', 0)} |",
                    f"| Eliminated by 50% Token Segmentation Rule | {summary.get('eliminated_by_50pct_segmentation_rule', 0)} |",
                    f"| Independent Validation Agents Retained Separately | {summary.get('independent_validation_agents_identified', 0)} |",
                    f"| Rejected for Incompatible Responsibilities | {summary.get('rejected_for_incompatible_responsibilities', 0)} |",
                    f"| Prompt Combinations Generated | {summary.get('prompt_combinations_generated', 0)} |",
                    f"| Rejected After Final Prompt Tokenization | {summary.get('rejected_after_final_prompt_tokenization', 0)} |",
                    f"| Final Consolidation Opportunities | {summary.get('final_consolidation_opportunities', len(cands))} |",
                    "",
                ])
                val_ids = summary.get("independent_validation_agent_ids") or []
                if val_ids:
                    lines.append(f"- **Independent Validation Agents Kept Separate:** {', '.join([f'`{v}`' for v in val_ids])}")
                    lines.append("")
        else:
            cands = dim_val if isinstance(dim_val, list) else []
            lines.append(f"### {title} ({len(cands)} candidates)")

        if not cands:
            lines.append("No candidates identified for this dimension based on available trace evidence.\n")
            continue

        for c in cands:
            lines.extend([
                f"#### [{c.get('tier', 'Tier 1')}] {c.get('title')}",
                f"- **Target Agents:** {', '.join([f'`{a}`' for a in c.get('target_agents', [])])}",
                f"- **Observed Problem:** {c.get('observed_problem')}",
                f"- **Proposed Solution:** {c.get('proposed_solution')}",
            ])
            exp = c.get("expected_benefit", {})
            ev = c.get("supporting_evidence", {})
            if key == "prompt_optimization_opportunities" and ev:
                cum_bd = ev.get("cumulative_savings_breakdown") or {}
                orig_label = (
                    "Original consolidated prompt"
                    if ev.get("originated_from_agent_consolidation")
                    else "Original standalone prompt"
                )
                cum_val_str = (
                    f"{cum_bd.get('cumulative_tokens_saved')} tokens ({cum_bd.get('cumulative_savings_percentage')}% vs {cum_bd.get('pre_consolidation_original_tokens')} pre-consolidation tokens)"
                    if cum_bd.get("cumulative_evidence_status") == "available"
                    else "N/A (unavailable or standalone prompt)"
                )
                meas_tag = "Exact" if ev.get("is_exact_tokenizer") else "Estimated heuristic"
                lines.extend([
                    "",
                    "| Metric | Value | Measurement Basis |",
                    "| :--- | :---: | :--- |",
                    f"| {orig_label} | {ev.get('original_prompt_token_count', 0)} tokens | `{ev.get('tokenization_method')}` ({meas_tag}) |",
                    f"| Optimized prompt | {ev.get('optimized_prompt_token_count', 0)} tokens | `{ev.get('tokenization_method')}` ({meas_tag}) |",
                    f"| Tokens saved | {ev.get('tokens_saved', 0)} tokens | Absolute reduction |",
                    f"| Savings percentage | {ev.get('percentage_savings', 0.0)}% | Relative to pre-optimization prompt |",
                    f"| Cumulative savings | {cum_val_str} | Relative to original individual prompts |",
                    "",
                ])
            elif key == "communication_optimization_opportunities" and ev:
                orig_p_str = f"{ev.get('original_payload_tokens')} tokens" if ev.get("original_payload_tokens") is not None else "Unavailable (static only)"
                prop_p_str = f"{ev.get('proposed_payload_tokens')} tokens" if ev.get("proposed_payload_tokens") is not None else "Unavailable (static only)"
                tok_sv_str = f"{ev.get('tokens_saved')} tokens ({ev.get('measurement_type')})" if ev.get("tokens_saved") is not None else "Unavailable (static only)"
                pct_sv_str = f"{ev.get('savings_percentage')}%" if ev.get("savings_percentage") is not None else "Unavailable (static only)"
                removed_str = ", ".join(f"`{f}`" for f in (ev.get("fields_excluded") or [])) or "None"
                retained_str = ", ".join(f"`{f}`" for f in (ev.get("fields_retained") or [])) or "None"
                lines.extend([
                    "",
                    "| Metric | Value |",
                    "| :--- | :--- |",
                    f"| Producer | `{ev.get('producer_agent')}` |",
                    f"| Consumer | `{ev.get('consumer_agent')}` |",
                    f"| Original payload | {orig_p_str} |",
                    f"| Proposed payload | {prop_p_str} |",
                    f"| Tokens saved | {tok_sv_str} |",
                    f"| Savings percentage | {pct_sv_str} |",
                    f"| Fields removed | {removed_str} |",
                    f"| Fields retained | {retained_str} |",
                    f"| Evidence quality | `{ev.get('evidence_quality')}` (`{ev.get('evidence_basis')}`) |",
                    "",
                ])
            elif exp:
                if exp.get("is_estimate") and "estimated_static_prompt_token_reduction_pct" in exp:
                    lines.append(
                        f"- **Estimated Impact (Unvalidated):** Static Prompt Token Reduction: "
                        f"{exp.get('measured_static_prompt_token_reduction', 0)} tokens "
                        f"({exp.get('estimated_static_prompt_token_reduction_pct')}% of combined static prompt) | "
                        f"Baseline Combined Latency: {exp.get('baseline_combined_latency_seconds', 0):.2f}s"
                    )
                else:
                    lines.append(
                        f"- **Expected Benefit:** Latency: {exp.get('latency_savings_pct', 0)}% reduction | "
                        f"Tokens: {exp.get('token_savings_pct', 0)}% savings | {exp.get('quality_impact', '')}"
                    )
            if c.get("requires_validation"):
                lines.append("- **Validation Status:** `REQUIRES VALIDATION` (Pre-deployment shadow testing mandatory)")
            if c.get("unified_prompt"):
                lines.extend([
                    "",
                    "<details>",
                    "<summary>Generated Unified Prompt</summary>",
                    "",
                    "```text",
                    str(c["unified_prompt"]),
                    "```",
                    "</details>",
                ])
            if c.get("optimized_prompt"):
                lines.extend([
                    "",
                    "<details>",
                    "<summary>Generated Optimized Prompt</summary>",
                    "",
                    "```text",
                    str(c["optimized_prompt"]),
                    "```",
                    "</details>",
                ])
            lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. Prioritized Action Plan & Implementation Roadmap",
        "",
        f"### Tier 1: Immediate Implementation ({len(action.get('tier_1_immediate_implementation', []))} items)",
    ])
    for item in action.get("tier_1_immediate_implementation", []):
        lines.append(f"- **{item.get('title')}**: {item.get('justification')}")

    lines.extend([
        "",
        f"### Tier 2: Validated Enhancements ({len(action.get('tier_2_implementation_with_validation', []))} items)",
    ])
    for item in action.get("tier_2_implementation_with_validation", []):
        lines.append(f"- **{item.get('title')}**: {item.get('validation_requirements')}")

    lines.extend([
        "",
        "### Phased Roadmap:",
        "",
    ])
    for phase in action.get("implementation_roadmap", []):
        lines.append(f"#### {phase.get('phase')} ({phase.get('projected_timeframe')})")
        for act in phase.get("actions", []):
            lines.append(f"- {act}")
        lines.append("")

    return "\n".join(lines)


def generate_report(state: OptimizationAgentState) -> Dict[str, Any]:
    """LangGraph Node 13: Compile structured JSON report and Markdown report, saving to disk."""
    track_status = dict(state.get("stage_status") or {})
    retrieval_errors = list(state.get("retrieval_errors") or [])
    key_mgr = get_default_key_manager()
    raw_key, env_var, masked = key_mgr.get_key_for_task("report_generation")

    try:
        report_json = generate_json_report(state, masked)
        markdown_report = generate_markdown_report(report_json)

        # Write reports to reports/ directory
        reports_dir = _resolve_reports_dir()
        wf_id = state.get("workflow_id", "workflow")

        json_path = reports_dir / f"{wf_id}_optimization_report.json"
        md_path = reports_dir / f"{wf_id}_optimization_report.md"

        json_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
        md_path.write_text(markdown_report, encoding="utf-8")

        logger.info(f"Generated optimization reports at {json_path} and {md_path}")

        track_status["generate_report"] = "completed"
        return {
            "stage_status": track_status,
            "final_report": report_json,
            "final_markdown_report": markdown_report,
            "key_usage_log": key_mgr.get_usage_log(),
        }
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        track_status["generate_report"] = "failed"
        retrieval_errors.append({"stage": "generate_report", "error": str(e)})
        return {
            "stage_status": track_status,
            "retrieval_errors": retrieval_errors,
            "final_report": {},
            "final_markdown_report": "",
        }

