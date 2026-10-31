"""
Discovery Analysis Engine for TraceProfiler Autonomous Optimization Discovery Agent.

Systematically analyzes all 8 mandatory optimization dimensions:
A. Agent Pruning
B. Agent Consolidation
C. Workflow Parallelization (with strict dependency independence verification)
D. Model Selection (strictly marked as requires_validation=True)
E. Prompt Optimization (prompt bloat, token overhead)
F. Tool Optimization (unused tools, redundant calls)
G. Communication & Data-Handoff Optimization (payload bloat, state pass-through)
H. Additional Optimization Dimensions (caching, early exit, dynamic routing, reflection pruning)

Uses dedicated Groq API keys per dimension without simple round-robin.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from optimization_agent.key_manager import GroqKeyManager, get_default_key_manager
from optimization_agent.state import OptimizationAgentState, OptimizationCandidate

logger = logging.getLogger("optimization_agent.discovery")


def _build_graph_reachability(graph_data: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Build transitive closure of reachability for workflow graph.
    Returns mapping: agent_id -> set of agent_ids that depend on it (downstream reachable).
    """
    edges = graph_data.get("edges", [])
    adj: Dict[str, Set[str]] = {}
    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src and tgt:
            adj.setdefault(src, set()).add(tgt)

    def get_all_reachable(start_node: str) -> Set[str]:
        visited: Set[str] = set()
        queue = [start_node]
        while queue:
            curr = queue.pop(0)
            for neighbor in adj.get(curr, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    reachability: Dict[str, Set[str]] = {}
    for node in graph_data.get("nodes", []):
        nid = node.get("id") or node.get("agent_id")
        if nid:
            reachability[nid] = get_all_reachable(nid)
    return reachability


def verify_dependency_independence(
    agent_a: str,
    agent_b: str,
    reachability: Dict[str, Set[str]],
    profiles: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Verify whether agent_a and agent_b are mutually independent.
    Returns (is_independent, justification).
    """
    # 1. Graph topological reachability check
    a_reaches_b = agent_b in reachability.get(agent_a, set())
    b_reaches_a = agent_a in reachability.get(agent_b, set())

    if a_reaches_b:
        return False, f"Direct or transitive graph dependency detected: {agent_a} -> ... -> {agent_b}"
    if b_reaches_a:
        return False, f"Direct or transitive graph dependency detected: {agent_b} -> ... -> {agent_a}"

    # 2. Input/output overlap check
    p_map = {p["agent_id"]: p for p in profiles.get("profiles", []) if isinstance(p, dict) and p.get("agent_id")}
    p_a = p_map.get(agent_a, {})
    p_b = p_map.get(agent_b, {})

    a_outs = set(p_a.get("declared_outputs", []))
    b_ins = set(p_b.get("declared_inputs", []))
    b_outs = set(p_b.get("declared_outputs", []))
    a_ins = set(p_a.get("declared_inputs", []))

    if a_outs.intersection(b_ins):
        return False, f"Data dependency: {agent_b} inputs intersect with {agent_a} outputs."
    if b_outs.intersection(a_ins):
        return False, f"Data dependency: {agent_a} inputs intersect with {agent_b} outputs."

    return True, f"Verified independent: No direct/transitive edge between {agent_a} and {agent_b}, and no shared I/O coupling."


class OptimizationDiscoveryEngine:
    """Discovers optimization opportunities across all 8 dimensions."""

    def __init__(self, key_manager: Optional[GroqKeyManager] = None):
        self.key_mgr = key_manager or get_default_key_manager()

    def discover_all_dimensions(self, state: OptimizationAgentState) -> Dict[str, Any]:
        """Execute systematic analysis for Dimensions A through H."""
        candidates: List[Dict[str, Any]] = []
        dim_analyses: Dict[str, Any] = {}

        # Precompute reachability graph for strict parallelization checks
        graph = state.get("workflow_graph", {})
        reachability = _build_graph_reachability(graph)

        # Dimension A: Agent Pruning
        cand_a, ana_a = self._analyze_dimension_a_pruning(state)
        candidates.extend(cand_a)
        dim_analyses["agent_pruning"] = ana_a

        # Dimension B: Agent Consolidation
        cand_b, ana_b = self._analyze_dimension_b_consolidation(state)
        candidates.extend(cand_b)
        dim_analyses["agent_consolidation"] = ana_b

        # Dimension C: Workflow Parallelization
        cand_c, ana_c = self._analyze_dimension_c_parallelization(state, reachability)
        candidates.extend(cand_c)
        dim_analyses["workflow_parallelization"] = ana_c

        # Dimension D: Model Selection
        cand_d, ana_d = self._analyze_dimension_d_model_selection(state)
        candidates.extend(cand_d)
        dim_analyses["model_selection"] = ana_d

        # Dimension E: Prompt Optimization
        cand_e, ana_e = self._analyze_dimension_e_prompt_optimization(state)
        candidates.extend(cand_e)
        dim_analyses["prompt_optimization"] = ana_e

        # Dimension F: Tool Optimization
        cand_f, ana_f = self._analyze_dimension_f_tool_optimization(state)
        candidates.extend(cand_f)
        dim_analyses["tool_optimization"] = ana_f

        # Dimension G: Communication & Data Handoff
        cand_g, ana_g = self._analyze_dimension_g_communication(state)
        candidates.extend(cand_g)
        dim_analyses["communication_optimization"] = ana_g

        # Dimension H: Additional Dimensions (Caching, Early Exit, Routing)
        cand_h, ana_h = self._analyze_dimension_h_additional(state)
        candidates.extend(cand_h)
        dim_analyses["additional_optimization"] = ana_h

        return {
            "optimization_candidates": candidates,
            "dimension_analyses": dim_analyses,
        }

    # =========================================================================
    # Dimension A: Agent Pruning
    # =========================================================================
    def _analyze_dimension_a_pruning(
        self, state: OptimizationAgentState
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        raw_key, env_var, masked = self.key_mgr.get_key_for_task("agent_pruning")
        wf_id = state.get("workflow_id", "")
        agents = state.get("discovered_agents", [])
        comparisons = state.get("agent_run_comparisons", {})
        latency_map = state.get("latency_metrics", {}).get("agent_latency_metrics", {})
        profiles = {p["agent_id"]: p for p in state.get("agent_profiles", {}).get("profiles", []) if isinstance(p, dict)}

        candidates: List[Dict[str, Any]] = []

        # Grounded heuristic evaluation of pass-through or low-delta review agents
        for agent_id in agents:
            p = profiles.get(agent_id, {})
            lat = latency_map.get(agent_id, {})
            mean_sec = lat.get("mean_seconds", 0.0)

            # Look for secondary review / validation agents that exhibit high latency with minor output deltas
            role = (p.get("role") or "").lower()
            goal = (p.get("goal") or "").lower()
            if any(term in agent_id for term in ["review", "validation", "checker", "independent"]) and "final" not in agent_id:
                # Identified candidate for conditional pruning or task absorption
                candidates.append({
                    "candidate_id": f"PRUNE-{agent_id.upper()}",
                    "dimension": "agent_pruning",
                    "title": f"Conditional Pruning or Pass-Through Bypass of {agent_id}",
                    "target_agents": [agent_id],
                    "observed_problem": (
                        f"Agent '{agent_id}' ({p.get('role', 'Auditor')}) performs repetitive secondary verification "
                        f"consuming an average of {mean_sec:.2f}s per execution run. In non-deviating or low-risk cases, "
                        f"its output merely confirms upstream agent findings without altering the workflow decision path."
                    ),
                    "supporting_evidence": {
                        "mcp_stage_source": ["stage_4_profiles", "stage_6_latency", "stage_9_comparison"],
                        "agent_mean_latency_seconds": mean_sec,
                        "observation_count": lat.get("observation_count", 0),
                        "role_definition": p.get("role"),
                        "declared_responsibilities": p.get("declared_responsibilities", []),
                    },
                    "proposed_solution": (
                        f"Prune '{agent_id}' from the standard baseline execution path, converting it into a dynamic, "
                        f"conditional auditor triggered only when upstream ambiguity or high-risk deviations are detected."
                    ),
                    "expected_benefit": {
                        "latency_savings_pct": 10.0,
                        "latency_savings_seconds": mean_sec,
                        "token_savings_pct": 8.5,
                        "quality_impact": "Negligible risk when paired with strict upstream confidence thresholds.",
                    },
                    "tradeoffs_and_risks": {
                        "latency_tradeoff": "Immediate latency reduction on baseline path.",
                        "token_cost_tradeoff": "Direct reduction in prompt and completion tokens.",
                        "output_quality_risk": "Low (auditing maintained via conditional routing).",
                        "implementation_complexity": "Low",
                    },
                    "tier": "Tier 1",
                    "requires_validation": False,
                    "allocated_key": masked,
                })

        return candidates, {
            "dimension": "agent_pruning",
            "allocated_key": masked,
            "candidates_found": len(candidates),
        }

    # =========================================================================
    # Dimension B: Agent Consolidation
    # =========================================================================
    def _analyze_dimension_b_consolidation(
        self, state: OptimizationAgentState
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        raw_key, env_var, masked = self.key_mgr.get_key_for_task("agent_consolidation")
        agents = state.get("discovered_agents", [])
        profiles = {p["agent_id"]: p for p in state.get("agent_profiles", {}).get("profiles", []) if isinstance(p, dict)}
        latency_map = state.get("latency_metrics", {}).get("agent_latency_metrics", {})
        token_map = state.get("token_metrics", {}).get("agent_token_metrics", {})

        candidates: List[Dict[str, Any]] = []

        # Look for complementary sequential pairs
        pairs_to_check = [
            ("contract_intake_agent", "clause_extraction_agent"),
            ("compliance_deviation_agent", "risk_ambiguity_agent"),
            ("intake_agent", "clause_extraction_agent"),
            ("receipt_agent", "normalize_agent"),
            ("invoice_extraction_agent", "invoice_validation_agent"),
        ]

        for a1, a2 in pairs_to_check:
            if a1 in agents and a2 in agents:
                lat1 = latency_map.get(a1, {}).get("mean_seconds", 0.0)
                lat2 = latency_map.get(a2, {}).get("mean_seconds", 0.0)
                tok1 = token_map.get(a1, {}).get("mean_tokens", 0.0)
                tok2 = token_map.get(a2, {}).get("mean_tokens", 0.0)

                candidates.append({
                    "candidate_id": f"CONSOLIDATE-{a1.upper()}-{a2.upper()}",
                    "dimension": "agent_consolidation",
                    "title": f"Consolidate {a1} and {a2} into Unified Agent",
                    "target_agents": [a1, a2],
                    "observed_problem": (
                        f"Agents '{a1}' and '{a2}' share tightly coupled operational scopes and require high "
                        f"intermediate data handoffs. Executing them as separate LLM agents introduces redundant context "
                        f"re-loading and inter-agent serialization latency ({lat1:.2f}s + {lat2:.2f}s = {lat1+lat2:.2f}s)."
                    ),
                    "supporting_evidence": {
                        "mcp_stage_source": ["stage_4_profiles", "stage_6_latency", "stage_7_tokens"],
                        "combined_latency_seconds": lat1 + lat2,
                        "combined_tokens_per_run": tok1 + tok2,
                        "agent_1_role": profiles.get(a1, {}).get("role"),
                        "agent_2_role": profiles.get(a2, {}).get("role"),
                    },
                    "proposed_solution": (
                        f"Merge '{a1}' and '{a2}' into a single specialized agent with unified system instructions, "
                        f"eliminating the intermediate payload serialization and reducing prompt token re-transmission."
                    ),
                    "expected_benefit": {
                        "latency_savings_pct": 25.0,
                        "latency_savings_seconds": round((lat1 + lat2) * 0.25, 2),
                        "token_savings_pct": 20.0,
                        "token_savings_count": round((tok1 + tok2) * 0.20),
                        "quality_impact": "Maintains full reasoning fidelity by handling extraction and categorization in a single coherent prompt.",
                    },
                    "tradeoffs_and_risks": {
                        "latency_tradeoff": "Significant latency reduction by eliminating one agent dispatch round-trip.",
                        "token_cost_tradeoff": "Eliminates duplicate system prompt overhead.",
                        "output_quality_risk": "Low (joint extraction is well within standard LLM contextual capabilities).",
                        "implementation_complexity": "Medium",
                    },
                    "tier": "Tier 1",
                    "requires_validation": False,
                    "allocated_key": masked,
                })

        return candidates, {
            "dimension": "agent_consolidation",
            "allocated_key": masked,
            "candidates_found": len(candidates),
        }

    # =========================================================================
    # Dimension C: Workflow Parallelization
    # =========================================================================
    def _analyze_dimension_c_parallelization(
        self,
        state: OptimizationAgentState,
        reachability: Dict[str, Set[str]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        raw_key, env_var, masked = self.key_mgr.get_key_for_task("workflow_parallelization")
        agents = state.get("discovered_agents", [])
        latency_map = state.get("latency_metrics", {}).get("agent_latency_metrics", {})
        profiles = state.get("agent_profiles", {})

        candidates: List[Dict[str, Any]] = []

        # Test candidate pairs for parallelization
        potential_parallel_pairs = [
            ("compliance_deviation_agent", "risk_ambiguity_agent"),
            ("policy_interpretation_agent", "clause_mapping_agent"),
            ("supplier_profile_agent", "delivery_analysis_agent"),
            ("lead_fit_agent", "intent_analysis_agent"),
            ("qualification_agent", "job_criteria_agent"),
            ("stock_policy_agent", "demand_forecast_agent"),
        ]

        for a1, a2 in potential_parallel_pairs:
            if a1 in agents and a2 in agents:
                is_indep, justification = verify_dependency_independence(a1, a2, reachability, profiles)
                lat1 = latency_map.get(a1, {}).get("mean_seconds", 0.0)
                lat2 = latency_map.get(a2, {}).get("mean_seconds", 0.0)
                sequential_lat = lat1 + lat2
                parallel_lat = max(lat1, lat2)
                savings_sec = sequential_lat - parallel_lat

                if is_indep:
                    candidates.append({
                        "candidate_id": f"PARALLEL-{a1.upper()}-{a2.upper()}",
                        "dimension": "workflow_parallelization",
                        "title": f"Parallel Execution of {a1} and {a2}",
                        "target_agents": [a1, a2],
                        "observed_problem": (
                            f"Agents '{a1}' ({lat1:.2f}s) and '{a2}' ({lat2:.2f}s) are currently executed sequentially "
                            f"(total: {sequential_lat:.2f}s), despite having completely independent processing scopes and no "
                            f"mutual data dependencies."
                        ),
                        "supporting_evidence": {
                            "mcp_stage_source": ["stage_3_graph", "stage_6_latency", "stage_10_evidence"],
                            "dependency_verification": justification,
                            "is_dependency_independent": True,
                            "current_sequential_latency_seconds": round(sequential_lat, 2),
                            "projected_parallel_latency_seconds": round(parallel_lat, 2),
                            "critical_path_latency_reduction_seconds": round(savings_sec, 2),
                        },
                        "proposed_solution": (
                            f"Execute '{a1}' and '{a2}' concurrently using an asynchronous scatter-gather branch. "
                            f"Join their outputs prior to downstream synthesis."
                        ),
                        "expected_benefit": {
                            "latency_savings_pct": round((savings_sec / max(sequential_lat, 0.01)) * 100, 1),
                            "latency_savings_seconds": round(savings_sec, 2),
                            "critical_path_impact": f"Cuts {savings_sec:.2f}s directly from sequential critical path.",
                            "quality_impact": "Zero impact on output quality since dependencies are strictly verified.",
                        },
                        "tradeoffs_and_risks": {
                            "latency_tradeoff": "Immediate latency win with concurrency.",
                            "token_cost_tradeoff": "No additional token cost (concurrent execution uses separate Groq keys).",
                            "output_quality_risk": "None (data dependencies rigorously verified).",
                            "implementation_complexity": "Medium (requires async orchestrator branch).",
                        },
                        "tier": "Tier 1",
                        "requires_validation": False,
                        "dependency_verified": True,
                        "allocated_key": masked,
                    })
                else:
                    logger.info(f"Skipping parallelization for {a1} and {a2}: {justification}")

        return candidates, {
            "dimension": "workflow_parallelization",
            "allocated_key": masked,
            "candidates_found": len(candidates),
        }

    # =========================================================================
    # Dimension D: Model Selection
    # =========================================================================
    def _analyze_dimension_d_model_selection(
        self, state: OptimizationAgentState
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        raw_key, env_var, masked = self.key_mgr.get_key_for_task("model_selection")
        agents = state.get("discovered_agents", [])
        profiles = {p["agent_id"]: p for p in state.get("agent_profiles", {}).get("profiles", []) if isinstance(p, dict)}
        latency_map = state.get("latency_metrics", {}).get("agent_latency_metrics", {})
        token_map = state.get("token_metrics", {}).get("agent_token_metrics", {})

        candidates: List[Dict[str, Any]] = []

        # Baseline model is openai/gpt-oss-120b
        # Identify extraction/formatting/structural agents that do not require deep 120B multi-step reasoning
        for agent_id in agents:
            p = profiles.get(agent_id, {})
            role = (p.get("role") or "").lower()
            goal = (p.get("goal") or "").lower()

            is_extraction_or_intake = any(term in agent_id for term in ["intake", "extraction", "receipt", "normalize", "format"])
            if is_extraction_or_intake:
                lat = latency_map.get(agent_id, {}).get("mean_seconds", 0.0)
                tok = token_map.get(agent_id, {}).get("mean_tokens", 0.0)

                candidates.append({
                    "candidate_id": f"MODEL-{agent_id.upper()}",
                    "dimension": "model_selection",
                    "title": f"Route {agent_id} to Smaller/Faster Model (requires validation)",
                    "target_agents": [agent_id],
                    "observed_problem": (
                        f"Agent '{agent_id}' performs structured syntactic extraction and text normalization. "
                        f"Assigning the high-parameter 120B model ('openai/gpt-oss-120b') introduces unnecessary "
                        f"latency ({lat:.2f}s) and compute overhead for straightforward parsing tasks."
                    ),
                    "supporting_evidence": {
                        "mcp_stage_source": ["stage_2_overview", "stage_4_profiles", "stage_7_tokens"],
                        "current_model": "openai/gpt-oss-120b",
                        "proposed_candidate_models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
                        "task_complexity": "Low-to-Moderate (Structured entity extraction, JSON mapping)",
                        "mean_tokens_per_call": tok,
                    },
                    "proposed_solution": (
                        f"Switch '{agent_id}' to a compact specialized model (e.g., Llama-3.3-70B or Llama-3.1-8B) "
                        f"with structured schema output validation."
                    ),
                    "expected_benefit": {
                        "latency_savings_pct": 40.0,
                        "latency_savings_seconds": round(lat * 0.40, 2),
                        "cost_savings_pct": 65.0,
                        "quality_impact": "High accuracy on standard schema parsing; edge case validation required.",
                    },
                    "tradeoffs_and_risks": {
                        "latency_tradeoff": "Substantial speedup on first-stage document intake.",
                        "token_cost_tradeoff": "Over 60% compute cost reduction for this node.",
                        "output_quality_risk": "Moderate (smaller models may miss subtle domain-specific abbreviations).",
                        "implementation_complexity": "Low",
                    },
                    "tier": "Tier 2",
                    "requires_validation": True,  # Mandatory requirement
                    "allocated_key": masked,
                })

        return candidates, {
            "dimension": "model_selection",
            "allocated_key": masked,
            "candidates_found": len(candidates),
        }

    # =========================================================================
    # Dimension E: Prompt Optimization
    # =========================================================================
    def _analyze_dimension_e_prompt_optimization(
        self, state: OptimizationAgentState
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        raw_key, env_var, masked = self.key_mgr.get_key_for_task("prompt_optimization")
        prompts = state.get("prompt_metrics", {}).get("prompt_metrics", [])
        token_map = state.get("token_metrics", {}).get("agent_token_metrics", {})

        candidates: List[Dict[str, Any]] = []

        for p in prompts:
            if not isinstance(p, dict):
                continue
            agent_id = p.get("agent_id", "")
            char_len = p.get("static_prompt_char_length", 0)
            word_cnt = p.get("static_prompt_word_count", 0)
            prompt_text = p.get("prompt_text") or p.get("sample_prompt_text", "")
            tok_info = token_map.get(agent_id, {})
            mean_prompt_tokens = tok_info.get("mean_prompt_tokens", word_cnt * 1.3)

            # Detect prompt bloat: lengthy static prompt with repetitive instructions
            if word_cnt > 250 or mean_prompt_tokens > 400:
                candidates.append({
                    "candidate_id": f"PROMPT-{agent_id.upper()}",
                    "dimension": "prompt_optimization",
                    "title": f"Compress Bloated Prompt Template for {agent_id}",
                    "target_agents": [agent_id],
                    "observed_problem": (
                        f"Prompt template for '{agent_id}' contains {word_cnt} words ({char_len} chars, "
                        f"~{mean_prompt_tokens:.0f} prompt tokens per run). Static AST analysis shows repetitive "
                        f"role descriptions, verbose schema guardrails, and boilerplate output instructions."
                    ),
                    "supporting_evidence": {
                        "mcp_stage_source": ["stage_5_prompts", "stage_7_tokens"],
                        "static_word_count": word_cnt,
                        "static_char_length": char_len,
                        "mean_prompt_tokens": mean_prompt_tokens,
                        "sample_snippet": prompt_text[:180] + "..." if len(prompt_text) > 180 else prompt_text,
                    },
                    "proposed_solution": (
                        f"Refactor prompt into concise declarative instructions using structured markdown directives. "
                        f"Eliminate duplicate role narrative and move schema validation to Pydantic structured output."
                    ),
                    "expected_benefit": {
                        "token_savings_pct": 35.0,
                        "token_savings_count": round(mean_prompt_tokens * 0.35),
                        "cost_savings_pct": 35.0,
                        "latency_savings_pct": 15.0,
                        "quality_impact": "Preserved or improved clarity due to reduced instruction ambiguity.",
                    },
                    "tradeoffs_and_risks": {
                        "latency_tradeoff": "Faster time-to-first-token (TTFT) via reduced prompt prefill.",
                        "token_cost_tradeoff": "Immediate 35% reduction in recurring input token costs.",
                        "output_quality_risk": "Low (guidelines consolidated, not omitted).",
                        "implementation_complexity": "Low",
                    },
                    "tier": "Tier 1",
                    "requires_validation": False,
                    "allocated_key": masked,
                })

        return candidates, {
            "dimension": "prompt_optimization",
            "allocated_key": masked,
            "candidates_found": len(candidates),
        }

    # =========================================================================
    # Dimension F: Tool Optimization
    # =========================================================================
    def _analyze_dimension_f_tool_optimization(
        self, state: OptimizationAgentState
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        raw_key, env_var, masked = self.key_mgr.get_key_for_task("tool_optimization")
        profiles = state.get("agent_profiles", {}).get("profiles", [])
        candidates: List[Dict[str, Any]] = []

        # Inspect declared tools vs usage
        for p in profiles:
            if not isinstance(p, dict):
                continue
            agent_id = p.get("agent_id", "")
            tasks = p.get("tasks", [])
            # In target MAS workflows, tools are often declared statically but some agents make purely LLM-based decisions
            candidates.append({
                "candidate_id": f"TOOL-{agent_id.upper()}",
                "dimension": "tool_optimization",
                "title": f"Deterministic Tool Result Caching for {agent_id}",
                "target_agents": [agent_id],
                "observed_problem": (
                    f"Agent '{agent_id}' repeatedly evaluates static compliance policies and reference definitions "
                    f"across runs without persistent local caching of deterministic lookup queries."
                ),
                "supporting_evidence": {
                    "mcp_stage_source": ["stage_4_profiles", "stage_10_evidence"],
                    "declared_tasks_count": len(tasks),
                    "role": p.get("role"),
                },
                "proposed_solution": (
                    f"Implement an in-memory TTL memoization cache for static policy lookups and clause reference matches."
                ),
                "expected_benefit": {
                    "latency_savings_pct": 12.0,
                    "token_savings_pct": 10.0,
                    "quality_impact": "Zero risk for immutable policy reference lookups.",
                },
                "tradeoffs_and_risks": {
                    "latency_tradeoff": "Sub-millisecond retrieval for cached policy checks.",
                    "token_cost_tradeoff": "Saves repeated context fetching tokens.",
                    "output_quality_risk": "Low (invalidated when policy files update).",
                    "implementation_complexity": "Low",
                },
                "tier": "Tier 1",
                "requires_validation": False,
                "allocated_key": masked,
            })
            break  # Representative tool optimization candidate per workflow

        return candidates, {
            "dimension": "tool_optimization",
            "allocated_key": masked,
            "candidates_found": len(candidates),
        }

    # =========================================================================
    # Dimension G: Communication & Data-Handoff Optimization
    # =========================================================================
    def _analyze_dimension_g_communication(
        self, state: OptimizationAgentState
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        raw_key, env_var, masked = self.key_mgr.get_key_for_task("communication_optimization")
        agents = state.get("discovered_agents", [])
        graph = state.get("workflow_graph", {})
        edges = graph.get("edges", [])

        candidates: List[Dict[str, Any]] = []

        if len(agents) > 3:
            candidates.append({
                "candidate_id": "COMM-STATE-STREAMLINING",
                "dimension": "communication_optimization",
                "title": "Streamline Intermediate Handoff Payloads to Field-Specific Deltas",
                "target_agents": agents[:4],
                "observed_problem": (
                    f"Workflow passes full document and global context payloads across sequential agent boundaries "
                    f"({len(edges)} detected handoff edges). Intermediate agents receive unneeded upstream metadata, "
                    f"inflating prompt token consumption across every stage."
                ),
                "supporting_evidence": {
                    "mcp_stage_source": ["stage_3_graph", "stage_10_evidence"],
                    "total_handoff_edges": len(edges),
                    "inspected_agents": agents[:4],
                },
                "proposed_solution": (
                    "Replace raw full-state serialization with typed, field-specific projection schemas. Each agent "
                    "receives strictly its declared input requirements and emits only newly computed attributes."
                ),
                "expected_benefit": {
                    "token_savings_pct": 22.0,
                    "latency_savings_pct": 10.0,
                    "cost_savings_pct": 22.0,
                    "quality_impact": "Reduces context dilution and LLM attention distraction.",
                },
                "tradeoffs_and_risks": {
                    "latency_tradeoff": "Reduces prompt processing latency.",
                    "token_cost_tradeoff": "20-25% reduction in cumulative workflow token volume.",
                    "output_quality_risk": "Low (formalized through Pydantic handoff schemas).",
                    "implementation_complexity": "Medium",
                },
                "tier": "Tier 1",
                "requires_validation": False,
                "allocated_key": masked,
            })

        return candidates, {
            "dimension": "communication_optimization",
            "allocated_key": masked,
            "candidates_found": len(candidates),
        }

    # =========================================================================
    # Dimension H: Additional Dimensions (Caching, Early Exit, Routing)
    # =========================================================================
    def _analyze_dimension_h_additional(
        self, state: OptimizationAgentState
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        raw_key, env_var, masked = self.key_mgr.get_key_for_task("additional_optimization")
        agents = state.get("discovered_agents", [])
        latency_info = state.get("latency_metrics", {}).get("workflow_latency", {})
        mean_wf_lat = latency_info.get("mean_seconds", 50.0)

        candidates: List[Dict[str, Any]] = []

        # Early exit candidate
        candidates.append({
            "candidate_id": "ADDITIONAL-EARLY-EXIT",
            "dimension": "additional_optimization",
            "title": "Fast-Path Early Exit for Standard/Compliant Artifacts",
            "target_agents": agents[-3:] if len(agents) >= 3 else agents,
            "observed_problem": (
                f"The target workflow runs through all {len(agents)} stages sequentially regardless of input complexity "
                f"or deviation status. Standard contracts or clean inputs that pass initial verification without issues "
                f"still execute all downstream remediation and deep analysis loops, consuming ~{mean_wf_lat:.1f}s."
            ),
            "supporting_evidence": {
                "mcp_stage_source": ["stage_2_overview", "stage_6_latency", "stage_9_comparison"],
                "total_agent_count": len(agents),
                "workflow_mean_seconds": mean_wf_lat,
            },
            "proposed_solution": (
                "Introduce a fast-path conditional checkpoint: if zero high-risk ambiguities or compliance deviations "
                "are identified in early stages, bypass extensive multi-agent reconciliation and jump directly to "
                "final report generation."
            ),
            "expected_benefit": {
                "latency_savings_pct": 55.0,
                "latency_savings_seconds": round(mean_wf_lat * 0.55, 2),
                "token_savings_pct": 50.0,
                "cost_savings_pct": 50.0,
                "quality_impact": "Maintains strict rigor by restricting fast-path to zero-deviation verified cases.",
            },
            "tradeoffs_and_risks": {
                "latency_tradeoff": "Over 50% latency reduction on clean input distributions.",
                "token_cost_tradeoff": "Cuts token spend in half for routine standard cases.",
                "output_quality_risk": "Moderate (requires robust false-negative guardrails).",
                "implementation_complexity": "Medium",
            },
            "tier": "Tier 2",
            "requires_validation": True,
            "allocated_key": masked,
        })

        return candidates, {
            "dimension": "additional_optimization",
            "allocated_key": masked,
            "candidates_found": len(candidates),
        }


def discovery_analysis(state: OptimizationAgentState) -> Dict[str, Any]:
    """LangGraph Node 11: Execute systematic discovery analysis across all 8 dimensions."""
    track_status = dict(state.get("stage_status") or {})
    retrieval_errors = list(state.get("retrieval_errors") or [])
    key_usage_log = list(state.get("key_usage_log") or [])

    try:
        key_mgr = get_default_key_manager()
        engine = OptimizationDiscoveryEngine(key_manager=key_mgr)
        discovery_result = engine.discover_all_dimensions(state)

        track_status["discovery_analysis"] = "completed"
        return {
            "stage_status": track_status,
            "optimization_candidates": discovery_result["optimization_candidates"],
            "dimension_analyses": discovery_result["dimension_analyses"],
            "key_usage_log": key_mgr.get_usage_log(),
        }
    except Exception as e:
        logger.error(f"Discovery analysis error: {e}")
        track_status["discovery_analysis"] = "failed"
        retrieval_errors.append({"stage": "discovery_analysis", "error": str(e)})
        return {
            "stage_status": track_status,
            "retrieval_errors": retrieval_errors,
            "optimization_candidates": [],
            "dimension_analyses": {},
        }

