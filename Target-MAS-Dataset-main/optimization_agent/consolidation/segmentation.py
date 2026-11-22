"""
Stage 2: Dynamic Sequential-Chain Extraction & 50% Cumulative Token Segmentation.

Identifies verified strictly sequential execution chains from declared DAG edges
(using observed execution order only as supporting validation, never as sole proof)
and segments each chain using the recursive 50% cumulative token contribution rule.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

GENERIC_INPUT_MARKERS = {"all_previous_stages", "previous_stage", "input", "context", ""}


def _normalize_agent_id(raw_id: Any) -> str:
    """Normalize MCP node identifiers by stripping 'agent:' prefixes."""
    if not isinstance(raw_id, str):
        return ""
    val = raw_id.strip()
    if val.startswith("agent:"):
        val = val[len("agent:") :].strip()
    elif val.startswith("workflow:") or val.startswith("task:"):
        return ""
    return val


def _has_directed_path(
    start: str,
    target: str,
    edges: Set[Tuple[str, str]],
) -> bool:
    """BFS reachability check used for transitive reduction."""
    adj: Dict[str, List[str]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)

    visited: Set[str] = {start}
    queue: List[str] = [start]
    while queue:
        curr = queue.pop(0)
        for nxt in adj.get(curr, []):
            if nxt == target:
                return True
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return False


def _compute_observed_ranks(
    observed_edges: Set[Tuple[str, str]],
    execution_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """Derive deterministic observed execution order ranks from telemetry."""
    ranks: Dict[str, int] = {}

    if isinstance(execution_evidence, dict):
        for aid, ev in execution_evidence.items():
            if not isinstance(ev, dict):
                continue
            orders = ev.get("observed_execution_orders")
            if isinstance(orders, list):
                int_orders = [int(o) for o in orders if isinstance(o, (int, float)) and not isinstance(o, bool)]
                if int_orders:
                    ranks[aid] = min(int_orders)

    if not ranks and observed_edges:
        in_deg: Dict[str, int] = {}
        adj: Dict[str, List[str]] = {}
        nodes: Set[str] = set()
        for u, v in observed_edges:
            nodes.update([u, v])
            adj.setdefault(u, []).append(v)
            in_deg[v] = in_deg.get(v, 0) + 1
            in_deg.setdefault(u, 0)

        queue = sorted([n for n in nodes if in_deg.get(n, 0) == 0])
        step = 1
        while queue:
            curr = queue.pop(0)
            if curr not in ranks:
                ranks[curr] = step
                step += 1
            for nxt in sorted(adj.get(curr, [])):
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)
                    queue.sort()

    return ranks


def extract_sequential_chains(
    workflow_graph: Dict[str, Any],
    discovered_agents: List[str],
    execution_evidence: Optional[Dict[str, Any]] = None,
    agent_profiles: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract verified strictly sequential agent chains from the workflow DAG.
    Never infers sequential dependency purely from consecutive observed execution order.
    """
    valid_agents: Set[str] = set(discovered_agents or [])
    edges_raw = (workflow_graph or {}).get("edges", []) or []
    warnings: List[str] = []

    declared_edges: Set[Tuple[str, str]] = set()
    observed_edges: Set[Tuple[str, str]] = set()
    malformed_declared_count = 0

    for edge in edges_raw:
        if not isinstance(edge, dict):
            continue
        rel = edge.get("relationship")
        u = _normalize_agent_id(edge.get("source"))
        v = _normalize_agent_id(edge.get("target"))

        if rel in ("declared_dependency", None):
            if u in valid_agents and v in valid_agents and u != v:
                declared_edges.add((u, v))
            elif rel == "declared_dependency" and (not u or not v):
                malformed_declared_count += 1
        elif rel == "observed_execution_order":
            if u in valid_agents and v in valid_agents and u != v:
                observed_edges.add((u, v))

    if malformed_declared_count:
        warnings.append(
            f"Ignored {malformed_declared_count} malformed declared_dependency edge(s) missing source/target agent IDs."
        )

    obs_rank = _compute_observed_ranks(observed_edges, execution_evidence)

    # Fallback ONLY to explicit agent I/O producer->consumer contracts (never observed order alone)
    if not declared_edges and agent_profiles:
        profiles = [
            p for p in (agent_profiles.get("profiles") or [])
            if isinstance(p, dict) and p.get("agent_id") in valid_agents
        ]
        producers_by_key: Dict[str, Set[str]] = {}
        for p in profiles:
            aid = p["agent_id"]
            for out_k in (p.get("declared_outputs") or []):
                if isinstance(out_k, str):
                    norm_k = out_k.strip().lower()
                    if norm_k and norm_k not in GENERIC_INPUT_MARKERS:
                        producers_by_key.setdefault(norm_k, set()).add(aid)

        for p in profiles:
            consumer = p["agent_id"]
            for in_k in (p.get("declared_inputs") or []):
                if isinstance(in_k, str):
                    norm_k = in_k.strip().lower()
                    if not norm_k or norm_k in GENERIC_INPUT_MARKERS:
                        continue
                    for producer in producers_by_key.get(norm_k, set()):
                        if producer == consumer:
                            continue
                        if obs_rank and producer in obs_rank and consumer in obs_rank:
                            if obs_rank[producer] >= obs_rank[consumer]:
                                continue
                        declared_edges.add((producer, consumer))

    if not declared_edges:
        warnings.append(
            "No verified declared dependency edges available; observed execution order alone is insufficient to establish sequential dependency."
        )
        return {
            "chains": [],
            "declared_edges": [],
            "transitive_reduced_edges": [],
            "observed_order_ranks": obs_rank,
            "dep_parents_map": {a: set() for a in valid_agents},
            "dep_children_map": {a: set() for a in valid_agents},
            "dependency_evidence_status": "insufficient_dependency_evidence",
            "warnings": warnings,
        }

    dep_parents_map: Dict[str, Set[str]] = {a: set() for a in valid_agents}
    dep_children_map: Dict[str, Set[str]] = {a: set() for a in valid_agents}
    for u, v in declared_edges:
        dep_children_map.setdefault(u, set()).add(v)
        dep_parents_map.setdefault(v, set()).add(u)

    # Transitive reduction to uncover direct sequential backbone links
    reduced_edges: Set[Tuple[str, str]] = set()
    for edge in sorted(declared_edges):
        u, v = edge
        remaining = declared_edges - {edge}
        if not _has_directed_path(u, v, remaining):
            reduced_edges.add(edge)

    in_deg: Dict[str, int] = {a: 0 for a in valid_agents}
    out_deg: Dict[str, int] = {a: 0 for a in valid_agents}
    for u, v in reduced_edges:
        out_deg[u] = out_deg.get(u, 0) + 1
        in_deg[v] = in_deg.get(v, 0) + 1

    seq_next: Dict[str, str] = {}
    seq_prev: Dict[str, str] = {}
    for u, v in sorted(reduced_edges):
        if out_deg.get(u, 0) == 1 and in_deg.get(v, 0) == 1:
            if u in obs_rank and v in obs_rank and obs_rank[u] > obs_rank[v]:
                warnings.append(
                    f"Skipped sequential edge {u} -> {v} because it contradicts observed execution order ({obs_rank[u]} > {obs_rank[v]})."
                )
                continue
            seq_next[u] = v
            seq_prev[v] = u

    # Stitch maximal contiguous sequential chains of length >= 2
    starts = sorted(
        [u for u in seq_next if u not in seq_prev],
        key=lambda a: (obs_rank.get(a, 10**9), a),
    )
    chains: List[List[str]] = []
    visited_in_chains: Set[str] = set()

    for start in starts:
        if start in visited_in_chains:
            continue
        chain = [start]
        visited_in_chains.add(start)
        curr = start
        while curr in seq_next:
            nxt = seq_next[curr]
            if nxt in visited_in_chains:
                break
            chain.append(nxt)
            visited_in_chains.add(nxt)
            curr = nxt
        if len(chain) >= 2:
            chains.append(chain)

    chains.sort(key=lambda ch: (obs_rank.get(ch[0], 10**9), ch[0]))

    return {
        "chains": chains,
        "declared_edges": sorted(declared_edges),
        "transitive_reduced_edges": sorted(reduced_edges),
        "observed_order_ranks": obs_rank,
        "dep_parents_map": dep_parents_map,
        "dep_children_map": dep_children_map,
        "dependency_evidence_status": "verified" if chains else "no_sequential_chains_detected",
        "warnings": warnings,
    }


def segment_sequential_chain(
    chain: List[str],
    token_metrics_map: Dict[str, Dict[str, Any]],
    chain_id: str = "chain_1",
) -> Dict[str, Any]:
    """
    Segment a sequential chain using the recursive 50% cumulative token contribution rule.

    Preserves execution order, keeps input/output tokens separately recorded, isolates
    any single agent exceeding 50% of the original chain contribution, and falls back
    conservatively to singletons if token evidence is missing or unreliable.
    """
    if not chain:
        return {
            "chain_id": chain_id,
            "chain": [],
            "evidence_status": "insufficient_evidence",
            "fallback_applied": True,
            "total_chain_contribution": 0.0,
            "agent_metrics": {},
            "groups": [],
        }

    agent_metrics: Dict[str, Dict[str, Any]] = {}
    agent_contrib: Dict[str, float] = {}
    missing_or_bad: List[str] = []

    for aid in chain:
        tm = token_metrics_map.get(aid) or {}
        mean_in = tm.get("mean_prompt_tokens")
        mean_out = tm.get("mean_completion_tokens")
        mean_tot = tm.get("mean_tokens")
        obs = tm.get("observation_count", 0)

        valid = (
            isinstance(mean_in, (int, float))
            and not isinstance(mean_in, bool)
            and isinstance(mean_out, (int, float))
            and not isinstance(mean_out, bool)
            and mean_in >= 0
            and mean_out >= 0
            and isinstance(obs, (int, float))
            and obs > 0
        )
        if not valid:
            missing_or_bad.append(aid)
        else:
            c_val = float(mean_in) + float(mean_out)
            agent_contrib[aid] = c_val
            agent_metrics[aid] = {
                "agent_id": aid,
                "mean_input_tokens": round(float(mean_in), 2),
                "mean_output_tokens": round(float(mean_out), 2),
                "mean_total_tokens": round(float(mean_tot), 2) if isinstance(mean_tot, (int, float)) else None,
                "observation_count": int(obs),
                "agent_contribution": round(c_val, 2),
            }

    total_chain = sum(agent_contrib.values()) if not missing_or_bad else 0.0
    if missing_or_bad or total_chain <= 0:
        fallback_groups = [
            {
                "group_id": f"{chain_id}_group_{idx + 1}",
                "chain_id": chain_id,
                "agent_ids": [aid],
                "group_input_tokens": agent_metrics.get(aid, {}).get("mean_input_tokens"),
                "group_output_tokens": agent_metrics.get(aid, {}).get("mean_output_tokens"),
                "group_contribution_tokens": agent_metrics.get(aid, {}).get("agent_contribution"),
                "group_contribution_pct_of_chain": None,
                "cumulative_chain_pct_start": None,
                "cumulative_chain_pct_end": None,
                "boundary_after_agent": aid if idx < len(chain) - 1 else None,
                "is_isolated_singleton": True,
                "segmentation_depth": 0,
                "segmentation_reason": (
                    f"conservative_fallback_missing_or_unreliable_token_metrics:{','.join(missing_or_bad) or 'zero_total'}"
                ),
            }
            for idx, aid in enumerate(chain)
        ]
        return {
            "chain_id": chain_id,
            "chain": list(chain),
            "evidence_status": "insufficient_evidence",
            "fallback_applied": True,
            "total_chain_contribution": round(total_chain, 2),
            "agent_metrics": agent_metrics,
            "groups": fallback_groups,
        }

    for aid in chain:
        agent_metrics[aid]["contribution_index"] = round(agent_contrib[aid] / total_chain, 6)

    def _segment_subchain(subchain: List[str], depth: int = 0) -> List[Tuple[List[str], bool, int, str]]:
        if not subchain:
            return []

        if len(subchain) == 1:
            aid = subchain[0]
            over_50 = agent_contrib[aid] > (0.50 * total_chain + 1e-9)
            reason = (
                "single_agent_exceeds_50pct_threshold"
                if over_50
                else "singleton_group_after_boundary"
            )
            return [(list(subchain), over_50, depth, reason)]

        # Rule 6: Isolate any single agent contributing > 50% of the original chain
        for idx, aid in enumerate(subchain):
            if agent_contrib[aid] > (0.50 * total_chain + 1e-9):
                parts: List[Tuple[List[str], bool, int, str]] = []
                if idx > 0:
                    parts.extend(_segment_subchain(subchain[:idx], depth + 1))
                parts.append(([aid], True, depth, "single_agent_exceeds_50pct_threshold"))
                if idx + 1 < len(subchain):
                    parts.extend(_segment_subchain(subchain[idx + 1 :], depth + 1))
                return parts

        sub_total = sum(agent_contrib[a] for a in subchain)
        if depth > 0 and sub_total <= (0.50 * total_chain + 1e-9):
            return [(list(subchain), False, depth, "within_50pct_original_chain_contribution")]

        # Traverse subchain in execution order and split when cumulative >= 50%
        target = 0.50 * sub_total
        cum = 0.0
        cut = len(subchain) - 1
        for i, aid in enumerate(subchain):
            cum += agent_contrib[aid]
            if cum >= target - 1e-9:
                cut = i + 1
                break
        if cut <= 0:
            cut = 1
        if cut >= len(subchain):
            cut = len(subchain) - 1

        left = subchain[:cut]
        right = subchain[cut:]
        return _segment_subchain(left, depth + 1) + _segment_subchain(right, depth + 1)

    raw_groups = _segment_subchain(list(chain), depth=0)

    annotated_groups: List[Dict[str, Any]] = []
    running_cum = 0.0
    for idx, (grp_agents, is_singleton_over, depth, reason) in enumerate(raw_groups):
        g_in = sum(agent_metrics[a]["mean_input_tokens"] for a in grp_agents)
        g_out = sum(agent_metrics[a]["mean_output_tokens"] for a in grp_agents)
        g_contrib = sum(agent_contrib[a] for a in grp_agents)
        start_pct = round((running_cum / total_chain) * 100.0, 2)
        running_cum += g_contrib
        end_pct = round((running_cum / total_chain) * 100.0, 2)
        g_pct = round((g_contrib / total_chain) * 100.0, 2)

        annotated_groups.append({
            "group_id": f"{chain_id}_group_{idx + 1}",
            "chain_id": chain_id,
            "agent_ids": grp_agents,
            "group_input_tokens": round(g_in, 2),
            "group_output_tokens": round(g_out, 2),
            "group_contribution_tokens": round(g_contrib, 2),
            "group_contribution_pct_of_chain": g_pct,
            "cumulative_chain_pct_start": start_pct,
            "cumulative_chain_pct_end": end_pct,
            "boundary_after_agent": grp_agents[-1] if idx < len(raw_groups) - 1 else None,
            "is_isolated_singleton": bool(is_singleton_over),
            "segmentation_depth": depth,
            "segmentation_reason": reason,
        })

    # Order invariant check
    flattened = [a for g in annotated_groups for a in g["agent_ids"]]
    if flattened != list(chain):
        raise RuntimeError(f"Segmentation order invariant violated: {flattened} != {chain}")

    return {
        "chain_id": chain_id,
        "chain": list(chain),
        "evidence_status": "sufficient",
        "fallback_applied": False,
        "total_chain_contribution": round(total_chain, 2),
        "agent_metrics": agent_metrics,
        "groups": annotated_groups,
    }
