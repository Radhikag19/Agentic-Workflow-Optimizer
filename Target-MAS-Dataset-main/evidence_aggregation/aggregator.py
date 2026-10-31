"""
Evidence Aggregator for TraceProfiler.

Consolidates:
1. workflow.json (Declared design)
2. workflow_context.json (Static AST implementation context)
3. live_*.json traces (Empirical runtime telemetry)
into unified, schema-compliant workflow_evidence.json artifacts.
"""

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evidence_aggregation.schema import (
    AgentReconciliation,
    AgentRuntimeMetrics,
    CostStats,
    DeclaredAgent,
    DeclaredDependency,
    DeclaredModelConfig,
    DependencyReconciliation,
    EvidenceMetadata,
    GraphEdge,
    GraphNode,
    InputFilesMetadata,
    LatencyStats,
    ModelReconciliation,
    ReconciliationAndLimitations,
    RunSummaryRecord,
    RuntimeObservations,
    StaticAgentContext,
    StaticImplementationContext,
    StaticTaskInfo,
    TokenStats,
    WorkflowDefinition,
    WorkflowEvidenceDocument,
    WorkflowGraph,
)


class EvidenceAggregator:
    """Aggregates declared, static, and runtime evidence for multi-agent workflows."""

    def __init__(
        self,
        agents_dir: Optional[Path] = None,
        traces_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        cwd = Path.cwd()
        # Resolve agents_dir
        if agents_dir:
            self.agents_dir = Path(agents_dir)
        elif (cwd / "Target-MAS-Dataset-main" / "Agents").exists():
            self.agents_dir = cwd / "Target-MAS-Dataset-main" / "Agents"
        elif (cwd / "Agents").exists():
            self.agents_dir = cwd / "Agents"
        else:
            self.agents_dir = Path("Agents")

        # Resolve traces_dir
        if traces_dir:
            self.traces_dir = Path(traces_dir)
        elif (cwd / "Ingestion_Traces_final" / "Ingestion_Traces").exists():
            self.traces_dir = cwd / "Ingestion_Traces_final" / "Ingestion_Traces"
        elif (cwd / ".." / "Ingestion_Traces_final" / "Ingestion_Traces").exists():
            self.traces_dir = (cwd / ".." / "Ingestion_Traces_final" / "Ingestion_Traces").resolve()
        elif (cwd / "Ingestion_Traces" / "Ingestion_Traces").exists():
            self.traces_dir = cwd / "Ingestion_Traces" / "Ingestion_Traces"
        else:
            self.traces_dir = Path("Ingestion_Traces_final/Ingestion_Traces")

        # Resolve output_dir
        if output_dir:
            self.output_dir = Path(output_dir)
        elif (cwd / "Target-MAS-Dataset-main").exists():
            self.output_dir = cwd / "Target-MAS-Dataset-main" / "Workflow_Evidence"
        else:
            self.output_dir = cwd / "Workflow_Evidence"

    def discover_workflows(self) -> List[str]:
        """Discover all workflow identifiers that have a directory in agents_dir."""
        if not self.agents_dir.exists():
            return []
        workflows = [
            d.name
            for d in sorted(self.agents_dir.iterdir())
            if d.is_dir() and not d.name.startswith((".", "_"))
        ]
        return workflows

    def aggregate_workflow(self, workflow_id: str) -> WorkflowEvidenceDocument:
        """Consolidate declared, static, and runtime evidence for a single workflow."""
        wf_dir = self.agents_dir / workflow_id
        wf_json_path = wf_dir / "workflow.json"
        wf_ctx_path = wf_dir / "workflow_context.json"
        trace_wf_dir = self.traces_dir / workflow_id

        # Load workflow.json
        wf_json_data = {}
        if wf_json_path.exists():
            try:
                wf_json_data = json.loads(wf_json_path.read_text(encoding="utf-8"))
            except Exception as e:
                wf_json_data = {"error": f"Failed to parse workflow.json: {str(e)}"}

        # Load workflow_context.json
        wf_ctx_data = {}
        if wf_ctx_path.exists():
            try:
                wf_ctx_data = json.loads(wf_ctx_path.read_text(encoding="utf-8"))
            except Exception as e:
                wf_ctx_data = {"error": f"Failed to parse workflow_context.json: {str(e)}"}

        # Load runtime trace files
        trace_files = []
        traces_data = []
        if trace_wf_dir.exists():
            trace_files = sorted([f.name for f in trace_wf_dir.glob("live_*.json")])
            for tf_name in trace_files:
                tf_path = trace_wf_dir / tf_name
                try:
                    t_data = json.loads(tf_path.read_text(encoding="utf-8"))
                    traces_data.append((tf_name, t_data))
                except Exception:
                    pass

        # 1. Build Metadata
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata = EvidenceMetadata(
            workflow_id=workflow_id,
            workflow_name=wf_json_data.get("workflow_name", workflow_id.replace("_", " ").title()),
            evidence_schema_version="1.0",
            generated_at=now_iso,
            traces_discovered=len(trace_files),
            traces_processed=len(traces_data),
            traces_failed=len(trace_files) - len(traces_data),
            input_files=InputFilesMetadata(
                workflow_json=str(wf_json_path),
                workflow_context_json=str(wf_ctx_path),
                traces_directory=str(trace_wf_dir),
                trace_files=trace_files,
            ),
            aggregation_status="complete" if traces_data else "incomplete_no_traces",
        )

        # 2. Build Workflow Definition (from workflow.json)
        workflow_definition = self._build_workflow_definition(wf_json_data)

        # 3. Build Static Implementation Context (from workflow_context.json)
        static_context = self._build_static_context(wf_ctx_data)

        # 4. Build Runtime Observations (from traces_data)
        runtime_observations = self._build_runtime_observations(traces_data)

        # 5. Build Workflow Graph (nodes & edges across all 3 tiers)
        workflow_graph = self._build_workflow_graph(
            workflow_id=workflow_id,
            definition=workflow_definition,
            static_context=static_context,
            runtime_obs=runtime_observations,
        )

        # 6. Build Reconciliation and Limitations
        reconciliation = self._build_reconciliation(
            workflow_definition=workflow_definition,
            static_context=static_context,
            runtime_obs=runtime_observations,
        )

        return WorkflowEvidenceDocument(
            metadata=metadata,
            workflow_definition=workflow_definition,
            static_implementation_context=static_context,
            runtime_observations=runtime_observations,
            workflow_graph=workflow_graph,
            reconciliation_and_limitations=reconciliation,
        )

    def _build_workflow_definition(self, data: Dict[str, Any]) -> WorkflowDefinition:
        """Extract and structure declared design from workflow.json."""
        raw_agents = data.get("agents", [])
        declared_agents: List[DeclaredAgent] = []
        declared_dependencies: List[DeclaredDependency] = []

        for a in raw_agents:
            if not isinstance(a, dict):
                continue
            aid = a.get("agent_id", "")
            deps = a.get("dependencies", [])
            declared_agents.append(
                DeclaredAgent(
                    agent_id=aid,
                    name=a.get("agent_name", a.get("name")),
                    role=a.get("role"),
                    responsibilities=a.get("responsibilities"),
                    declared_inputs=a.get("inputs", a.get("input_fields", [])),
                    declared_outputs=a.get("outputs", a.get("output_fields", [])),
                    declared_dependencies=deps,
                    assigned_key_env=a.get("assigned_key_env"),
                )
            )
            for d in deps:
                declared_dependencies.append(
                    DeclaredDependency(source=d, target=aid, relationship_type="declared_dependency")
                )

        exec_order = data.get("execution_order", data.get("baseline_execution_order", []))
        model_cfg = DeclaredModelConfig(
            model=data.get("fixed_model", data.get("baseline_model", data.get("model_configuration", {}).get("model") if isinstance(data.get("model_configuration"), dict) else None)),
            provider=data.get("llm_provider"),
            assigned_groq_key_env=data.get("assigned_groq_key_env"),
        )

        # Preserve optimizer annotations as declared hypotheses
        opt_annotations = {}
        if "optimizer_annotations" in data:
            opt_annotations["optimizer_annotations"] = data["optimizer_annotations"]
        if "optimization_opportunities" in data:
            opt_annotations["optimization_opportunities"] = data["optimization_opportunities"]
        if opt_annotations:
            opt_annotations["hypothesis_disclaimer"] = (
                "Declared candidate hypothesis from workflow specification, "
                "not an empirical or validated optimization opportunity."
            )

        return WorkflowDefinition(
            description=data.get("description"),
            version=str(data.get("version", "1.0")),
            declared_agents=declared_agents,
            declared_dependencies=declared_dependencies,
            declared_execution_order=exec_order,
            declared_model_configuration=model_cfg,
            optimizer_annotations=opt_annotations,
        )

    def _build_static_context(self, data: Dict[str, Any]) -> StaticImplementationContext:
        """Extract and structure implementation evidence from workflow_context.json."""
        source_files = [
            f.get("file_path", "")
            for f in data.get("source_inventory", {}).get("files", [])
            if isinstance(f, dict)
        ]

        agents: List[StaticAgentContext] = []
        for ad in data.get("agent_details", []):
            if not isinstance(ad, dict):
                continue
            tasks = []
            for t in ad.get("tasks", []):
                if isinstance(t, dict):
                    tasks.append(
                        StaticTaskInfo(
                            task_name=t.get("task_name"),
                            description=t.get("description"),
                            expected_output=t.get("expected_output"),
                            tools=t.get("tools", []),
                        )
                    )

            # Collect prompts preserving full verbatim text
            prompts = []
            for t in tasks:
                if t.description:
                    prompts.append(t.description)
            if ad.get("backstory"):
                prompts.append(f"Backstory: {ad['backstory']}")

            llm_info = ad.get("llm", {}) or {}
            src_def = ad.get("definition", {}) or {}

            agents.append(
                StaticAgentContext(
                    agent_id=ad.get("agent_id", ""),
                    role=ad.get("role"),
                    goal=ad.get("goal"),
                    backstory=ad.get("backstory"),
                    tasks=tasks,
                    prompts=prompts,
                    model=llm_info.get("model"),
                    provider=llm_info.get("provider"),
                    resolution_status=llm_info.get("resolution_status"),
                    registered_tools=ad.get("registered_tools", []),
                    helper_functions=ad.get("helper_functions", []),
                    source_file=src_def.get("source_file"),
                    start_line=src_def.get("start_line"),
                    end_line=src_def.get("end_line"),
                    evidence_refs=ad.get("evidence_refs", []),
                )
            )

        outer_flow = data.get("outer_workflow_flow", {}) or {}
        invocations = []
        for s in outer_flow.get("execution_steps", []):
            if isinstance(s, dict) and s.get("agent_id"):
                invocations.append(s)

        tools_helpers = data.get("tool_and_helper_inventory", {}) or {}
        limitations = data.get("analysis_limitations", {}) or {}

        return StaticImplementationContext(
            source_files=source_files,
            agents=agents,
            invocation_relationships=invocations,
            data_handoffs=[],  # Static AST extractor did not populate dynamic handoffs
            conditional_branches=[],
            tools_and_helpers=tools_helpers,
            analysis_limitations=limitations,
        )

    def _build_runtime_observations(
        self, traces: List[Tuple[str, Dict[str, Any]]]
    ) -> RuntimeObservations:
        """Calculate statistical metrics and structure runtime evidence across all traces."""
        total_runs = len(traces)
        if total_runs == 0:
            empty_latency = LatencyStats(
                mean_seconds=0.0, median_seconds=0.0, min_seconds=0.0, max_seconds=0.0, observation_count=0
            )
            empty_tokens = TokenStats(
                total_tokens=0, mean_tokens=0.0, total_prompt_tokens=0, mean_prompt_tokens=0.0,
                total_completion_tokens=0, mean_completion_tokens=0.0, observation_count=0
            )
            empty_cost = CostStats(
                total_cost_usd=0.0, mean_cost_usd=0.0, pricing_source="no_observations", observation_count=0
            )
            return RuntimeObservations(
                total_runs=0,
                successful_runs=0,
                failed_runs=0,
                observed_models=[],
                key_distribution={},
                workflow_latency=empty_latency,
                workflow_tokens=empty_tokens,
                workflow_cost=empty_cost,
                agent_metrics={},
                runs_summary=[],
            )

        wf_durations: List[float] = []
        wf_tokens: List[int] = []
        wf_prompt_tokens: List[int] = []
        wf_comp_tokens: List[int] = []
        wf_costs: List[float] = []
        observed_models: set = set()
        key_dist: Dict[str, int] = {}
        successful_runs = 0
        failed_runs = 0

        # Per-agent collection buckets
        per_agent_durs: Dict[str, List[float]] = {}
        per_agent_tokens: Dict[str, List[int]] = {}
        per_agent_prompt: Dict[str, List[int]] = {}
        per_agent_comp: Dict[str, List[int]] = {}
        per_agent_orders: Dict[str, set] = {}
        per_agent_keys: Dict[str, set] = {}
        per_agent_inputs: Dict[str, Any] = {}
        per_agent_outputs: Dict[str, Any] = {}
        per_agent_success: Dict[str, int] = {}
        per_agent_failure: Dict[str, int] = {}

        runs_summary: List[RunSummaryRecord] = []

        for tf_name, t in traces:
            status = t.get("status", "UNKNOWN")
            if status == "SUCCESS":
                successful_runs += 1
            else:
                failed_runs += 1

            dur = float(t.get("duration_seconds", 0.0))
            tot_tok = int(t.get("total_tokens", 0))
            p_tok = int(t.get("prompt_tokens", 0))
            c_tok = int(t.get("completion_tokens", 0))
            cost = float(t.get("estimated_cost_usd", 0.0))
            model = t.get("model")
            if model:
                observed_models.add(model)

            wf_durations.append(dur)
            wf_tokens.append(tot_tok)
            wf_prompt_tokens.append(p_tok)
            wf_comp_tokens.append(c_tok)
            wf_costs.append(cost)

            agent_execs = t.get("agent_executions", [])
            runs_summary.append(
                RunSummaryRecord(
                    run_id=t.get("run_id", tf_name.replace(".json", "")),
                    status=status,
                    duration_seconds=dur,
                    total_tokens=tot_tok,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    estimated_cost_usd=cost,
                    agent_count=len(agent_execs),
                    timestamp=t.get("timestamp", ""),
                    trace_file=tf_name,
                )
            )

            for a in agent_execs:
                if not isinstance(a, dict):
                    continue
                aid = a.get("agent_id", "")
                if not aid:
                    continue

                if aid not in per_agent_durs:
                    per_agent_durs[aid] = []
                    per_agent_tokens[aid] = []
                    per_agent_prompt[aid] = []
                    per_agent_comp[aid] = []
                    per_agent_orders[aid] = set()
                    per_agent_keys[aid] = set()
                    per_agent_success[aid] = 0
                    per_agent_failure[aid] = 0

                per_agent_durs[aid].append(float(a.get("duration_seconds", 0.0)))
                per_agent_tokens[aid].append(int(a.get("tokens", 0)))
                per_agent_prompt[aid].append(int(a.get("prompt_tokens", 0)))
                per_agent_comp[aid].append(int(a.get("completion_tokens", 0)))

                if "order" in a:
                    per_agent_orders[aid].add(int(a["order"]))

                key_id = a.get("assigned_key_identifier")
                if key_id:
                    k_short = key_id.split()[0]
                    per_agent_keys[aid].add(k_short)
                    key_dist[k_short] = key_dist.get(k_short, 0) + 1

                if a.get("status") == "SUCCESS":
                    per_agent_success[aid] += 1
                else:
                    per_agent_failure[aid] += 1

                # Capture first available structured output and input
                if aid not in per_agent_outputs and a.get("output"):
                    per_agent_outputs[aid] = a.get("output")
                if aid not in per_agent_inputs and a.get("input"):
                    per_agent_inputs[aid] = a.get("input")

        # Calculate workflow aggregates
        wf_latency = LatencyStats(
            mean_seconds=round(sum(wf_durations) / len(wf_durations), 4),
            median_seconds=round(statistics.median(wf_durations), 4),
            min_seconds=round(min(wf_durations), 4),
            max_seconds=round(max(wf_durations), 4),
            observation_count=len(wf_durations),
        )

        wf_token_stats = TokenStats(
            total_tokens=sum(wf_tokens),
            mean_tokens=round(sum(wf_tokens) / len(wf_tokens), 1),
            min_tokens=min(wf_tokens),
            max_tokens=max(wf_tokens),
            total_prompt_tokens=sum(wf_prompt_tokens),
            mean_prompt_tokens=round(sum(wf_prompt_tokens) / len(wf_prompt_tokens), 1),
            total_completion_tokens=sum(wf_comp_tokens),
            mean_completion_tokens=round(sum(wf_comp_tokens) / len(wf_comp_tokens), 1),
            observation_count=len(wf_tokens),
        )

        wf_cost_stats = CostStats(
            total_cost_usd=round(sum(wf_costs), 5),
            mean_cost_usd=round(sum(wf_costs) / len(wf_costs), 6),
            pricing_source="trace_telemetry_estimates",
            observation_count=len(wf_costs),
        )

        # Calculate per-agent aggregates
        agent_metrics: Dict[str, AgentRuntimeMetrics] = {}
        for aid, durs in per_agent_durs.items():
            toks = per_agent_tokens[aid]
            ptoks = per_agent_prompt[aid]
            ctoks = per_agent_comp[aid]
            count = len(durs)

            a_lat = LatencyStats(
                mean_seconds=round(sum(durs) / count, 4),
                median_seconds=round(statistics.median(durs), 4),
                min_seconds=round(min(durs), 4),
                max_seconds=round(max(durs), 4),
                observation_count=count,
            )

            a_toks = TokenStats(
                total_tokens=sum(toks),
                mean_tokens=round(sum(toks) / count, 1),
                min_tokens=min(toks) if toks else None,
                max_tokens=max(toks) if toks else None,
                total_prompt_tokens=sum(ptoks),
                mean_prompt_tokens=round(sum(ptoks) / count, 1),
                total_completion_tokens=sum(ctoks),
                mean_completion_tokens=round(sum(ctoks) / count, 1),
                observation_count=count,
            )

            agent_metrics[aid] = AgentRuntimeMetrics(
                agent_id=aid,
                invocation_count=count,
                success_count=per_agent_success.get(aid, count),
                failure_count=per_agent_failure.get(aid, 0),
                observed_execution_orders=sorted(list(per_agent_orders.get(aid, set()))),
                latency=a_lat,
                tokens=a_toks,
                cost={
                    "total_cost_usd": None,  # Not independently tracked per agent in traces
                    "mean_cost_usd": None,
                    "calculation_basis": "unresolved_at_agent_level_trace_provides_workflow_level_cost",
                },
                assigned_keys=sorted(list(per_agent_keys.get(aid, set()))),
                sample_inputs=per_agent_inputs.get(aid),
                sample_outputs=per_agent_outputs.get(aid),
                error_count=per_agent_failure.get(aid, 0),
            )

        return RuntimeObservations(
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            observed_models=sorted(list(observed_models)),
            key_distribution=key_dist,
            workflow_latency=wf_latency,
            workflow_tokens=wf_token_stats,
            workflow_cost=wf_cost_stats,
            agent_metrics=agent_metrics,
            runs_summary=runs_summary,
        )

    def _build_workflow_graph(
        self,
        workflow_id: str,
        definition: WorkflowDefinition,
        static_context: StaticImplementationContext,
        runtime_obs: RuntimeObservations,
    ) -> WorkflowGraph:
        """Construct multi-tier graph with nodes and provenance-annotated edges."""
        nodes_dict: Dict[str, GraphNode] = {}
        edges: List[GraphEdge] = []

        # 1. Workflow Node
        wf_node_id = f"workflow:{workflow_id}"
        nodes_dict[wf_node_id] = GraphNode(
            id=wf_node_id, type="workflow", name=workflow_id, source="multi_source"
        )

        # 2. Agent Nodes
        declared_agents_set = {a.agent_id for a in definition.declared_agents}
        static_agents_set = {a.agent_id for a in static_context.agents}
        runtime_agents_set = set(runtime_obs.agent_metrics.keys())
        all_agent_ids = sorted(list(declared_agents_set | static_agents_set | runtime_agents_set))

        for aid in all_agent_ids:
            node_id = f"agent:{aid}"
            srcs = []
            if aid in declared_agents_set:
                srcs.append("declared")
            if aid in static_agents_set:
                srcs.append("statically_observed")
            if aid in runtime_agents_set:
                srcs.append("runtime_observed")

            nodes_dict[node_id] = GraphNode(
                id=node_id,
                type="agent",
                name=aid,
                source="multi_source" if len(srcs) > 1 else (srcs[0] if srcs else "unknown"),
            )

        # 3. Task Nodes
        for ag in static_context.agents:
            for t in ag.tasks:
                t_name = t.task_name or f"{ag.agent_id}_task"
                t_node_id = f"task:{t_name}"
                nodes_dict[t_node_id] = GraphNode(
                    id=t_node_id, type="task", name=t_name, source="statically_observed"
                )
                # Edge: Agent -> Task Assignment
                edges.append(
                    GraphEdge(
                        source=f"agent:{ag.agent_id}",
                        target=t_node_id,
                        relationship="agent_task_assignment",
                        provenance="statically_observed",
                        evidence_ref=ag.evidence_refs,
                    )
                )

        # 4. Helper Function Nodes
        for ag in static_context.agents:
            for hf in ag.helper_functions:
                fn_node_id = f"function:{hf}"
                nodes_dict[fn_node_id] = GraphNode(
                    id=fn_node_id, type="function", name=hf, source="statically_observed"
                )
                edges.append(
                    GraphEdge(
                        source=f"agent:{ag.agent_id}",
                        target=fn_node_id,
                        relationship="function_call",
                        provenance="statically_observed",
                        evidence_ref=ag.evidence_refs,
                    )
                )

        # 5. Tool Nodes
        for ag in static_context.agents:
            for tool in ag.registered_tools:
                tool_node_id = f"tool:{tool}"
                nodes_dict[tool_node_id] = GraphNode(
                    id=tool_node_id, type="tool", name=tool, source="statically_observed"
                )
                edges.append(
                    GraphEdge(
                        source=f"agent:{ag.agent_id}",
                        target=tool_node_id,
                        relationship="explicit_tool_invocation",
                        provenance="statically_observed",
                        evidence_ref=ag.evidence_refs,
                    )
                )

        # 6. Workflow -> Agent Invocations
        for aid in static_agents_set:
            edges.append(
                GraphEdge(
                    source=wf_node_id,
                    target=f"agent:{aid}",
                    relationship="agent_invocation",
                    provenance="statically_observed",
                    evidence_ref="outer_workflow_flow",
                )
            )

        # 7. Declared Dependencies
        for dep in definition.declared_dependencies:
            edges.append(
                GraphEdge(
                    source=f"agent:{dep.source}",
                    target=f"agent:{dep.target}",
                    relationship="declared_dependency",
                    provenance="declared",
                    evidence_ref="workflow.json",
                )
            )

        # 8. Observed Runtime Execution Order
        # Extract consecutive agent sequence from runtime metrics
        sorted_runtime_agents = sorted(
            runtime_obs.agent_metrics.items(),
            key=lambda x: x[1].observed_execution_orders[0] if x[1].observed_execution_orders else 999,
        )
        for i in range(len(sorted_runtime_agents) - 1):
            curr_aid = sorted_runtime_agents[i][0]
            next_aid = sorted_runtime_agents[i + 1][0]
            edges.append(
                GraphEdge(
                    source=f"agent:{curr_aid}",
                    target=f"agent:{next_aid}",
                    relationship="observed_execution_order",
                    provenance="runtime_observed",
                    evidence_ref="traces_telemetry",
                )
            )

        return WorkflowGraph(nodes=list(nodes_dict.values()), edges=edges)

    def _build_reconciliation(
        self,
        workflow_definition: WorkflowDefinition,
        static_context: StaticImplementationContext,
        runtime_obs: RuntimeObservations,
    ) -> ReconciliationAndLimitations:
        """Reconcile design, static implementation, and runtime observations."""
        declared_agents = {a.agent_id: a for a in workflow_definition.declared_agents}
        static_agents = {a.agent_id: a for a in static_context.agents}
        runtime_agents = runtime_obs.agent_metrics

        all_agent_ids = sorted(list(set(declared_agents.keys()) | set(static_agents.keys()) | set(runtime_agents.keys())))

        agent_reconciliation: List[AgentReconciliation] = []
        for aid in all_agent_ids:
            in_dec = aid in declared_agents
            in_stat = aid in static_agents
            in_run = aid in runtime_agents

            if in_dec and in_stat and in_run:
                status = "fully_verified"
                notes = "Agent declared in design, identified in AST, and verified across all runtime traces."
            elif in_dec and in_stat and not in_run:
                status = "not_observed_at_runtime"
                notes = "Agent declared and implemented in code but not executed in captured runtime traces."
            elif not in_dec and in_stat and in_run:
                status = "undeclared_runtime_agent"
                notes = "Agent implemented in code and observed at runtime but missing from workflow.json manifest."
            elif in_dec and not in_stat:
                status = "declared_unimplemented"
                notes = "Agent declared in design but not found in Python source code."
            else:
                status = "discrepancy"
                notes = f"Dec={in_dec}, Stat={in_stat}, Run={in_run}"

            agent_reconciliation.append(
                AgentReconciliation(
                    agent_id=aid,
                    declared=in_dec,
                    statically_observed=in_stat,
                    runtime_observed=in_run,
                    status=status,
                    notes=notes,
                )
            )

        # Dependency Reconciliation
        dep_reconciliation: List[DependencyReconciliation] = []
        for dep in workflow_definition.declared_dependencies:
            # Check if this transition appeared in runtime observed orders
            src_orders = runtime_obs.agent_metrics.get(dep.source, None)
            tgt_orders = runtime_obs.agent_metrics.get(dep.target, None)
            runtime_verified = False
            if src_orders and tgt_orders and src_orders.observed_execution_orders and tgt_orders.observed_execution_orders:
                runtime_verified = src_orders.observed_execution_orders[0] < tgt_orders.observed_execution_orders[0]

            dep_reconciliation.append(
                DependencyReconciliation(
                    source=dep.source,
                    target=dep.target,
                    declared=True,
                    statically_observed=False,  # AST graph lacked direct data-flow edges
                    runtime_observed=runtime_verified,
                    status="runtime_order_consistent" if runtime_verified else "runtime_unverified",
                    notes="Declared dependency matches runtime chronological execution sequence." if runtime_verified else "Dependency not verified at runtime.",
                )
            )

        # Model Reconciliation
        dec_model = workflow_definition.declared_model_configuration.model
        stat_models = {a.model for a in static_context.agents if a.model}
        run_models = runtime_obs.observed_models

        run_model = run_models[0] if run_models else None
        stat_model = list(stat_models)[0] if stat_models else None

        if dec_model and run_model and dec_model == run_model:
            model_status = "matching"
            model_notes = f"Declared model '{dec_model}' matches runtime enforced model '{run_model}'."
        else:
            model_status = "conflicting_or_unresolved"
            model_notes = f"Declared: '{dec_model}', Statically extracted: '{stat_model}', Runtime observed: '{run_model}'."

        model_rec = ModelReconciliation(
            declared_model=dec_model,
            statically_extracted_model=stat_model,
            runtime_observed_model=run_model,
            status=model_status,
            notes=model_notes,
        )

        limitations = [
            "Per-agent cost in traces is not tracked individually; only workflow-level cost is recorded.",
            "Dynamic runtime branch outcomes and retry loops are recorded as sequential executions.",
            "Static prompt text reflects task description parameters; dynamic LLM runtime prompt interpolations are reflected in trace prompt tokens.",
            "Static AST dependency graph was a flat star topology; inter-agent dependencies are reconciled from workflow.json and runtime trace sequences.",
        ]

        return ReconciliationAndLimitations(
            agent_reconciliation=agent_reconciliation,
            dependency_reconciliation=dep_reconciliation,
            model_reconciliation=model_rec,
            limitations=limitations,
        )

    def generate_all(self) -> Dict[str, Path]:
        """Generate workflow_evidence.json for all discovered workflows."""
        workflows = self.discover_workflows()
        results: Dict[str, Path] = {}

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for wf in workflows:
            doc = self.aggregate_workflow(wf)
            wf_out_dir = self.output_dir / wf
            wf_out_dir.mkdir(parents=True, exist_ok=True)
            out_file = wf_out_dir / "workflow_evidence.json"

            # Serialize using Pydantic
            out_file.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
            results[wf] = out_file

        return results

