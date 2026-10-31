"""
TraceProfiler Workflow Analyzer Module.

Reconstructs outer workflow control flow, resolves agent invocations via AST symbol
and call graph tracing, extracts internal orchestrator configurations, and builds
evidence-backed dependency graphs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from schemas.workflow_context_schema import (
    CONTROLLED_EXECUTION_OPERATIONS,
    CONTROLLED_NODE_TYPES,
    RELATIONSHIP_TYPES,
)
from ingestion.ast_parser import (
    ASTParser,
    CallInfo,
    FunctionDefInfo,
    ParsedModule,
    default_ast_parser,
    safe_eval_literal,
)
from ingestion.agent_extractor import AgentExtractor, ExtractedAgent, ExtractedTask


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WorkflowAnalysisResult:
    workflow_id: str
    workflow_name: str
    source_dir: str
    entrypoint: str
    parsed_files: List[Dict[str, Any]]
    failed_files: List[Dict[str, Any]]
    discovered_agent_files: int
    identified_agents: List[ExtractedAgent]
    invoked_agent_ids: Set[str]
    uninvoked_agent_ids: Set[str]
    tasks: List[ExtractedTask]
    outer_flow: Dict[str, Any]
    orchestrator: Dict[str, Any]
    dependency_nodes: List[Dict[str, Any]]
    dependency_edges: List[Dict[str, Any]]
    registered_tools: List[str]
    explicit_code_invocations: List[str]
    helper_functions: List[str]
    unresolved_references: List[Dict[str, Any]]
    evidence_records: List[Dict[str, Any]]


# ============================================================================
# Workflow Analyzer Engine
# ============================================================================

class WorkflowAnalyzer:
    """Analyzes a multi-agent workflow directory using AST without executing source code."""

    def __init__(self, parser: Optional[ASTParser] = None) -> None:
        self.parser = parser or default_ast_parser
        self.agent_extractor = AgentExtractor(parser_cache=self.parser.cache)

    def analyze_workflow_dir(self, workflow_dir: Union[str, Path]) -> WorkflowAnalysisResult:
        """Inspect and reconstruct the multi-agent workflow in the specified directory."""
        wf_dir_path = Path(workflow_dir).resolve()
        wf_id = wf_dir_path.name
        wf_name = wf_id.replace("_", " ").title()

        workflow_py = wf_dir_path / "workflow.py"
        entrypoint_file = "workflow.py" if workflow_py.exists() else ""

        # 1. Discover all Python files (excluding __init__.py and hidden files)
        py_files = sorted([p for p in wf_dir_path.glob("*.py") if not p.name.startswith(".") and p.name != "__init__.py"])
        candidate_agent_files = [p for p in py_files if p.name != "workflow.py"]

        parsed_files_info: List[Dict[str, Any]] = []
        failed_files_info: List[Dict[str, Any]] = []
        parsed_modules: Dict[str, ParsedModule] = {}

        for p in py_files:
            rel_name = p.name
            mod = self.parser.parse_file(str(p))
            parsed_modules[rel_name] = mod
            if mod.parse_status == "parsed":
                parsed_files_info.append({
                    "path": rel_name,
                    "sha256": mod.sha256,
                    "parse_status": "parsed",
                })
            else:
                failed_files_info.append({
                    "path": rel_name,
                    "sha256": None,
                    "parse_status": "failed",
                    "error": mod.error_message,
                })

        # 2. Extract agent definitions from candidate agent files
        all_identified_agents: List[ExtractedAgent] = []
        agent_by_module: Dict[str, List[ExtractedAgent]] = {}

        for p in candidate_agent_files:
            rel_name = p.name
            mod = parsed_modules.get(rel_name)
            if mod and mod.parse_status == "parsed":
                extracted = self.agent_extractor.extract_from_module(mod)
                agent_by_module[rel_name] = extracted
                all_identified_agents.extend(extracted)

        # 3. Analyze workflow.py
        wf_mod = parsed_modules.get("workflow.py")
        if not wf_mod or wf_mod.parse_status != "parsed":
            # Workflow missing or unparseable
            return self._build_empty_result(
                wf_id, wf_name, str(wf_dir_path), entrypoint_file,
                len(py_files), len(candidate_agent_files), parsed_files_info, failed_files_info,
            )

        # 4. Resolve agent invocations from workflow.py
        invoked_agent_ids, uninvoked_agent_ids, invocation_edges = self._resolve_agent_invocations(
            wf_mod=wf_mod,
            all_agents=all_identified_agents,
            agent_by_module=agent_by_module,
            parsed_modules=parsed_modules,
        )

        # 5. Extract Outer Workflow Flow
        outer_flow, flow_edges = self._extract_outer_workflow_flow(wf_mod, all_identified_agents)

        # 6. Extract Orchestrator Configuration
        orchestrator_proc, orch_edges = self._extract_orchestrator_process(wf_mod, parsed_modules)

        # 7. Collect all tasks across workflow and agent modules
        # Check tasks created in workflow.py and associate with corresponding agents
        wf_tasks = self._extract_tasks_from_workflow(wf_mod, all_identified_agents)
        for t in wf_tasks:
            target_ag = next(
                (ag for ag in all_identified_agents if ag.agent_id == t.agent_id or ag.name == t.agent_id),
                None,
            )
            if target_ag:
                target_ag.tasks.append(t)
            elif all_identified_agents:
                all_identified_agents[0].tasks.append(t)

        all_tasks: List[ExtractedTask] = []
        for ag in all_identified_agents:
            all_tasks.extend(ag.tasks)

        # Deduplicate tasks by task_id
        unique_tasks_dict: Dict[str, ExtractedTask] = {}
        for t in all_tasks:
            if t.task_id not in unique_tasks_dict:
                unique_tasks_dict[t.task_id] = t
        unique_tasks = list(unique_tasks_dict.values())

        # 8. Build Tools, Helpers, Invocations
        registered_tools: Set[str] = set()
        helper_functions: Set[str] = set()
        explicit_invocations: Set[str] = set()

        for ag in all_identified_agents:
            registered_tools.update(ag.registered_tools)
            helper_functions.update(ag.helper_functions)
            explicit_invocations.update(ag.explicit_code_invocations)

        # Workflow helper functions & explicit calls
        for fn_name, fn_info in wf_mod.functions.items():
            if fn_name not in ("run_workflow", "main") and not fn_name.startswith("create_"):
                helper_functions.add(fn_name)

        for call in wf_mod.all_calls:
            if call.method_name == "run" and call.caller_obj in registered_tools:
                explicit_invocations.add(f"{call.caller_obj}.run()")

        # 9. Build Evidence Records
        evidence_records, ev_map = self._build_evidence_records(
            wf_mod=wf_mod,
            all_agents=all_identified_agents,
            tasks=unique_tasks,
            outer_flow=outer_flow,
            candidate_files=candidate_agent_files,
        )

        # 10. Construct Dependency Graph
        dep_nodes, dep_edges = self._build_dependency_graph(
            wf_mod=wf_mod,
            all_agents=all_identified_agents,
            invoked_ids=invoked_agent_ids,
            tasks=unique_tasks,
            registered_tools=registered_tools,
            helper_functions=helper_functions,
            invocation_edges=invocation_edges,
            flow_edges=flow_edges,
            orch_edges=orch_edges,
            ev_map=ev_map,
        )

        return WorkflowAnalysisResult(
            workflow_id=wf_id,
            workflow_name=wf_name,
            source_dir=str(wf_dir_path),
            entrypoint=entrypoint_file,
            parsed_files=parsed_files_info,
            failed_files=failed_files_info,
            discovered_agent_files=len(candidate_agent_files),
            identified_agents=all_identified_agents,
            invoked_agent_ids=invoked_agent_ids,
            uninvoked_agent_ids=uninvoked_agent_ids,
            tasks=unique_tasks,
            outer_flow=outer_flow,
            orchestrator=orchestrator_proc,
            dependency_nodes=dep_nodes,
            dependency_edges=dep_edges,
            registered_tools=sorted(list(registered_tools)),
            explicit_code_invocations=sorted(list(explicit_invocations)),
            helper_functions=sorted(list(helper_functions)),
            unresolved_references=[],
            evidence_records=evidence_records,
        )

    def _resolve_agent_invocations(
        self,
        wf_mod: ParsedModule,
        all_agents: List[ExtractedAgent],
        agent_by_module: Dict[str, List[ExtractedAgent]],
        parsed_modules: Dict[str, ParsedModule],
    ) -> Tuple[Set[str], Set[str], List[Dict[str, Any]]]:
        """
        Resolve agent invocations from workflow.py using AST symbol resolution:
        1. Pattern A: from agent_a import create_agent_a -> calls create_agent_a()
        2. Pattern B: import intake_agent -> calls intake_agent.run_contract_intake()
        """
        invoked_ids: Set[str] = set()
        edges: List[Dict[str, Any]] = []

        wf_dir_name = Path(wf_mod.file_path).parent.name

        # Map imported symbol -> module filename (e.g. "intake_agent.py")
        imported_symbols: Dict[str, str] = {}
        imported_modules: Set[str] = set()

        for imp in wf_mod.imports:
            if imp.is_from and imp.module:
                mod_parts = imp.module.split(".")
                last_part = mod_parts[-1]
                if last_part == wf_dir_name or last_part == "Agents":
                    # e.g. from Agents.contract_review import intake_agent
                    for it in imp.names:
                        mod_filename = f"{it.name}.py"
                        imported_symbols[it.effective_name] = mod_filename
                        imported_modules.add(mod_filename)
                else:
                    # e.g. from Agents.customer_support.ticket_classification_agent import create_agent
                    mod_filename = f"{last_part}.py"
                    for it in imp.names:
                        imported_symbols[it.effective_name] = mod_filename
                        imported_modules.add(mod_filename)
            elif not imp.is_from:
                for it in imp.names:
                    mod_filename = f"{it.name}.py"
                    imported_modules.add(mod_filename)
                    imported_symbols[it.effective_name] = mod_filename

        # Check calls in workflow.py
        for call in wf_mod.all_calls:
            func_name = call.func_name
            caller = call.caller_obj
            method = call.method_name

            # 1. Direct function call: e.g. create_classification_agent() or create_category_agent()
            if func_name in imported_symbols:
                target_mod_name = imported_symbols[func_name]
                agents_in_mod = agent_by_module.get(target_mod_name, [])
                for ag in agents_in_mod:
                    invoked_ids.add(ag.agent_id)
                    edges.append({
                        "source": "workflow:workflow.py",
                        "target": f"agent:{ag.agent_id}",
                        "relationship": "agent_invocation",
                        "data_reference": None,
                        "resolution_status": "static",
                        "evidence_refs": [],
                    })

            # 2. Module-qualified call: e.g. intake_agent.create_agent() or intake_agent.run_contract_intake()
            if caller and caller in imported_symbols:
                target_mod_name = imported_symbols[caller]
                agents_in_mod = agent_by_module.get(target_mod_name, [])
                for ag in agents_in_mod:
                    invoked_ids.add(ag.agent_id)
                    edges.append({
                        "source": "workflow:workflow.py",
                        "target": f"agent:{ag.agent_id}",
                        "relationship": "agent_invocation",
                        "data_reference": None,
                        "resolution_status": "static",
                        "evidence_refs": [],
                    })

            # 3. Direct agent ID / name call fallback
            for ag in all_agents:
                if func_name == ag.name or (caller and caller == ag.name):
                    invoked_ids.add(ag.agent_id)
                    edges.append({
                        "source": "workflow:workflow.py",
                        "target": f"agent:{ag.agent_id}",
                        "relationship": "agent_invocation",
                        "data_reference": None,
                        "resolution_status": "static",
                        "evidence_refs": [],
                    })

        all_agent_ids = {ag.agent_id for ag in all_agents}
        uninvoked_ids = all_agent_ids - invoked_ids

        return invoked_ids, uninvoked_ids, edges

    def _extract_outer_workflow_flow(
        self,
        wf_mod: ParsedModule,
        all_agents: List[ExtractedAgent],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Extract procedural execution steps, branches, loops, and IO from workflow.py."""
        steps: List[Dict[str, Any]] = []
        branches: List[Dict[str, Any]] = []
        loops: List[Dict[str, Any]] = []
        early_returns: List[Dict[str, Any]] = []
        flow_edges: List[Dict[str, Any]] = []

        step_idx = 0

        # Classify function calls into controlled execution operations
        for call in wf_mod.all_calls:
            target = call.func_name
            caller = call.caller_obj
            method = call.method_name

            op = "function_call"
            if method == "kickoff" or "kickoff" in target:
                op = "crew_kickoff"
            elif target in ("Crew", "crewai.Crew"):
                op = "crew_creation"
            elif target in ("Task", "crewai.Task"):
                op = "task_creation"
            elif target in ("Agent", "crewai.Agent"):
                op = "agent_instantiation"
            elif "create_" in target and "agent" in target:
                op = "agent_factory_call"
            elif method == "run" or "tool" in target.lower():
                op = "tool_invocation"

            step_id = f"step_{step_idx:02d}"
            step_idx += 1

            steps.append({
                "step_id": step_id,
                "operation": op,
                "target": target,
                "sequence_index": step_idx - 1,
                "arguments": call.keywords if call.keywords else call.args,
                "assigned_variable": call.assigned_to,
                "start_line": call.start_line,
                "end_line": call.end_line,
                "evidence_refs": [],
            })

        # Extract conditional branches, loops, returns by inspecting AST nodes
        if wf_mod.ast_tree:
            for node in ast.walk(wf_mod.ast_tree):
                if isinstance(node, ast.If):
                    cond_repr = ""
                    try:
                        cond_repr = ast.unparse(node.test)
                    except Exception:
                        cond_repr = "<condition>"
                    branches.append({
                        "condition": cond_repr,
                        "start_line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                    })
                elif isinstance(node, (ast.For, ast.While)):
                    loop_repr = ""
                    try:
                        loop_repr = ast.unparse(node.target if isinstance(node, ast.For) else node.test)
                    except Exception:
                        loop_repr = "<loop>"
                    loops.append({
                        "type": "for" if isinstance(node, ast.For) else "while",
                        "expression": loop_repr,
                        "start_line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                    })
                elif isinstance(node, ast.Return):
                    ret_stat, ret_val, ret_repr = safe_eval_literal(node.value)
                    early_returns.append({
                        "return_value": ret_val if ret_stat else ret_repr,
                        "start_line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                    })

        # Extract workflow inputs/outputs
        wf_inputs: List[Dict[str, Any]] = []
        wf_outputs: List[Dict[str, Any]] = []

        # Find main workflow entrypoint function (e.g. run_workflow)
        main_fn = wf_mod.functions.get("run_workflow") or wf_mod.functions.get("main")
        if main_fn:
            for arg in main_fn.args:
                wf_inputs.append({"name": arg, "type": "parameter"})
            for ret in main_fn.returns:
                wf_outputs.append({"value": ret.get("representation"), "start_line": ret.get("start_line")})

        flow = {
            "entrypoint": "workflow.py",
            "execution_steps": steps,
            "conditional_branches": branches,
            "loops": loops,
            "early_returns": early_returns,
            "workflow_inputs": wf_inputs,
            "workflow_outputs": wf_outputs,
        }
        return flow, flow_edges

    def _extract_orchestrator_process(
        self,
        wf_mod: ParsedModule,
        parsed_modules: Dict[str, ParsedModule],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Extract CrewAI framework configuration, process type, and manager settings."""
        framework = "crewai"
        process_value: Optional[str] = None
        process_res_status = "unresolved"
        delegation_agents: List[str] = []
        manager_llm: Optional[str] = None
        manager_agent: Optional[str] = None
        context_links: List[Dict[str, Any]] = []
        async_tasks: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # Search for Crew(...) call in workflow.py
        for call in wf_mod.all_calls:
            target = call.method_name or call.func_name
            if target == "Crew":
                kws = call.keywords
                proc_arg = kws.get("process")
                if proc_arg:
                    proc_str = str(proc_arg)
                    if "sequential" in proc_str.lower():
                        process_value = "sequential"
                        process_res_status = "static"
                    elif "hierarchical" in proc_str.lower():
                        process_value = "hierarchical"
                        process_res_status = "static"

                if "manager_llm" in kws:
                    manager_llm = str(kws["manager_llm"])
                if "manager_agent" in kws:
                    manager_agent = str(kws["manager_agent"])

            if target == "Task":
                kws = call.keywords
                ctx_arg = kws.get("context")
                if isinstance(ctx_arg, (list, tuple)):
                    for c in ctx_arg:
                        context_links.append({"task": call.assigned_to, "context_task": str(c)})
                        if call.assigned_to and str(c):
                            edges.append({
                                "source": f"task:{str(c)}",
                                "target": f"task:{call.assigned_to}",
                                "relationship": "task_context",
                                "data_reference": None,
                                "resolution_status": "static",
                                "evidence_refs": [],
                            })
                if kws.get("async_execution") in (True, "True"):
                    async_tasks.append({"task": call.assigned_to})

        # Check delegation across all parsed modules
        for mod in parsed_modules.values():
            for call in mod.all_calls:
                if call.method_name == "Agent" or call.func_name == "Agent":
                    if call.keywords.get("allow_delegation") in (True, "True"):
                        role = call.keywords.get("role") or call.assigned_to
                        if role:
                            delegation_agents.append(str(role))

        orch = {
            "framework": framework,
            "process_type": {
                "value": process_value,
                "resolution_status": process_res_status,
            },
            "delegation_enabled_agents": sorted(list(set(delegation_agents))),
            "manager": {
                "manager_llm": manager_llm,
                "manager_agent": manager_agent,
            },
            "task_context_links": context_links,
            "async_tasks": async_tasks,
            "runtime_scheduling_observable": False,
        }
        return orch, edges

    def _resolve_text_value(self, val_node: Optional[ast.AST], module: ParsedModule, call: CallInfo) -> Optional[str]:
        """Resolve a string or variable reference to its text value."""
        if val_node is None:
            return None
        stat, val, raw = safe_eval_literal(val_node)
        if stat and isinstance(val, str):
            return val
        if isinstance(val_node, ast.Name):
            var_name = val_node.id
            for fn in module.functions.values():
                if fn.start_line <= call.start_line <= fn.end_line:
                    for assign in fn.assignments:
                        if var_name in assign.targets and assign.value_node:
                            sub_stat, sub_val, sub_raw = safe_eval_literal(assign.value_node)
                            if sub_stat and isinstance(sub_val, str):
                                return sub_val
                            return sub_raw
            for assign in module.top_level_assignments:
                if var_name in assign.targets and assign.value_node:
                    sub_stat, sub_val, sub_raw = safe_eval_literal(assign.value_node)
                    if sub_stat and isinstance(sub_val, str):
                        return sub_val
                    return sub_raw
        return val if stat else raw

    def _extract_tasks_from_workflow(
        self,
        wf_mod: ParsedModule,
        all_agents: List[ExtractedAgent],
    ) -> List[ExtractedTask]:
        """Extract Task definitions created inside workflow.py."""
        tasks: List[ExtractedTask] = []
        task_idx = 1
        for call in wf_mod.all_calls:
            target = call.method_name or call.func_name
            if target == "Task":
                kws = call.keywords
                desc = None
                expected_out = None
                agent_val = kws.get("agent")

                if call.raw_node:
                    for kw in call.raw_node.keywords:
                        if kw.arg == "description":
                            desc = self._resolve_text_value(kw.value, wf_mod, call)
                        elif kw.arg == "expected_output":
                            expected_out = self._resolve_text_value(kw.value, wf_mod, call)

                if desc is None and isinstance(kws.get("description"), str):
                    desc = kws.get("description")
                if expected_out is None and isinstance(kws.get("expected_output"), str):
                    expected_out = kws.get("expected_output")

                context_tasks = []
                ctx_arg = kws.get("context")
                if isinstance(ctx_arg, (list, tuple)):
                    for c in ctx_arg:
                        if isinstance(c, str):
                            context_tasks.append(c)

                assigned_agent_id = str(agent_val) if agent_val else "unknown_agent"
                t_id = call.assigned_to or f"task_{task_idx}"
                task_idx += 1

                tasks.append(ExtractedTask(
                    task_id=t_id,
                    agent_id=assigned_agent_id,
                    description=desc,
                    expected_output=expected_out,
                    context_task_ids=context_tasks,
                    source_file=wf_mod.file_path,
                    start_line=call.start_line,
                    end_line=call.end_line,
                    evidence_refs=[],
                ))
        return tasks

    def _build_evidence_records(
        self,
        wf_mod: ParsedModule,
        all_agents: List[ExtractedAgent],
        tasks: List[ExtractedTask],
        outer_flow: Dict[str, Any],
        candidate_files: List[Path],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Construct deterministic line-range evidence records and map entity -> evidence_id."""
        evidence: List[Dict[str, Any]] = []
        ev_map: Dict[str, str] = {}
        ev_counter = 1

        # 1. Workflow entrypoint evidence
        wf_ev_id = f"ev_wf_{ev_counter:03d}"
        ev_counter += 1
        evidence.append({
            "evidence_id": wf_ev_id,
            "source_file": "workflow.py",
            "start_line": 1,
            "end_line": max(1, len(wf_mod.source_code.splitlines())),
            "entity_id": "workflow:workflow.py",
            "evidence_type": "workflow_entrypoint",
        })
        ev_map["workflow:workflow.py"] = wf_ev_id

        # 2. Agent definition evidence
        for ag in all_agents:
            rel_file = Path(ag.source_file).name
            ev_id = f"ev_agent_{ev_counter:03d}"
            ev_counter += 1
            evidence.append({
                "evidence_id": ev_id,
                "source_file": rel_file,
                "start_line": max(1, ag.start_line),
                "end_line": max(1, ag.end_line),
                "entity_id": f"agent:{ag.agent_id}",
                "evidence_type": "agent_definition",
            })
            ag.evidence_refs = [ev_id]
            ev_map[f"agent:{ag.agent_id}"] = ev_id

        # 3. Task evidence
        for t in tasks:
            rel_file = Path(t.source_file).name if t.source_file else "workflow.py"
            ev_id = f"ev_task_{ev_counter:03d}"
            ev_counter += 1
            evidence.append({
                "evidence_id": ev_id,
                "source_file": rel_file,
                "start_line": max(1, t.start_line),
                "end_line": max(1, t.end_line),
                "entity_id": f"task:{t.task_id}",
                "evidence_type": "task_definition",
            })
            t.evidence_refs = [ev_id]
            ev_map[f"task:{t.task_id}"] = ev_id

        # 4. Step evidence
        for st in outer_flow.get("execution_steps", []):
            st_id = st["step_id"]
            s_line = st.get("start_line", 1)
            e_line = st.get("end_line", s_line)
            ev_id = f"ev_step_{ev_counter:03d}"
            ev_counter += 1
            evidence.append({
                "evidence_id": ev_id,
                "source_file": "workflow.py",
                "start_line": max(1, s_line),
                "end_line": max(1, e_line),
                "entity_id": f"step:{st_id}",
                "evidence_type": "execution_step",
            })
            st["evidence_refs"] = [ev_id]
            # Remove internal lines from step dict to match schema
            st.pop("start_line", None)
            st.pop("end_line", None)

        return evidence, ev_map

    def _build_dependency_graph(
        self,
        wf_mod: ParsedModule,
        all_agents: List[ExtractedAgent],
        invoked_ids: Set[str],
        tasks: List[ExtractedTask],
        registered_tools: Set[str],
        helper_functions: Set[str],
        invocation_edges: List[Dict[str, Any]],
        flow_edges: List[Dict[str, Any]],
        orch_edges: List[Dict[str, Any]],
        ev_map: Dict[str, str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Construct valid dependency graph nodes and edges with strict referential integrity."""
        nodes: Dict[str, Dict[str, Any]] = {}

        # 1. Workflow Node
        wf_node_id = "workflow:workflow.py"
        nodes[wf_node_id] = {
            "id": wf_node_id,
            "type": "workflow",
            "source_file": "workflow.py",
            "evidence_refs": [ev_map[wf_node_id]] if wf_node_id in ev_map else [],
        }

        # 2. Agent Nodes
        for ag in all_agents:
            ag_node_id = f"agent:{ag.agent_id}"
            rel_file = Path(ag.source_file).name
            ev_id = ev_map.get(ag_node_id)
            nodes[ag_node_id] = {
                "id": ag_node_id,
                "type": "agent",
                "source_file": rel_file,
                "evidence_refs": [ev_id] if ev_id else [],
            }

        # 3. Task Nodes
        for t in tasks:
            t_node_id = f"task:{t.task_id}"
            rel_file = Path(t.source_file).name if t.source_file else "workflow.py"
            ev_id = ev_map.get(t_node_id)
            nodes[t_node_id] = {
                "id": t_node_id,
                "type": "task",
                "source_file": rel_file,
                "evidence_refs": [ev_id] if ev_id else [],
            }

        # 4. Tool Nodes
        for tool in registered_tools:
            tool_node_id = f"tool:{tool}"
            nodes[tool_node_id] = {
                "id": tool_node_id,
                "type": "tool",
                "source_file": None,
                "evidence_refs": [],
            }

        # 5. Helper Function Nodes
        for h in helper_functions:
            fn_node_id = f"function:{h}"
            nodes[fn_node_id] = {
                "id": fn_node_id,
                "type": "function",
                "source_file": None,
                "evidence_refs": [],
            }

        # Combine Edges
        all_raw_edges = []
        all_raw_edges.extend(invocation_edges)
        all_raw_edges.extend(flow_edges)
        all_raw_edges.extend(orch_edges)

        # Add explicit tool invocation edges
        for ag in all_agents:
            for tool_call in ag.explicit_code_invocations:
                # e.g. search_market.run()
                tool_name = tool_call.split(".")[0].replace("()", "")
                tool_node_id = f"tool:{tool_name}"
                if tool_node_id in nodes:
                    all_raw_edges.append({
                        "source": f"agent:{ag.agent_id}",
                        "target": tool_node_id,
                        "relationship": "explicit_tool_invocation",
                        "data_reference": None,
                        "resolution_status": "static",
                        "evidence_refs": ag.evidence_refs,
                    })

        # Deduplicate and validate that source and target exist in nodes
        final_edges: List[Dict[str, Any]] = []
        edge_keys: Set[Tuple[str, str, str]] = set()

        for e in all_raw_edges:
            src = e["source"]
            tgt = e["target"]
            rel = e["relationship"]
            if src in nodes and tgt in nodes:
                key = (src, tgt, rel)
                if key not in edge_keys:
                    edge_keys.add(key)
                    # Attach evidence
                    if not e.get("evidence_refs"):
                        if src in ev_map:
                            e["evidence_refs"] = [ev_map[src]]
                        elif tgt in ev_map:
                            e["evidence_refs"] = [ev_map[tgt]]
                    final_edges.append(e)

        return list(nodes.values()), final_edges

    def _build_empty_result(
        self,
        wf_id: str,
        wf_name: str,
        source_dir: str,
        entrypoint: str,
        tot_py: int,
        cand_files: int,
        parsed_files: List[Dict[str, Any]],
        failed_files: List[Dict[str, Any]],
    ) -> WorkflowAnalysisResult:
        """Construct empty failed analysis result."""
        return WorkflowAnalysisResult(
            workflow_id=wf_id,
            workflow_name=wf_name,
            source_dir=source_dir,
            entrypoint=entrypoint,
            parsed_files=parsed_files,
            failed_files=failed_files,
            discovered_agent_files=cand_files,
            identified_agents=[],
            invoked_agent_ids=set(),
            uninvoked_agent_ids=set(),
            tasks=[],
            outer_flow={
                "entrypoint": entrypoint,
                "execution_steps": [],
                "conditional_branches": [],
                "loops": [],
                "early_returns": [],
                "workflow_inputs": [],
                "workflow_outputs": [],
            },
            orchestrator={
                "framework": "crewai",
                "process_type": {"value": None, "resolution_status": "unresolved"},
                "delegation_enabled_agents": [],
                "manager": {"manager_llm": None, "manager_agent": None},
                "task_context_links": [],
                "async_tasks": [],
                "runtime_scheduling_observable": False,
            },
            dependency_nodes=[],
            dependency_edges=[],
            registered_tools=[],
            explicit_code_invocations=[],
            helper_functions=[],
            unresolved_references=[],
            evidence_records=[],
        )
