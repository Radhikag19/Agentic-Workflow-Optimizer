"""
TraceProfiler Agent Extractor Module.

Extracts internal agent definitions, configuration (role, goal, backstory, LLM),
tasks, registered tools, explicit code invocations, and helper functions using AST.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ingestion.ast_parser import (
    CallInfo,
    FunctionDefInfo,
    ParsedModule,
    safe_eval_literal,
)


# ============================================================================
# Extracted Data Structures
# ============================================================================

@dataclass
class ExtractedLLM:
    provider: Optional[str] = None
    model: Optional[str] = None
    resolution_status: str = "unresolved"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "resolution_status": self.resolution_status,
        }


@dataclass
class ExtractedTask:
    task_id: str
    agent_id: str
    description: Optional[str] = None
    expected_output: Optional[str] = None
    context_task_ids: List[str] = field(default_factory=list)
    source_file: str = ""
    start_line: int = 0
    end_line: int = 0
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "description": self.description,
            "expected_output": self.expected_output,
            "context_task_ids": self.context_task_ids,
            "evidence_refs": self.evidence_refs,
        }


@dataclass
class ExtractedAgent:
    agent_id: str
    name: str
    definition_type: str  # "function" | "class" | "assignment"
    source_file: str
    start_line: int
    end_line: int
    role: Optional[str] = None
    goal: Optional[str] = None
    backstory: Optional[str] = None
    llm: ExtractedLLM = field(default_factory=ExtractedLLM)
    tasks: List[ExtractedTask] = field(default_factory=list)
    registered_tools: List[str] = field(default_factory=list)
    helper_functions: List[str] = field(default_factory=list)
    explicit_code_invocations: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    raw_node: Optional[ast.AST] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "definition": {
                "type": self.definition_type,
                "name": self.name,
                "source_file": self.source_file,
                "start_line": self.start_line,
                "end_line": self.end_line,
            },
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "llm": self.llm.to_dict(),
            "tasks": [t.to_dict() for t in self.tasks],
            "registered_tools": self.registered_tools,
            "helper_functions": self.helper_functions,
            "explicit_code_invocations": self.explicit_code_invocations,
            "evidence_refs": self.evidence_refs,
        }


# ============================================================================
# LLM Model Resolution Helpers
# ============================================================================

def parse_llm_model_string(model_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse provider and model identifier from strings like 'groq/llama-3.3-70b-versatile'
    or 'openai/gpt-4o' or 'claude-3-5-sonnet'.
    """
    if not model_str or not isinstance(model_str, str):
        return None, None

    if "/" in model_str:
        parts = model_str.split("/", 1)
        provider = parts[0].strip().title()
        model_name = parts[1].strip()
        return provider, model_name

    # Known common prefix heuristics
    lower = model_str.lower()
    if lower.startswith("llama") or "groq" in lower:
        return "Groq", model_str
    if lower.startswith("gpt") or lower.startswith("text-embedding"):
        return "OpenAI", model_str
    if lower.startswith("claude"):
        return "Anthropic", model_str
    if lower.startswith("gemini"):
        return "Google", model_str

    return "Unknown", model_str


def resolve_llm_reference(
    node: Optional[ast.AST],
    scope_assignments: Dict[str, ast.AST],
    module: ParsedModule,
) -> ExtractedLLM:
    """
    Statically resolve an LLM reference from a node (e.g. LLM(model=...), 'groq/model', or var name).
    """
    if node is None:
        return ExtractedLLM(provider=None, model=None, resolution_status="unresolved")

    # Direct literal string e.g. llm="groq/llama-3.3-70b-versatile"
    stat, val, _ = safe_eval_literal(node)
    if stat and isinstance(val, str):
        prov, model_name = parse_llm_model_string(val)
        return ExtractedLLM(provider=prov, model=model_name, resolution_status="static")

    # If it's a Call: LLM(model=...) or ChatGroq(model_name=...)
    if isinstance(node, ast.Call):
        func_name = getattr(node.func, "id", "") or ast.unparse(node.func)
        model_val = None
        for kw in node.keywords:
            if kw.arg in ("model", "model_name", "model_id"):
                m_stat, m_val, _ = safe_eval_literal(kw.value)
                if m_stat and isinstance(m_val, str):
                    model_val = m_val
                    break
        if not model_val and node.args:
            m_stat, m_val, _ = safe_eval_literal(node.args[0])
            if m_stat and isinstance(m_val, str):
                model_val = m_val

        if model_val:
            prov, model_name = parse_llm_model_string(model_val)
            if prov == "Unknown" and "groq" in func_name.lower():
                prov = "Groq"
            return ExtractedLLM(provider=prov, model=model_name, resolution_status="static")

        return ExtractedLLM(provider=None, model=None, resolution_status="unresolved")

    # If it's a Name: lookup in local scope assignments
    if isinstance(node, ast.Name):
        var_name = node.id
        if var_name in scope_assignments:
            assigned_node = scope_assignments[var_name]
            return resolve_llm_reference(assigned_node, scope_assignments, module)

        # Check top level assignments in module
        for top_assign in module.top_level_assignments:
            if var_name in top_assign.targets and top_assign.value_node:
                return resolve_llm_reference(top_assign.value_node, scope_assignments, module)

        return ExtractedLLM(provider=None, model=None, resolution_status="unresolved")

    return ExtractedLLM(provider=None, model=None, resolution_status="unresolved")


# ============================================================================
# Agent Extractor Engine
# ============================================================================

class AgentExtractor:
    """Extracts agent definitions, tasks, tools, and helpers from parsed AST modules."""

    def __init__(self, parser_cache: Optional[Dict[str, ParsedModule]] = None) -> None:
        self.parser_cache: Dict[str, ParsedModule] = parser_cache or {}

    def extract_from_module(self, module: ParsedModule) -> List[ExtractedAgent]:
        """Extract all agent definitions present in a single module."""
        if module.parse_status != "parsed" or not module.ast_tree:
            return []

        agents: List[ExtractedAgent] = []

        # 1. Discover all tools and helper functions in this module
        module_tools = self._discover_tools(module)
        module_helpers = self._discover_helpers(module, module_tools)
        module_explicit_invocations = self._discover_explicit_invocations(module, module_tools)

        # 2. Check function definitions
        for fn_name, fn_info in module.functions.items():
            agent = self._inspect_function_for_agent(fn_info, module, module_tools, module_helpers, module_explicit_invocations)
            if agent:
                agents.append(agent)

        # 3. Check class definitions
        for cls_name, cls_info in module.classes.items():
            agent = self._inspect_class_for_agent(cls_info, module, module_tools, module_helpers, module_explicit_invocations)
            if agent:
                agents.append(agent)

        # 4. Check top-level assignments (e.g. `research_agent = Agent(...)`)
        for assign_info in module.top_level_assignments:
            if assign_info.call_info and self._is_agent_call(assign_info.call_info):
                agent = self._build_agent_from_call(
                    call=assign_info.call_info,
                    agent_name=assign_info.targets[0] if assign_info.targets else "unnamed_agent",
                    definition_type="assignment",
                    source_file=module.file_path,
                    start_line=assign_info.start_line,
                    end_line=assign_info.end_line,
                    scope_assignments={},
                    module=module,
                    module_tools=module_tools,
                    module_helpers=module_helpers,
                    module_explicit_invocations=module_explicit_invocations,
                )
                agents.append(agent)

        return agents

    def _is_agent_call(self, call: CallInfo) -> bool:
        """Check if call is constructing an Agent (e.g. Agent(...), create_crew_agent(...), etc.)."""
        target = call.method_name or call.func_name
        return target in ("Agent", "create_agent", "create_crew_agent", "CustomAgent")

    def _is_tool_decorator(self, decorator_name: str) -> bool:
        return decorator_name in ("tool", "crewai.tools.tool")

    def _discover_tools(self, module: ParsedModule) -> Set[str]:
        """Find functions decorated with @tool or classes inheriting from BaseTool."""
        tools: Set[str] = set()

        # Functions
        for fn_name, fn_info in module.functions.items():
            for dec in fn_info.decorators:
                if self._is_tool_decorator(dec):
                    tools.add(fn_name)

        # Classes
        for cls_name, cls_info in module.classes.items():
            if any("BaseTool" in base for base in cls_info.bases):
                tools.add(cls_name)

        return tools

    def _discover_helpers(self, module: ParsedModule, registered_tools: Set[str]) -> List[str]:
        """
        Identify ordinary deterministic helper functions.
        A function is a helper if it is NOT decorated with @tool and NOT an agent factory.
        """
        helpers: List[str] = []
        for fn_name, fn_info in module.functions.items():
            if fn_name in registered_tools:
                continue
            # Check if this function creates an agent
            has_agent_call = any(self._is_agent_call(c) for c in fn_info.calls)
            if not has_agent_call:
                helpers.append(fn_name)
        return sorted(helpers)

    def _discover_explicit_invocations(self, module: ParsedModule, registered_tools: Set[str]) -> List[str]:
        """Find explicit programmatic invocations like tool.run(...) or known_tool(...)."""
        explicit: List[str] = []
        for call in module.all_calls:
            # tool.run(...)
            if call.method_name == "run" and call.caller_obj in registered_tools:
                explicit.append(f"{call.caller_obj}.run()")
            elif call.func_name in registered_tools:
                explicit.append(f"{call.func_name}()")
            elif call.method_name == "run" and call.caller_obj:
                # e.g. search_tool.run()
                if "tool" in call.caller_obj.lower():
                    explicit.append(f"{call.caller_obj}.run()")
        return sorted(list(set(explicit)))

    def _inspect_function_for_agent(
        self,
        fn_info: FunctionDefInfo,
        module: ParsedModule,
        module_tools: Set[str],
        module_helpers: List[str],
        module_explicit_invocations: List[str],
    ) -> Optional[ExtractedAgent]:
        """Inspect a function definition for Agent construction."""
        # Find Agent call inside function
        agent_call = None
        for call in fn_info.calls:
            if self._is_agent_call(call):
                agent_call = call
                break

        if not agent_call:
            return None

        # Build local scope assignment map
        scope_assignments: Dict[str, ast.AST] = {}
        for assign in fn_info.assignments:
            for t in assign.targets:
                if assign.value_node:
                    scope_assignments[t] = assign.value_node

        # Resolve agent_id:
        # 1. If AGENT_ID constant exists at module level, use its value
        agent_id = None
        for assign in module.top_level_assignments:
            if "AGENT_ID" in assign.targets and assign.value_node:
                s_stat, s_val, _ = safe_eval_literal(assign.value_node)
                if s_stat and isinstance(s_val, str):
                    agent_id = s_val
                    break

        if not agent_id:
            fn_name = fn_info.name
            if fn_name in ("create_agent", "get_agent", "build_agent"):
                agent_id = Path(module.file_path).stem
            elif fn_name.startswith("create_"):
                agent_id = fn_name[len("create_"):]
            else:
                agent_id = fn_name

        return self._build_agent_from_call(
            call=agent_call,
            agent_name=agent_id,
            definition_type="function",
            source_file=module.file_path,
            start_line=fn_info.start_line,
            end_line=fn_info.end_line,
            scope_assignments=scope_assignments,
            module=module,
            module_tools=module_tools,
            module_helpers=module_helpers,
            module_explicit_invocations=module_explicit_invocations,
        )

    def _inspect_class_for_agent(
        self,
        cls_info: Any,
        module: ParsedModule,
        module_tools: Set[str],
        module_helpers: List[str],
        module_explicit_invocations: List[str],
    ) -> Optional[ExtractedAgent]:
        """Inspect a class definition for Agent creation."""
        for method_name, method_info in cls_info.methods.items():
            agent_call = next((c for c in method_info.calls if self._is_agent_call(c)), None)
            if agent_call:
                scope_assignments = {}
                for assign in method_info.assignments:
                    for t in assign.targets:
                        if assign.value_node:
                            scope_assignments[t] = assign.value_node
                return self._build_agent_from_call(
                    call=agent_call,
                    agent_name=cls_info.name,
                    definition_type="class",
                    source_file=module.file_path,
                    start_line=cls_info.start_line,
                    end_line=cls_info.end_line,
                    scope_assignments=scope_assignments,
                    module=module,
                    module_tools=module_tools,
                    module_helpers=module_helpers,
                    module_explicit_invocations=module_explicit_invocations,
                )
        return None

    def _build_agent_from_call(
        self,
        call: CallInfo,
        agent_name: str,
        definition_type: str,
        source_file: str,
        start_line: int,
        end_line: int,
        scope_assignments: Dict[str, ast.AST],
        module: ParsedModule,
        module_tools: Set[str],
        module_helpers: List[str],
        module_explicit_invocations: List[str],
    ) -> ExtractedAgent:
        """Extract configuration attributes from Agent(...) call."""
        kws = call.keywords

        role = None
        goal = None
        backstory = None

        if call.raw_node:
            for kw in call.raw_node.keywords:
                if kw.arg == "role":
                    role = self._resolve_text_value(kw.value, module, call)
                elif kw.arg == "goal":
                    goal = self._resolve_text_value(kw.value, module, call)
                elif kw.arg == "backstory":
                    backstory = self._resolve_text_value(kw.value, module, call)

        if role is None and isinstance(kws.get("role"), str):
            role = kws.get("role")
        if goal is None and isinstance(kws.get("goal"), str):
            goal = kws.get("goal")
        if backstory is None and isinstance(kws.get("backstory"), str):
            backstory = kws.get("backstory")

        # Resolve LLM
        llm_node = None
        if call.raw_node:
            for kw in call.raw_node.keywords:
                if kw.arg == "llm":
                    llm_node = kw.value
                    break

        extracted_llm = resolve_llm_reference(llm_node, scope_assignments, module)

        # Resolve registered tools assigned to this agent
        assigned_tools: List[str] = []
        if call.raw_node:
            for kw in call.raw_node.keywords:
                if kw.arg == "tools":
                    if isinstance(kw.value, (ast.List, ast.Tuple)):
                        for elt in kw.value.elts:
                            tool_name = getattr(elt, "id", None)
                            if not tool_name:
                                try:
                                    tool_name = ast.unparse(elt)
                                except Exception:
                                    pass
                            if tool_name:
                                assigned_tools.append(tool_name)
                    elif isinstance(kw.value, ast.Name):
                        assigned_tools.append(kw.value.id)

        if not assigned_tools:
            tools_arg = kws.get("tools")
            if isinstance(tools_arg, (list, tuple)):
                for t in tools_arg:
                    if isinstance(t, str):
                        assigned_tools.append(t)
            elif tools_arg and isinstance(tools_arg, str):
                assigned_tools.append(tools_arg)

        # Filter registered tools to those matching module tools or named tools
        filtered_registered_tools = [t for t in assigned_tools if t in module_tools or t.isidentifier()]

        # Extract tasks defined in this module that might reference this agent
        tasks = self._extract_tasks_from_module(module, agent_name)

        return ExtractedAgent(
            agent_id=agent_name,
            name=agent_name,
            definition_type=definition_type,
            source_file=source_file,
            start_line=start_line,
            end_line=end_line,
            role=role,
            goal=goal,
            backstory=backstory,
            llm=extracted_llm,
            tasks=tasks,
            registered_tools=filtered_registered_tools,
            helper_functions=module_helpers,
            explicit_code_invocations=module_explicit_invocations,
            evidence_refs=[],
            raw_node=call.raw_node,
        )

    def _resolve_text_value(self, val_node: Optional[ast.AST], module: ParsedModule, call: CallInfo) -> Optional[str]:
        """Resolve a string or variable reference to its text value."""
        if val_node is None:
            return None
        stat, val, raw = safe_eval_literal(val_node)
        if stat and isinstance(val, str):
            return val
        if isinstance(val_node, ast.Name):
            var_name = val_node.id
            # Search enclosing function assignments
            for fn in module.functions.values():
                if fn.start_line <= call.start_line <= fn.end_line:
                    for assign in fn.assignments:
                        if var_name in assign.targets and assign.value_node:
                            sub_stat, sub_val, sub_raw = safe_eval_literal(assign.value_node)
                            if sub_stat and isinstance(sub_val, str):
                                return sub_val
                            return sub_raw
            # Search top-level assignments
            for assign in module.top_level_assignments:
                if var_name in assign.targets and assign.value_node:
                    sub_stat, sub_val, sub_raw = safe_eval_literal(assign.value_node)
                    if sub_stat and isinstance(sub_val, str):
                        return sub_val
                    return sub_raw
        return val if stat else raw

    def _extract_tasks_from_module(self, module: ParsedModule, target_agent_id: str) -> List[ExtractedTask]:
        """Extract Task definitions associated with an agent."""
        tasks: List[ExtractedTask] = []
        task_idx = 1

        for call in module.all_calls:
            func_name = call.method_name or call.func_name
            if func_name in ("Task", "run_agent_task"):
                kws = call.keywords
                desc = None
                expected_out = None
                agent_val = kws.get("agent") or kws.get("agent_id")

                if call.raw_node:
                    for kw in call.raw_node.keywords:
                        if kw.arg in ("description", "task_description"):
                            desc = self._resolve_text_value(kw.value, module, call)
                        elif kw.arg in ("expected_output", "expected_output_desc"):
                            expected_out = self._resolve_text_value(kw.value, module, call)

                if desc is None:
                    desc = kws.get("description") or kws.get("task_description")
                    if not isinstance(desc, str):
                        desc = None
                if expected_out is None:
                    expected_out = kws.get("expected_output") or kws.get("expected_output_desc")
                    if not isinstance(expected_out, str):
                        expected_out = None

                context_tasks = []
                ctx_arg = kws.get("context")
                if isinstance(ctx_arg, (list, tuple)):
                    for c in ctx_arg:
                        if isinstance(c, str):
                            context_tasks.append(c)

                # Determine if this task belongs to target_agent_id
                agent_id_str = str(agent_val) if agent_val else target_agent_id

                t_id = call.assigned_to or f"task_{target_agent_id}_{task_idx}"
                task_idx += 1

                task = ExtractedTask(
                    task_id=t_id,
                    agent_id=target_agent_id,
                    description=desc,
                    expected_output=expected_out,
                    context_task_ids=context_tasks,
                    source_file=module.file_path,
                    start_line=call.start_line,
                    end_line=call.end_line,
                    evidence_refs=[],
                )
                tasks.append(task)

        return tasks
