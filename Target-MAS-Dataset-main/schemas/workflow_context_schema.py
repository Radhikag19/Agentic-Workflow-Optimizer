"""
TraceProfiler Workflow Context Schema.

Defines the data structures, validation rules, and serialization mechanisms
for `workflow_context.json`, which represents a multi-agent workflow analyzed
statically via AST parsing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Union


# ============================================================================
# Constants & Permitted Value Sets
# ============================================================================

SCHEMA_VERSION: str = "1.0"

RESOLUTION_STATUSES: frozenset[str] = frozenset({
    "static",
    "partial",
    "unresolved",
    "not_applicable",
})

ANALYSIS_STATUSES: frozenset[str] = frozenset({
    "completed",
    "partial",
    "failed",
})

INVOCATION_STATUSES: frozenset[str] = frozenset({
    "invoked",
    "uninvoked",
    "unresolved",
})

COMPLETENESS_VALUES: frozenset[str] = frozenset({
    "complete",
    "partial",
    "failed",
})

RELATIONSHIP_TYPES: frozenset[str] = frozenset({
    "agent_invocation",
    "task_context",
    "data_handoff",
    "function_call",
    "conditional_dependency",
    "explicit_tool_invocation",
})

CONTROLLED_NODE_TYPES: frozenset[str] = frozenset({
    "agent",
    "task",
    "tool",
    "function",
    "workflow",
    "other",
    "unknown",
})

CONTROLLED_EXECUTION_OPERATIONS: frozenset[str] = frozenset({
    "agent_factory_call",
    "agent_instantiation",
    "task_creation",
    "crew_creation",
    "crew_kickoff",
    "function_call",
    "tool_invocation",
    "conditional_branch",
    "loop",
    "return",
    "other",
    "unknown",
})

TOKEN_ESTIMATION_DEFAULT_METHOD: str = "heuristic_character_ratio"


# ============================================================================
# Exceptions
# ============================================================================

class ValidationError(Exception):
    """Raised when workflow context data violates schema structure or constraints."""

    def __init__(self, message: str, errors: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.errors: List[str] = errors or [message]

    def __str__(self) -> str:
        if len(self.errors) <= 1:
            return self.message
        return f"{self.message}\n  - " + "\n  - ".join(self.errors)


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class WorkflowMetadata:
    workflow_id: str
    workflow_name: str
    source_directory: str
    workflow_entrypoint: str
    analysis_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SourceFileEntry:
    path: str
    sha256: Optional[str] = None
    parse_status: str = "parsed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SourceInventory:
    total_python_files: int = 0
    agent_candidate_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    files: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentInventoryEntry:
    agent_id: str
    source_file: str
    definition_status: str = "static"
    invocation_status: str = "unresolved"
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentInventory:
    discovered_agent_files: int = 0
    identified_agent_definitions: int = 0
    workflow_invoked_agents: int = 0
    uninvoked_agent_definitions: int = 0
    agents: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentDefinitionInfo:
    type: str
    name: str
    source_file: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMReference:
    provider: Optional[str] = None
    model: Optional[str] = None
    resolution_status: str = "unresolved"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskDefinition:
    task_id: str
    agent_id: str
    description: Optional[str] = None
    expected_output: Optional[str] = None
    context_task_ids: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentDetail:
    agent_id: str
    definition: Dict[str, Any]
    role: Optional[str] = None
    goal: Optional[str] = None
    backstory: Optional[str] = None
    llm: Dict[str, Any] = field(default_factory=lambda: {
        "provider": None,
        "model": None,
        "resolution_status": "unresolved",
    })
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    registered_tools: List[str] = field(default_factory=list)
    helper_functions: List[str] = field(default_factory=list)
    explicit_code_invocations: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionStep:
    step_id: str
    operation: str
    target: str
    sequence_index: int
    arguments: Optional[Any] = None
    assigned_variable: Optional[str] = None
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OuterWorkflowFlow:
    entrypoint: str
    execution_steps: List[Dict[str, Any]] = field(default_factory=list)
    conditional_branches: List[Dict[str, Any]] = field(default_factory=list)
    loops: List[Dict[str, Any]] = field(default_factory=list)
    early_returns: List[Dict[str, Any]] = field(default_factory=list)
    workflow_inputs: List[Dict[str, Any]] = field(default_factory=list)
    workflow_outputs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessTypeInfo:
    value: Optional[str] = None
    resolution_status: str = "unresolved"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ManagerInfo:
    manager_llm: Optional[str] = None
    manager_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestratorProcess:
    framework: Optional[str] = None
    process_type: Dict[str, Any] = field(default_factory=lambda: {
        "value": None,
        "resolution_status": "unresolved",
    })
    delegation_enabled_agents: List[str] = field(default_factory=list)
    manager: Dict[str, Any] = field(default_factory=lambda: {
        "manager_llm": None,
        "manager_agent": None,
    })
    task_context_links: List[Dict[str, Any]] = field(default_factory=list)
    async_tasks: List[Dict[str, Any]] = field(default_factory=list)
    runtime_scheduling_observable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphNode:
    id: str
    type: str
    source_file: Optional[str] = None
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    source: str
    target: str
    relationship: str
    data_reference: Optional[str] = None
    resolution_status: str = "static"
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DependencyGraph:
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeLLMToolSelection:
    observable: bool = False
    reason: str = "Runtime LLM tool selection is unobservable via static AST analysis"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolAndHelperInventory:
    registered_tools: List[Any] = field(default_factory=list)
    explicit_code_invocations: List[Any] = field(default_factory=list)
    helper_functions: List[Any] = field(default_factory=list)
    runtime_llm_tool_selection: Dict[str, Any] = field(default_factory=lambda: {
        "observable": False,
        "reason": "Runtime LLM tool selection is unobservable via static AST analysis",
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceEntry:
    evidence_id: str
    source_file: str
    start_line: int
    end_line: int
    entity_id: str
    evidence_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceAndProvenance:
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisLimitations:
    unresolved_references: List[Dict[str, Any]] = field(default_factory=list)
    dynamic_expressions: List[Dict[str, Any]] = field(default_factory=list)
    unsupported_patterns: List[Dict[str, Any]] = field(default_factory=list)
    static_analysis_limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutputMetrics:
    serialized_bytes: int = 0
    estimated_tokens: int = 0
    token_estimation_method: str = TOKEN_ESTIMATION_DEFAULT_METHOD

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisSummary:
    total_agents: int = 0
    invoked_agents: int = 0
    uninvoked_agents: int = 0
    total_tasks: int = 0
    total_dependencies: int = 0
    unresolved_relationships: int = 0
    completeness: Union[str, float] = "complete"
    output_metrics: Dict[str, Any] = field(default_factory=lambda: {
        "serialized_bytes": 0,
        "estimated_tokens": 0,
        "token_estimation_method": TOKEN_ESTIMATION_DEFAULT_METHOD,
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowContext:
    workflow_metadata: Dict[str, Any]
    source_inventory: Dict[str, Any]
    agent_inventory: Dict[str, Any]
    agent_details: List[Dict[str, Any]]
    outer_workflow_flow: Dict[str, Any]
    orchestrator_process: Dict[str, Any]
    dependency_graph: Dict[str, Any]
    tool_and_helper_inventory: Dict[str, Any]
    evidence_and_provenance: Dict[str, Any]
    analysis_limitations: Dict[str, Any]
    analysis_summary: Dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workflow_metadata": self.workflow_metadata,
            "source_inventory": self.source_inventory,
            "agent_inventory": self.agent_inventory,
            "agent_details": self.agent_details,
            "outer_workflow_flow": self.outer_workflow_flow,
            "orchestrator_process": self.orchestrator_process,
            "dependency_graph": self.dependency_graph,
            "tool_and_helper_inventory": self.tool_and_helper_inventory,
            "evidence_and_provenance": self.evidence_and_provenance,
            "analysis_limitations": self.analysis_limitations,
            "analysis_summary": self.analysis_summary,
        }

    def to_json(self, indent: int = 2, final_validation: bool = False, strict_evidence: bool = False) -> str:
        data = self.to_dict()
        validate_workflow_context(data, final_validation=final_validation, strict_evidence=strict_evidence)
        return json.dumps(data, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], final_validation: bool = False, strict_evidence: bool = False) -> "WorkflowContext":
        validate_workflow_context(data, final_validation=final_validation, strict_evidence=strict_evidence)
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            workflow_metadata=data["workflow_metadata"],
            source_inventory=data["source_inventory"],
            agent_inventory=data["agent_inventory"],
            agent_details=data["agent_details"],
            outer_workflow_flow=data["outer_workflow_flow"],
            orchestrator_process=data["orchestrator_process"],
            dependency_graph=data["dependency_graph"],
            tool_and_helper_inventory=data["tool_and_helper_inventory"],
            evidence_and_provenance=data["evidence_and_provenance"],
            analysis_limitations=data["analysis_limitations"],
            analysis_summary=data["analysis_summary"],
        )

    @classmethod
    def from_json(cls, json_str: str, final_validation: bool = False, strict_evidence: bool = False) -> "WorkflowContext":
        data = json.loads(json_str)
        return cls.from_dict(data, final_validation=final_validation, strict_evidence=strict_evidence)


# ============================================================================
# Factory Functions
# ============================================================================

def create_empty_workflow_context(
    workflow_id: str,
    workflow_name: str,
    source_directory: str,
    workflow_entrypoint: str = "workflow.py",
    analysis_status: str = "completed",
) -> Dict[str, Any]:
    """Create a minimally populated, valid workflow context dictionary."""
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow_metadata": {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "source_directory": source_directory,
            "workflow_entrypoint": workflow_entrypoint,
            "analysis_status": analysis_status,
        },
        "source_inventory": {
            "total_python_files": 0,
            "agent_candidate_files": 0,
            "parsed_files": 0,
            "failed_files": 0,
            "files": [],
        },
        "agent_inventory": {
            "discovered_agent_files": 0,
            "identified_agent_definitions": 0,
            "workflow_invoked_agents": 0,
            "uninvoked_agent_definitions": 0,
            "agents": [],
        },
        "agent_details": [],
        "outer_workflow_flow": {
            "entrypoint": workflow_entrypoint,
            "execution_steps": [],
            "conditional_branches": [],
            "loops": [],
            "early_returns": [],
            "workflow_inputs": [],
            "workflow_outputs": [],
        },
        "orchestrator_process": {
            "framework": None,
            "process_type": {
                "value": None,
                "resolution_status": "unresolved",
            },
            "delegation_enabled_agents": [],
            "manager": {
                "manager_llm": None,
                "manager_agent": None,
            },
            "task_context_links": [],
            "async_tasks": [],
            "runtime_scheduling_observable": False,
        },
        "dependency_graph": {
            "nodes": [],
            "edges": [],
        },
        "tool_and_helper_inventory": {
            "registered_tools": [],
            "explicit_code_invocations": [],
            "helper_functions": [],
            "runtime_llm_tool_selection": {
                "observable": False,
                "reason": "Runtime LLM tool selection is unobservable via static AST analysis",
            },
        },
        "evidence_and_provenance": {
            "evidence": [],
        },
        "analysis_limitations": {
            "unresolved_references": [],
            "dynamic_expressions": [],
            "unsupported_patterns": [],
            "static_analysis_limitations": [],
        },
        "analysis_summary": {
            "total_agents": 0,
            "invoked_agents": 0,
            "uninvoked_agents": 0,
            "total_tasks": 0,
            "total_dependencies": 0,
            "unresolved_relationships": 0,
            "completeness": "complete",
            "output_metrics": {
                "serialized_bytes": 0,
                "estimated_tokens": 0,
                "token_estimation_method": TOKEN_ESTIMATION_DEFAULT_METHOD,
            },
        },
    }


# ============================================================================
# Validation Engine
# ============================================================================

REQUIRED_ROOT_SECTIONS: frozenset[str] = frozenset({
    "schema_version",
    "workflow_metadata",
    "source_inventory",
    "agent_inventory",
    "agent_details",
    "outer_workflow_flow",
    "orchestrator_process",
    "dependency_graph",
    "tool_and_helper_inventory",
    "evidence_and_provenance",
    "analysis_limitations",
    "analysis_summary",
})


def validate_workflow_context(
    data: Union[Dict[str, Any], WorkflowContext],
    final_validation: bool = False,
    strict_evidence: bool = False,
) -> None:
    """
    Validate a workflow context dictionary or instance against all structural,
    type, value set, identifier uniqueness, controlled vocabulary, and referential constraints.

    Args:
        data: The workflow context data to validate.
        final_validation: If True, enforces cross-section summary counts and evidence resolution.
        strict_evidence: If True, requires all evidence_refs to exist in evidence_and_provenance.

    Raises:
        ValidationError: If any validation checks fail.
    """
    if isinstance(data, WorkflowContext):
        data = data.to_dict()

    if not isinstance(data, dict):
        raise ValidationError(f"Workflow context must be a dictionary, got {type(data).__name__}")

    errors: List[str] = []

    # ------------------------------------------------------------------------
    # 1. Root Sections Check
    # ------------------------------------------------------------------------
    missing_roots = REQUIRED_ROOT_SECTIONS - set(data.keys())
    if missing_roots:
        errors.append(f"Missing required root sections: {sorted(missing_roots)}")
        raise ValidationError(
            f"Validation failed with {len(errors)} error(s)",
            errors=errors,
        )

    # ------------------------------------------------------------------------
    # 2. workflow_metadata Validation
    # ------------------------------------------------------------------------
    metadata = data.get("workflow_metadata")
    if not isinstance(metadata, dict):
        errors.append("workflow_metadata must be a dictionary")
    else:
        for req_field in ["workflow_id", "workflow_name", "source_directory", "workflow_entrypoint", "analysis_status"]:
            if req_field not in metadata:
                errors.append(f"workflow_metadata missing required field: '{req_field}'")
            elif not isinstance(metadata[req_field], str):
                errors.append(f"workflow_metadata['{req_field}'] must be a string")

        status = metadata.get("analysis_status")
        if isinstance(status, str) and status not in ANALYSIS_STATUSES:
            errors.append(
                f"workflow_metadata['analysis_status'] '{status}' is invalid; "
                f"must be one of {sorted(ANALYSIS_STATUSES)}"
            )

    # ------------------------------------------------------------------------
    # 3. source_inventory Validation
    # ------------------------------------------------------------------------
    src_inv = data.get("source_inventory")
    if not isinstance(src_inv, dict):
        errors.append("source_inventory must be a dictionary")
    else:
        for req_field in ["total_python_files", "agent_candidate_files", "parsed_files", "failed_files"]:
            val = src_inv.get(req_field)
            if not isinstance(val, int) or val < 0:
                errors.append(f"source_inventory['{req_field}'] must be a non-negative integer")
        files = src_inv.get("files")
        if not isinstance(files, list):
            errors.append("source_inventory['files'] must be a list")
        else:
            for idx, fentry in enumerate(files):
                if not isinstance(fentry, dict):
                    errors.append(f"source_inventory['files'][{idx}] must be a dictionary")
                    continue
                for ffield in ["path", "parse_status"]:
                    if ffield not in fentry or not isinstance(fentry[ffield], str):
                        errors.append(f"source_inventory['files'][{idx}] missing or non-string '{ffield}'")
                if "sha256" in fentry and fentry["sha256"] is not None and not isinstance(fentry["sha256"], str):
                    errors.append(f"source_inventory['files'][{idx}]['sha256'] must be string or null")

    # ------------------------------------------------------------------------
    # 4. agent_inventory & Namespace Uniqueness
    # ------------------------------------------------------------------------
    agent_inv = data.get("agent_inventory")
    inventory_agent_ids: Set[str] = set()
    invoked_count_inventory = 0
    uninvoked_count_inventory = 0

    if not isinstance(agent_inv, dict):
        errors.append("agent_inventory must be a dictionary")
    else:
        for req_field in ["discovered_agent_files", "identified_agent_definitions", "workflow_invoked_agents", "uninvoked_agent_definitions"]:
            val = agent_inv.get(req_field)
            if not isinstance(val, int) or val < 0:
                errors.append(f"agent_inventory['{req_field}'] must be a non-negative integer")
        agents = agent_inv.get("agents")
        if not isinstance(agents, list):
            errors.append("agent_inventory['agents'] must be a list")
        else:
            for idx, aentry in enumerate(agents):
                if not isinstance(aentry, dict):
                    errors.append(f"agent_inventory['agents'][{idx}] must be a dictionary")
                    continue
                for req_key in ["agent_id", "source_file", "definition_status", "invocation_status", "evidence_refs"]:
                    if req_key not in aentry:
                        errors.append(f"agent_inventory['agents'][{idx}] missing required field '{req_key}'")

                aid = aentry.get("agent_id")
                if isinstance(aid, str):
                    if aid in inventory_agent_ids:
                        errors.append(f"Duplicate agent_id '{aid}' in agent_inventory['agents']")
                    inventory_agent_ids.add(aid)
                elif aid is not None:
                    errors.append(f"agent_inventory['agents'][{idx}]['agent_id'] must be a string")

                inv_status = aentry.get("invocation_status")
                if inv_status in INVOCATION_STATUSES:
                    if inv_status == "invoked":
                        invoked_count_inventory += 1
                    elif inv_status == "uninvoked":
                        uninvoked_count_inventory += 1
                elif inv_status is not None:
                    errors.append(
                        f"agent_inventory['agents'][{idx}]['invocation_status'] '{inv_status}' is invalid; "
                        f"must be one of {sorted(INVOCATION_STATUSES)}"
                    )

                def_status = aentry.get("definition_status")
                if def_status is not None and def_status not in RESOLUTION_STATUSES:
                    errors.append(
                        f"agent_inventory['agents'][{idx}]['definition_status'] '{def_status}' is invalid; "
                        f"must be one of {sorted(RESOLUTION_STATUSES)}"
                    )

    # ------------------------------------------------------------------------
    # 5. agent_details Validation
    # ------------------------------------------------------------------------
    agent_details = data.get("agent_details")
    detail_agent_ids: Set[str] = set()
    all_task_ids: Set[str] = set()

    if not isinstance(agent_details, list):
        errors.append("agent_details must be a list")
    else:
        for idx, adetail in enumerate(agent_details):
            if not isinstance(adetail, dict):
                errors.append(f"agent_details[{idx}] must be a dictionary")
                continue
            for req_key in ["agent_id", "definition", "role", "goal", "backstory", "llm", "tasks",
                            "registered_tools", "helper_functions", "explicit_code_invocations", "evidence_refs"]:
                if req_key not in adetail:
                    errors.append(f"agent_details[{idx}] missing required field '{req_key}'")

            aid = adetail.get("agent_id")
            if isinstance(aid, str):
                if aid in detail_agent_ids:
                    errors.append(f"Duplicate agent_id '{aid}' in agent_details")
                detail_agent_ids.add(aid)
            elif aid is not None:
                errors.append(f"agent_details[{idx}]['agent_id'] must be a string")

            # Definition validation
            defn = adetail.get("definition")
            if not isinstance(defn, dict):
                errors.append(f"agent_details[{idx}]['definition'] must be a dictionary")
            else:
                for dfield in ["type", "name", "source_file"]:
                    if dfield not in defn or not isinstance(defn[dfield], str):
                        errors.append(f"agent_details[{idx}]['definition'] missing or non-string '{dfield}'")
                for lfield in ["start_line", "end_line"]:
                    if lfield in defn and defn[lfield] is not None and not isinstance(defn[lfield], int):
                        errors.append(f"agent_details[{idx}]['definition']['{lfield}'] must be integer or null")

            # LLM validation
            llm = adetail.get("llm")
            if not isinstance(llm, dict):
                errors.append(f"agent_details[{idx}]['llm'] must be a dictionary")
            else:
                for lkey in ["provider", "model", "resolution_status"]:
                    if lkey not in llm:
                        errors.append(f"agent_details[{idx}]['llm'] missing required field '{lkey}'")
                res_status = llm.get("resolution_status")
                if res_status is not None and res_status not in RESOLUTION_STATUSES:
                    errors.append(
                        f"agent_details[{idx}]['llm']['resolution_status'] '{res_status}' is invalid; "
                        f"must be one of {sorted(RESOLUTION_STATUSES)}"
                    )

            # Tasks validation
            tasks = adetail.get("tasks")
            if not isinstance(tasks, list):
                errors.append(f"agent_details[{idx}]['tasks'] must be a list")
            else:
                for tidx, task in enumerate(tasks):
                    if not isinstance(task, dict):
                        errors.append(f"agent_details[{idx}]['tasks'][{tidx}] must be a dictionary")
                        continue
                    for tfield in ["task_id", "description", "expected_output", "agent_id", "context_task_ids", "evidence_refs"]:
                        if tfield not in task:
                            errors.append(f"agent_details[{idx}]['tasks'][{tidx}] missing required field '{tfield}'")
                    tid = task.get("task_id")
                    if isinstance(tid, str):
                        if tid in all_task_ids:
                            errors.append(f"Duplicate task_id '{tid}' in agent tasks")
                        all_task_ids.add(tid)
                    elif tid is not None:
                        errors.append(f"agent_details[{idx}]['tasks'][{tidx}]['task_id'] must be a string")

    # ------------------------------------------------------------------------
    # 6. outer_workflow_flow Validation
    # ------------------------------------------------------------------------
    flow = data.get("outer_workflow_flow")
    execution_step_ids: Set[str] = set()
    if not isinstance(flow, dict):
        errors.append("outer_workflow_flow must be a dictionary")
    else:
        for rfield in ["entrypoint", "execution_steps", "conditional_branches", "loops",
                       "early_returns", "workflow_inputs", "workflow_outputs"]:
            if rfield not in flow:
                errors.append(f"outer_workflow_flow missing required field '{rfield}'")

        steps = flow.get("execution_steps")
        if not isinstance(steps, list):
            errors.append("outer_workflow_flow['execution_steps'] must be a list")
        else:
            for sidx, step in enumerate(steps):
                if not isinstance(step, dict):
                    errors.append(f"outer_workflow_flow['execution_steps'][{sidx}] must be a dictionary")
                    continue
                for sfield in ["step_id", "operation", "target", "sequence_index", "arguments", "assigned_variable", "evidence_refs"]:
                    if sfield not in step:
                        errors.append(f"outer_workflow_flow['execution_steps'][{sidx}] missing field '{sfield}'")

                op = step.get("operation")
                if op is not None and op not in CONTROLLED_EXECUTION_OPERATIONS:
                    errors.append(
                        f"outer_workflow_flow['execution_steps'][{sidx}]['operation'] '{op}' is invalid; "
                        f"must be one of {sorted(CONTROLLED_EXECUTION_OPERATIONS)}"
                    )

                sid = step.get("step_id")
                if isinstance(sid, str):
                    if sid in execution_step_ids:
                        errors.append(f"Duplicate step_id '{sid}' in execution_steps")
                    execution_step_ids.add(sid)
                elif sid is not None:
                    errors.append(f"outer_workflow_flow['execution_steps'][{sidx}]['step_id'] must be a string")

                sindex = step.get("sequence_index")
                if not isinstance(sindex, int) or sindex < 0:
                    errors.append(f"outer_workflow_flow['execution_steps'][{sidx}]['sequence_index'] must be non-negative integer")

    # ------------------------------------------------------------------------
    # 7. orchestrator_process Validation
    # ------------------------------------------------------------------------
    orch = data.get("orchestrator_process")
    if not isinstance(orch, dict):
        errors.append("orchestrator_process must be a dictionary")
    else:
        for ofield in ["framework", "process_type", "delegation_enabled_agents", "manager",
                       "task_context_links", "async_tasks", "runtime_scheduling_observable"]:
            if ofield not in orch:
                errors.append(f"orchestrator_process missing required field '{ofield}'")

        ptype = orch.get("process_type")
        if not isinstance(ptype, dict):
            errors.append("orchestrator_process['process_type'] must be a dictionary")
        else:
            if "resolution_status" in ptype:
                pstatus = ptype["resolution_status"]
                if pstatus not in RESOLUTION_STATUSES:
                    errors.append(
                        f"orchestrator_process['process_type']['resolution_status'] '{pstatus}' is invalid; "
                        f"must be one of {sorted(RESOLUTION_STATUSES)}"
                    )

        mgr = orch.get("manager")
        if not isinstance(mgr, dict):
            errors.append("orchestrator_process['manager'] must be a dictionary")
        else:
            for mfield in ["manager_llm", "manager_agent"]:
                if mfield not in mgr:
                    errors.append(f"orchestrator_process['manager'] missing field '{mfield}'")

        if not isinstance(orch.get("runtime_scheduling_observable"), bool):
            errors.append("orchestrator_process['runtime_scheduling_observable'] must be a boolean")

    # ------------------------------------------------------------------------
    # 8. dependency_graph Validation & Graph Referential Integrity
    # ------------------------------------------------------------------------
    dep_graph = data.get("dependency_graph")
    graph_node_ids: Set[str] = set()

    if not isinstance(dep_graph, dict):
        errors.append("dependency_graph must be a dictionary")
    else:
        for gfield in ["nodes", "edges"]:
            if gfield not in dep_graph:
                errors.append(f"dependency_graph missing required field '{gfield}'")

        nodes = dep_graph.get("nodes")
        if not isinstance(nodes, list):
            errors.append("dependency_graph['nodes'] must be a list")
        else:
            for nidx, node in enumerate(nodes):
                if not isinstance(node, dict):
                    errors.append(f"dependency_graph['nodes'][{nidx}] must be a dictionary")
                    continue
                for nfield in ["id", "type", "source_file", "evidence_refs"]:
                    if nfield not in node:
                        errors.append(f"dependency_graph['nodes'][{nidx}] missing required field '{nfield}'")
                nid = node.get("id")
                if isinstance(nid, str):
                    if nid in graph_node_ids:
                        errors.append(f"Duplicate node id '{nid}' in dependency_graph['nodes']")
                    graph_node_ids.add(nid)
                elif nid is not None:
                    errors.append(f"dependency_graph['nodes'][{nidx}]['id'] must be a string")

                ntype = node.get("type")
                if ntype is not None and ntype not in CONTROLLED_NODE_TYPES:
                    errors.append(
                        f"dependency_graph['nodes'][{nidx}]['type'] '{ntype}' is invalid; "
                        f"must be one of {sorted(CONTROLLED_NODE_TYPES)}"
                    )

        edges = dep_graph.get("edges")
        if not isinstance(edges, list):
            errors.append("dependency_graph['edges'] must be a list")
        else:
            for eidx, edge in enumerate(edges):
                if not isinstance(edge, dict):
                    errors.append(f"dependency_graph['edges'][{eidx}] must be a dictionary")
                    continue
                for efield in ["source", "target", "relationship", "data_reference", "resolution_status", "evidence_refs"]:
                    if efield not in edge:
                        errors.append(f"dependency_graph['edges'][{eidx}] missing required field '{efield}'")

                src = edge.get("source")
                tgt = edge.get("target")

                # Graph referential integrity
                if isinstance(src, str) and src not in graph_node_ids:
                    errors.append(
                        f"dependency_graph['edges'][{eidx}] references unknown source node '{src}'"
                    )
                if isinstance(tgt, str) and tgt not in graph_node_ids:
                    errors.append(
                        f"dependency_graph['edges'][{eidx}] references unknown target node '{tgt}'"
                    )

                rel = edge.get("relationship")
                if isinstance(rel, str) and rel not in RELATIONSHIP_TYPES:
                    errors.append(
                        f"dependency_graph['edges'][{eidx}]['relationship'] '{rel}' is invalid; "
                        f"must be one of {sorted(RELATIONSHIP_TYPES)}"
                    )

                res_status = edge.get("resolution_status")
                if isinstance(res_status, str) and res_status not in RESOLUTION_STATUSES:
                    errors.append(
                        f"dependency_graph['edges'][{eidx}]['resolution_status'] '{res_status}' is invalid; "
                        f"must be one of {sorted(RESOLUTION_STATUSES)}"
                    )

    # ------------------------------------------------------------------------
    # 9. tool_and_helper_inventory & Runtime Tool Observability Constraint
    # ------------------------------------------------------------------------
    tool_inv = data.get("tool_and_helper_inventory")
    if not isinstance(tool_inv, dict):
        errors.append("tool_and_helper_inventory must be a dictionary")
    else:
        for tfield in ["registered_tools", "explicit_code_invocations", "helper_functions", "runtime_llm_tool_selection"]:
            if tfield not in tool_inv:
                errors.append(f"tool_and_helper_inventory missing required field '{tfield}'")

        llm_tool_sel = tool_inv.get("runtime_llm_tool_selection")
        if not isinstance(llm_tool_sel, dict):
            errors.append("tool_and_helper_inventory['runtime_llm_tool_selection'] must be a dictionary")
        else:
            if "observable" not in llm_tool_sel:
                errors.append("runtime_llm_tool_selection missing 'observable' field")
            elif not isinstance(llm_tool_sel["observable"], bool):
                errors.append("runtime_llm_tool_selection['observable'] must be a boolean")
            elif llm_tool_sel["observable"] is True:
                errors.append(
                    "Invalid workflow context: 'runtime_llm_tool_selection.observable' cannot be True. "
                    "Static AST analysis cannot observe dynamic runtime LLM tool selection."
                )

            if "reason" not in llm_tool_sel or not isinstance(llm_tool_sel["reason"], str):
                errors.append("runtime_llm_tool_selection missing or non-string 'reason'")

    # ------------------------------------------------------------------------
    # 10. evidence_and_provenance Validation
    # ------------------------------------------------------------------------
    evidence_prov = data.get("evidence_and_provenance")
    all_evidence_ids: Set[str] = set()

    if not isinstance(evidence_prov, dict):
        errors.append("evidence_and_provenance must be a dictionary")
    else:
        if "evidence" not in evidence_prov:
            errors.append("evidence_and_provenance missing required field 'evidence'")
        else:
            ev_list = evidence_prov.get("evidence")
            if not isinstance(ev_list, list):
                errors.append("evidence_and_provenance['evidence'] must be a list")
            else:
                for ev_idx, ev_entry in enumerate(ev_list):
                    if not isinstance(ev_entry, dict):
                        errors.append(f"evidence_and_provenance['evidence'][{ev_idx}] must be a dictionary")
                        continue
                    for ev_field in ["evidence_id", "source_file", "start_line", "end_line", "entity_id", "evidence_type"]:
                        if ev_field not in ev_entry:
                            errors.append(f"evidence_and_provenance['evidence'][{ev_idx}] missing field '{ev_field}'")
                    evid = ev_entry.get("evidence_id")
                    if isinstance(evid, str):
                        if evid in all_evidence_ids:
                            errors.append(f"Duplicate evidence_id '{evid}' in evidence_and_provenance['evidence']")
                        all_evidence_ids.add(evid)
                    elif evid is not None:
                        errors.append(f"evidence_and_provenance['evidence'][{ev_idx}]['evidence_id'] must be a string")

                    for line_prop in ["start_line", "end_line"]:
                        val = ev_entry.get(line_prop)
                        if not isinstance(val, int) or val < 1:
                            errors.append(f"evidence_and_provenance['evidence'][{ev_idx}]['{line_prop}'] must be positive integer")

    # ------------------------------------------------------------------------
    # 11. analysis_limitations Validation
    # ------------------------------------------------------------------------
    limits = data.get("analysis_limitations")
    if not isinstance(limits, dict):
        errors.append("analysis_limitations must be a dictionary")
    else:
        for lfield in ["unresolved_references", "dynamic_expressions", "unsupported_patterns", "static_analysis_limitations"]:
            if lfield not in limits:
                errors.append(f"analysis_limitations missing required field '{lfield}'")
            elif not isinstance(limits[lfield], list):
                errors.append(f"analysis_limitations['{lfield}'] must be a list")

    # ------------------------------------------------------------------------
    # 12. analysis_summary Validation
    # ------------------------------------------------------------------------
    summary = data.get("analysis_summary")
    if not isinstance(summary, dict):
        errors.append("analysis_summary must be a dictionary")
    else:
        for sfield in ["total_agents", "invoked_agents", "uninvoked_agents", "total_tasks",
                       "total_dependencies", "unresolved_relationships", "completeness", "output_metrics"]:
            if sfield not in summary:
                errors.append(f"analysis_summary missing required field '{sfield}'")

        metrics = summary.get("output_metrics")
        if not isinstance(metrics, dict):
            errors.append("analysis_summary['output_metrics'] must be a dictionary")
        else:
            for mfield in ["serialized_bytes", "estimated_tokens", "token_estimation_method"]:
                if mfield not in metrics:
                    errors.append(f"analysis_summary['output_metrics'] missing field '{mfield}'")
            if not isinstance(metrics.get("serialized_bytes"), int) or metrics["serialized_bytes"] < 0:
                errors.append("output_metrics['serialized_bytes'] must be non-negative integer")
            if not isinstance(metrics.get("estimated_tokens"), int) or metrics["estimated_tokens"] < 0:
                errors.append("output_metrics['estimated_tokens'] must be non-negative integer")
            if not isinstance(metrics.get("token_estimation_method"), str):
                errors.append("output_metrics['token_estimation_method'] must be string")

    # ------------------------------------------------------------------------
    # 13. Evidence Resolution Check (Strict Mode / Final Validation)
    # ------------------------------------------------------------------------
    if (final_validation or strict_evidence) and isinstance(evidence_prov, dict):
        # Gather all referenced evidence_ids
        referenced_evidence: List[tuple[str, str]] = []  # (location, evidence_id)

        # In agents
        if isinstance(agent_inv, dict) and isinstance(agent_inv.get("agents"), list):
            for a in agent_inv["agents"]:
                for ref in a.get("evidence_refs", []):
                    referenced_evidence.append((f"agent_inventory.agent '{a.get('agent_id')}'", ref))

        # In agent details
        if isinstance(agent_details, list):
            for ad in agent_details:
                for ref in ad.get("evidence_refs", []):
                    referenced_evidence.append((f"agent_details '{ad.get('agent_id')}'", ref))
                for t in ad.get("tasks", []):
                    for ref in t.get("evidence_refs", []):
                        referenced_evidence.append((f"task '{t.get('task_id')}'", ref))

        # In execution steps
        if isinstance(flow, dict) and isinstance(flow.get("execution_steps"), list):
            for st in flow["execution_steps"]:
                for ref in st.get("evidence_refs", []):
                    referenced_evidence.append((f"execution_step '{st.get('step_id')}'", ref))

        # In nodes and edges
        if isinstance(dep_graph, dict):
            for n in dep_graph.get("nodes", []):
                for ref in n.get("evidence_refs", []):
                    referenced_evidence.append((f"graph node '{n.get('id')}'", ref))
            for e in dep_graph.get("edges", []):
                for ref in e.get("evidence_refs", []):
                    referenced_evidence.append((f"graph edge '{e.get('source')}->{e.get('target')}'", ref))

        for location, ref in referenced_evidence:
            if ref not in all_evidence_ids:
                errors.append(f"Unresolved evidence reference '{ref}' in {location}")

    # ------------------------------------------------------------------------
    # 14. Cross-Section Consistency (Final Validation Mode)
    # ------------------------------------------------------------------------
    if final_validation and isinstance(metadata, dict) and metadata.get("analysis_status") == "completed":
        if isinstance(summary, dict) and isinstance(agent_inv, dict) and isinstance(dep_graph, dict):
            tot_agents = summary.get("total_agents", 0)
            inv_agents = summary.get("invoked_agents", 0)
            uninv_agents = summary.get("uninvoked_agents", 0)
            tot_tasks = summary.get("total_tasks", 0)
            tot_deps = summary.get("total_dependencies", 0)
            unres_rels = summary.get("unresolved_relationships", 0)

            id_agent_defs = agent_inv.get("identified_agent_definitions", 0)
            wf_inv_agents = agent_inv.get("workflow_invoked_agents", 0)
            uninv_defs = agent_inv.get("uninvoked_agent_definitions", 0)

            edges = dep_graph.get("edges", [])
            actual_edge_count = len(edges)
            actual_unresolved_edges = sum(
                1 for e in edges if isinstance(e, dict) and e.get("resolution_status") == "unresolved"
            )

            # Check cross-section rules
            if id_agent_defs != (wf_inv_agents + uninv_defs):
                errors.append(
                    f"Cross-section mismatch: identified_agent_definitions ({id_agent_defs}) != "
                    f"workflow_invoked_agents ({wf_inv_agents}) + uninvoked_agent_definitions ({uninv_defs})"
                )
            if tot_agents != id_agent_defs:
                errors.append(
                    f"Cross-section mismatch: summary total_agents ({tot_agents}) != "
                    f"identified_agent_definitions ({id_agent_defs})"
                )
            if inv_agents != wf_inv_agents:
                errors.append(
                    f"Cross-section mismatch: summary invoked_agents ({inv_agents}) != "
                    f"workflow_invoked_agents ({wf_inv_agents})"
                )
            if uninv_agents != uninv_defs:
                errors.append(
                    f"Cross-section mismatch: summary uninvoked_agents ({uninv_agents}) != "
                    f"uninvoked_agent_definitions ({uninv_defs})"
                )
            if tot_tasks != len(all_task_ids):
                errors.append(
                    f"Cross-section mismatch: summary total_tasks ({tot_tasks}) != "
                    f"unique extracted task IDs ({len(all_task_ids)})"
                )
            if tot_deps != actual_edge_count:
                errors.append(
                    f"Cross-section mismatch: summary total_dependencies ({tot_deps}) != "
                    f"dependency graph edge count ({actual_edge_count})"
                )
            if unres_rels != actual_unresolved_edges:
                errors.append(
                    f"Cross-section mismatch: summary unresolved_relationships ({unres_rels}) != "
                    f"unresolved edge count ({actual_unresolved_edges})"
                )

    # ------------------------------------------------------------------------
    # Final error collection evaluation
    # ------------------------------------------------------------------------
    if errors:
        raise ValidationError(
            f"WorkflowContext validation failed with {len(errors)} error(s)",
            errors=errors,
        )
