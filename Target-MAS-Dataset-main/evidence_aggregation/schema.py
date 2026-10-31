"""
Unified Workflow Evidence Schema.

Defines the structure for workflow_evidence.json, which consolidates:
1. Declared workflow design (workflow.json)
2. Static AST implementation context (workflow_context.json)
3. Empirical runtime execution telemetry (30 live traces)
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class InputFilesMetadata(BaseModel):
    workflow_json: str
    workflow_context_json: str
    traces_directory: str
    trace_files: List[str] = Field(default_factory=list)


class EvidenceMetadata(BaseModel):
    workflow_id: str
    workflow_name: str
    evidence_schema_version: str = "1.0"
    generated_at: str
    traces_discovered: int
    traces_processed: int
    traces_failed: int
    input_files: InputFilesMetadata
    aggregation_status: str


class DeclaredAgent(BaseModel):
    agent_id: str
    name: Optional[str] = None
    role: Optional[str] = None
    responsibilities: Optional[str] = None
    declared_inputs: List[str] = Field(default_factory=list)
    declared_outputs: List[str] = Field(default_factory=list)
    declared_dependencies: List[str] = Field(default_factory=list)
    assigned_key_env: Optional[str] = None


class DeclaredDependency(BaseModel):
    source: str
    target: str
    relationship_type: str = "declared_dependency"


class DeclaredModelConfig(BaseModel):
    model: Optional[str] = None
    provider: Optional[str] = None
    assigned_groq_key_env: Optional[str] = None


class WorkflowDefinition(BaseModel):
    description: Optional[str] = None
    version: Optional[str] = None
    declared_agents: List[DeclaredAgent] = Field(default_factory=list)
    declared_dependencies: List[DeclaredDependency] = Field(default_factory=list)
    declared_execution_order: List[str] = Field(default_factory=list)
    declared_model_configuration: DeclaredModelConfig = Field(default_factory=DeclaredModelConfig)
    optimizer_annotations: Dict[str, Any] = Field(default_factory=dict)


class StaticTaskInfo(BaseModel):
    task_name: Optional[str] = None
    description: Optional[str] = None
    expected_output: Optional[str] = None
    tools: List[str] = Field(default_factory=list)


class StaticAgentContext(BaseModel):
    agent_id: str
    role: Optional[str] = None
    goal: Optional[str] = None
    backstory: Optional[str] = None
    tasks: List[StaticTaskInfo] = Field(default_factory=list)
    prompts: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    provider: Optional[str] = None
    resolution_status: Optional[str] = None
    registered_tools: List[str] = Field(default_factory=list)
    helper_functions: List[str] = Field(default_factory=list)
    source_file: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    evidence_refs: List[str] = Field(default_factory=list)


class StaticImplementationContext(BaseModel):
    source_files: List[str] = Field(default_factory=list)
    agents: List[StaticAgentContext] = Field(default_factory=list)
    invocation_relationships: List[Dict[str, Any]] = Field(default_factory=list)
    data_handoffs: List[Dict[str, Any]] = Field(default_factory=list)
    conditional_branches: List[Dict[str, Any]] = Field(default_factory=list)
    tools_and_helpers: Dict[str, Any] = Field(default_factory=dict)
    analysis_limitations: Dict[str, Any] = Field(default_factory=dict)


class LatencyStats(BaseModel):
    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    observation_count: int


class TokenStats(BaseModel):
    total_tokens: int
    mean_tokens: float
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    total_prompt_tokens: int
    mean_prompt_tokens: float
    total_completion_tokens: int
    mean_completion_tokens: float
    observation_count: int


class CostStats(BaseModel):
    total_cost_usd: float
    mean_cost_usd: float
    pricing_source: str
    observation_count: int


class AgentRuntimeMetrics(BaseModel):
    agent_id: str
    invocation_count: int
    success_count: int
    failure_count: int
    observed_execution_orders: List[int] = Field(default_factory=list)
    latency: LatencyStats
    tokens: TokenStats
    cost: Optional[Dict[str, Any]] = None
    assigned_keys: List[str] = Field(default_factory=list)
    sample_inputs: Optional[Any] = None
    sample_outputs: Optional[Any] = None
    error_count: int = 0


class RunSummaryRecord(BaseModel):
    run_id: str
    status: str
    duration_seconds: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    agent_count: int
    timestamp: str
    trace_file: str


class RuntimeObservations(BaseModel):
    total_runs: int
    successful_runs: int
    failed_runs: int
    observed_models: List[str] = Field(default_factory=list)
    key_distribution: Dict[str, int] = Field(default_factory=dict)
    workflow_latency: LatencyStats
    workflow_tokens: TokenStats
    workflow_cost: CostStats
    agent_metrics: Dict[str, AgentRuntimeMetrics] = Field(default_factory=dict)
    runs_summary: List[RunSummaryRecord] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    type: str  # workflow, agent, task, tool, function
    name: str
    source: str  # declared, statically_observed, runtime_observed, multi_source


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: str
    provenance: str  # declared, statically_observed, runtime_observed
    evidence_ref: Optional[Any] = None


class WorkflowGraph(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


class AgentReconciliation(BaseModel):
    agent_id: str
    declared: bool
    statically_observed: bool
    runtime_observed: bool
    status: str
    notes: Optional[str] = None


class DependencyReconciliation(BaseModel):
    source: str
    target: str
    declared: bool
    statically_observed: bool
    runtime_observed: bool
    status: str
    notes: Optional[str] = None


class ModelReconciliation(BaseModel):
    declared_model: Optional[str] = None
    statically_extracted_model: Optional[str] = None
    runtime_observed_model: Optional[str] = None
    status: str
    notes: Optional[str] = None


class ReconciliationAndLimitations(BaseModel):
    agent_reconciliation: List[AgentReconciliation] = Field(default_factory=list)
    dependency_reconciliation: List[DependencyReconciliation] = Field(default_factory=list)
    model_reconciliation: ModelReconciliation
    limitations: List[str] = Field(default_factory=list)


class WorkflowEvidenceDocument(BaseModel):
    metadata: EvidenceMetadata
    workflow_definition: WorkflowDefinition
    static_implementation_context: StaticImplementationContext
    runtime_observations: RuntimeObservations
    workflow_graph: WorkflowGraph
    reconciliation_and_limitations: ReconciliationAndLimitations

