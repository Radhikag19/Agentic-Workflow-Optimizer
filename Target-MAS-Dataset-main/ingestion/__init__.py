"""
TraceProfiler AST Ingestion Pipeline Package.

Exposes the complete 5-module pipeline:
1. ast_parser: Safe AST parsing, symbol resolution, and literal evaluation.
2. agent_extractor: Internal agent, task, tool, and helper extraction.
3. workflow_analyzer: Outer workflow flow, orchestrator config, and dependency graph.
4. context_generator: Consolidation, evidence linking, and JSON serialization.
5. orchestrator: Batch coordination, workflow discovery, and execution reporting.
"""

from ingestion.ast_parser import (
    ASTParser,
    ParsedModule,
    SourceLocation,
    ImportReference,
    ImportItem,
    CallInfo,
    AssignmentInfo,
    FunctionDefInfo,
    ClassDefInfo,
    safe_eval_literal,
    parse_file,
    parse_source,
    default_ast_parser,
)

from ingestion.agent_extractor import (
    AgentExtractor,
    ExtractedAgent,
    ExtractedTask,
    ExtractedLLM,
)

from ingestion.workflow_analyzer import (
    WorkflowAnalyzer,
    WorkflowAnalysisResult,
)

from ingestion.context_generator import (
    ContextGenerator,
)

from ingestion.orchestrator import (
    IngestionOrchestrator,
    WorkflowExecutionReport,
    IngestionBatchSummary,
)

__all__ = [
    # ast_parser
    "ASTParser",
    "ParsedModule",
    "SourceLocation",
    "ImportReference",
    "ImportItem",
    "CallInfo",
    "AssignmentInfo",
    "FunctionDefInfo",
    "ClassDefInfo",
    "safe_eval_literal",
    "parse_file",
    "parse_source",
    "default_ast_parser",
    # agent_extractor
    "AgentExtractor",
    "ExtractedAgent",
    "ExtractedTask",
    "ExtractedLLM",
    # workflow_analyzer
    "WorkflowAnalyzer",
    "WorkflowAnalysisResult",
    # context_generator
    "ContextGenerator",
    # orchestrator
    "IngestionOrchestrator",
    "WorkflowExecutionReport",
    "IngestionBatchSummary",
]
