"""
Evidence Aggregation Module for TraceProfiler.

Consolidates:
1. Declared workflow design (workflow.json)
2. Static AST implementation context (workflow_context.json)
3. Empirical runtime traces (live_*.json)
into unified workflow_evidence.json documents.
"""

from evidence_aggregation.schema import WorkflowEvidenceDocument
from evidence_aggregation.aggregator import EvidenceAggregator
from evidence_aggregation.validators import EvidenceValidator

__all__ = ["WorkflowEvidenceDocument", "EvidenceAggregator", "EvidenceValidator"]

