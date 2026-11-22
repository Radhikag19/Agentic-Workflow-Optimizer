"""
Optimization Agent Package.

Provides the Autonomous Optimization Discovery Agent using LangGraph, LangChain,
and the TraceProfiler MCP Evidence Server.
"""

from optimization_agent.agent import OptimizationDiscoveryAgent, build_optimization_graph
from optimization_agent.key_manager import GroqKeyManager, get_default_key_manager, mask_key
from optimization_agent.state import OptimizationAgentState, OptimizationCandidate
from optimization_agent.model_selection import (
    run_model_selection,
    build_model_selection_graph,
    AWOModelSelectionState,
)

__all__ = [
    "OptimizationDiscoveryAgent",
    "build_optimization_graph",
    "OptimizationAgentState",
    "OptimizationCandidate",
    "GroqKeyManager",
    "get_default_key_manager",
    "mask_key",
    "run_model_selection",
    "build_model_selection_graph",
    "AWOModelSelectionState",
]
