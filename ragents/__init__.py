"""
RAGents: Advanced Agentic RAG Framework

A comprehensive framework for building intelligent agents with multimodal RAG capabilities,
featuring type-safe LLM interactions and extensible processing pipelines.

Features:
- Multiple agent types: Decision Trees, Graph Planners, ReAct agents
- Pluggable vector stores: ChromaDB, Weaviate, pgvector, Elasticsearch
- Built-in evaluation with RAGAS-style metrics
- OpenInference observability and structured logging
- Elysia-compatible tool system
- DSPy-inspired query rewriting and prompt optimization
"""

# Core components
from .agents.base import Agent, AgentConfig
from .agents.decision_tree import DecisionTreeAgent
from .agents.graph_planner import GraphPlannerAgent
from .agents.react_agent import ReActAgent
from .rag.engine import RAGEngine
from .config.rag_config import RAGConfig
from .llm.client import LLMClient

# Vector stores
from .vector_stores import create_vector_store, VectorStoreConfig

# Evaluation
from .evaluation import RAGEvaluator, create_sample_dataset

# Observability
from .observability import RAGTracer, OpenInferenceIntegration, MetricsCollector

# Tools
from .tools import tool, ToolRegistry

# Query Rewriting
from .query_rewriting import (
    QueryRewriter,
    CoTRewriter,
    FewShotRewriter,
    ContextualRewriter,
    PromptOptimizer,
    InteractiveRewriter,
    RewriteEvaluator,
)

# Logical LLM
from .logical_llm import (
    LogicalReasoner,
    LogicalQuery,
    LogicalConstraint,
    QueryClarifier,
    ClarificationRequest,
    ClarificationResponse,
    SymbolicSolver,
    ConstraintEngine,
    LogicPattern,
    PatternMatcher,
    BuiltinPatterns,
)

# Reranking and Relevance Filtering
from .reranking import (
    Reranker,
    SemanticReranker,
    CrossEncoderReranker,
    HybridReranker,
    LLMReranker,
    AutocutFilter,
    RerankingEvaluator,
    RerankingConfig,
)

__version__ = "0.1.0"
__all__ = [
    # Agents
    "Agent",
    "AgentConfig",
    "DecisionTreeAgent",
    "GraphPlannerAgent",
    "ReActAgent",

    # RAG
    "RAGEngine",
    "RAGConfig",
    "LLMClient",

    # Vector stores
    "create_vector_store",
    "VectorStoreConfig",

    # Evaluation
    "RAGEvaluator",
    "create_sample_dataset",

    # Observability
    "RAGTracer",
    "OpenInferenceIntegration",
    "MetricsCollector",

    # Tools
    "tool",
    "ToolRegistry",

    # Query Rewriting
    "QueryRewriter",
    "CoTRewriter",
    "FewShotRewriter",
    "ContextualRewriter",
    "PromptOptimizer",
    "InteractiveRewriter",
    "RewriteEvaluator",

    # Logical LLM
    "LogicalReasoner",
    "LogicalQuery",
    "LogicalConstraint",
    "QueryClarifier",
    "ClarificationRequest",
    "ClarificationResponse",
    "SymbolicSolver",
    "ConstraintEngine",
    "LogicPattern",
    "PatternMatcher",
    "BuiltinPatterns",

    # Reranking and Relevance Filtering
    "Reranker",
    "SemanticReranker",
    "CrossEncoderReranker",
    "HybridReranker",
    "LLMReranker",
    "AutocutFilter",
    "RerankingEvaluator",
    "RerankingConfig",
]