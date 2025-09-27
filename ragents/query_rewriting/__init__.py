"""Query rewriting module for RAGents.

This module provides automatic and interactive prompt optimization capabilities
inspired by DSPy's approach to prompt construction and optimization.
"""

from .base import QueryRewriter, RewriteStrategy, RewriteResult
from .strategies import (
    CoTRewriter,
    FewShotRewriter,
    ChainOfDensityRewriter,
    HypothesisRefinementRewriter,
    ContextualRewriter,
)
from .optimizer import PromptOptimizer, OptimizationConfig
from .interactive import InteractiveRewriter
from .evaluator import RewriteEvaluator, RewriteMetrics

__all__ = [
    "QueryRewriter",
    "RewriteStrategy",
    "RewriteResult",
    "CoTRewriter",
    "FewShotRewriter",
    "ChainOfDensityRewriter",
    "HypothesisRefinementRewriter",
    "ContextualRewriter",
    "PromptOptimizer",
    "OptimizationConfig",
    "InteractiveRewriter",
    "RewriteEvaluator",
    "RewriteMetrics",
]