"""RAG evaluation framework with RAGAS-style metrics."""

from .metrics import (
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    RAGMetric,
)
from .evaluator import RAGEvaluator
from .datasets import EvaluationDataset, create_sample_dataset
from .types import EvaluationResult, EvaluationMetrics

__all__ = [
    "RAGMetric",
    "AnswerRelevance",
    "ContextPrecision",
    "ContextRecall",
    "Faithfulness",
    "RAGEvaluator",
    "EvaluationDataset",
    "create_sample_dataset",
    "EvaluationResult",
    "EvaluationMetrics",
]