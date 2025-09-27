"""Types for RAG evaluation system."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvaluationDataPoint(BaseModel):
    """A single evaluation data point."""
    question: str
    ground_truth: str
    contexts: List[str]
    answer: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    """Collection of evaluation metrics."""
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    overall_score: Optional[float] = None

    # Individual metric details
    metric_details: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result of evaluating a single data point."""
    data_point: EvaluationDataPoint
    metrics: EvaluationMetrics
    execution_time: float
    errors: List[str] = Field(default_factory=list)


class BatchEvaluationResult(BaseModel):
    """Result of evaluating multiple data points."""
    individual_results: List[EvaluationResult]
    aggregate_metrics: EvaluationMetrics
    total_execution_time: float
    success_rate: float

    def get_metric_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each metric."""
        metrics = {}

        for metric_name in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]:
            values = []
            for result in self.individual_results:
                metric_value = getattr(result.metrics, metric_name)
                if metric_value is not None:
                    values.append(metric_value)

            if values:
                metrics[metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5
                }

        return metrics