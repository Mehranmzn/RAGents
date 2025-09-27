"""RAG evaluation orchestrator."""

import asyncio
import time
from typing import List, Optional

from ..llm.client import LLMClient
from ..rag.engine import RAGEngine
from .metrics import RAGMetric, create_all_metrics
from .types import (
    BatchEvaluationResult,
    EvaluationDataPoint,
    EvaluationMetrics,
    EvaluationResult,
)


class RAGEvaluator:
    """Main evaluator for RAG systems with comprehensive metrics."""

    def __init__(
        self,
        llm_client: LLMClient,
        rag_engine: Optional[RAGEngine] = None,
        metrics: Optional[List[RAGMetric]] = None,
    ):
        self.llm_client = llm_client
        self.rag_engine = rag_engine
        self.metrics = metrics or create_all_metrics(llm_client)

    async def evaluate_single(
        self,
        data_point: EvaluationDataPoint,
        generate_answer: bool = True,
    ) -> EvaluationResult:
        """Evaluate a single data point."""
        start_time = time.time()
        errors = []

        # Generate answer if needed and RAG engine is available
        if generate_answer and self.rag_engine and not data_point.answer:
            try:
                rag_response = await self.rag_engine.query(data_point.question)
                data_point.answer = rag_response.answer
                # Update contexts with retrieved information
                if rag_response.context_chunks:
                    data_point.contexts = [chunk.content for chunk in rag_response.context_chunks]
            except Exception as e:
                errors.append(f"Failed to generate answer: {str(e)}")

        # Evaluate metrics
        metrics = EvaluationMetrics()
        metric_details = {}

        for metric in self.metrics:
            try:
                score = await metric.evaluate(data_point)
                setattr(metrics, metric.name, score)
                metric_details[metric.name] = {"score": score}
            except Exception as e:
                errors.append(f"Failed to evaluate {metric.name}: {str(e)}")
                metric_details[metric.name] = {"error": str(e)}

        # Calculate overall score
        scores = []
        for metric_name in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]:
            score = getattr(metrics, metric_name)
            if score is not None:
                scores.append(score)

        metrics.overall_score = sum(scores) / len(scores) if scores else None
        metrics.metric_details = metric_details

        execution_time = time.time() - start_time

        return EvaluationResult(
            data_point=data_point,
            metrics=metrics,
            execution_time=execution_time,
            errors=errors,
        )

    async def evaluate_batch(
        self,
        data_points: List[EvaluationDataPoint],
        generate_answers: bool = True,
        max_concurrent: int = 5,
    ) -> BatchEvaluationResult:
        """Evaluate a batch of data points with controlled concurrency."""
        start_time = time.time()

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_with_semaphore(dp):
            async with semaphore:
                return await self.evaluate_single(dp, generate_answers)

        # Execute evaluations
        individual_results = await asyncio.gather(
            *[evaluate_with_semaphore(dp) for dp in data_points],
            return_exceptions=True
        )

        # Handle exceptions
        valid_results = []
        for result in individual_results:
            if isinstance(result, Exception):
                # Create error result
                error_result = EvaluationResult(
                    data_point=EvaluationDataPoint(question="", ground_truth="", contexts=[]),
                    metrics=EvaluationMetrics(),
                    execution_time=0.0,
                    errors=[str(result)]
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)

        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(valid_results)

        # Calculate success rate
        successful_evaluations = sum(1 for r in valid_results if not r.errors)
        success_rate = successful_evaluations / len(valid_results) if valid_results else 0.0

        total_execution_time = time.time() - start_time

        return BatchEvaluationResult(
            individual_results=valid_results,
            aggregate_metrics=aggregate_metrics,
            total_execution_time=total_execution_time,
            success_rate=success_rate,
        )

    def _calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> EvaluationMetrics:
        """Calculate aggregate metrics from individual results."""
        if not results:
            return EvaluationMetrics()

        # Collect all metric values
        metric_values = {
            "faithfulness": [],
            "answer_relevance": [],
            "context_precision": [],
            "context_recall": [],
        }

        for result in results:
            if not result.errors:  # Only include successful evaluations
                for metric_name in metric_values.keys():
                    value = getattr(result.metrics, metric_name)
                    if value is not None:
                        metric_values[metric_name].append(value)

        # Calculate means
        aggregate = EvaluationMetrics()
        for metric_name, values in metric_values.items():
            if values:
                mean_value = sum(values) / len(values)
                setattr(aggregate, metric_name, mean_value)

        # Calculate overall score
        overall_scores = []
        for metric_name in metric_values.keys():
            score = getattr(aggregate, metric_name)
            if score is not None:
                overall_scores.append(score)

        aggregate.overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else None

        return aggregate

    async def evaluate_rag_system(
        self,
        questions: List[str],
        ground_truths: List[str],
        max_concurrent: int = 3,
    ) -> BatchEvaluationResult:
        """Evaluate a RAG system end-to-end with questions and expected answers."""
        if not self.rag_engine:
            raise ValueError("RAG engine is required for end-to-end evaluation")

        if len(questions) != len(ground_truths):
            raise ValueError("Number of questions must match number of ground truths")

        # Create data points
        data_points = []
        for question, ground_truth in zip(questions, ground_truths):
            data_points.append(
                EvaluationDataPoint(
                    question=question,
                    ground_truth=ground_truth,
                    contexts=[],  # Will be populated by RAG system
                )
            )

        return await self.evaluate_batch(data_points, generate_answers=True, max_concurrent=max_concurrent)

    def add_metric(self, metric: RAGMetric) -> None:
        """Add a custom metric to the evaluator."""
        self.metrics.append(metric)

    def remove_metric(self, metric_name: str) -> bool:
        """Remove a metric by name."""
        initial_length = len(self.metrics)
        self.metrics = [m for m in self.metrics if m.name != metric_name]
        return len(self.metrics) < initial_length

    def get_metric_names(self) -> List[str]:
        """Get names of all configured metrics."""
        return [metric.name for metric in self.metrics]

    async def quick_eval(
        self,
        question: str,
        ground_truth: str,
        contexts: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Quick evaluation of a single question-answer pair."""
        data_point = EvaluationDataPoint(
            question=question,
            ground_truth=ground_truth,
            contexts=contexts or [],
        )

        return await self.evaluate_single(data_point, generate_answer=contexts is None)

    def create_evaluation_report(self, batch_result: BatchEvaluationResult) -> str:
        """Create a human-readable evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("RAG EVALUATION REPORT")
        report.append("=" * 60)

        # Summary
        report.append(f"\nSUMMARY:")
        report.append(f"  Total Data Points: {len(batch_result.individual_results)}")
        report.append(f"  Success Rate: {batch_result.success_rate:.2%}")
        report.append(f"  Total Execution Time: {batch_result.total_execution_time:.2f}s")

        # Aggregate Metrics
        report.append(f"\nAGGREGATE METRICS:")
        agg = batch_result.aggregate_metrics
        if agg.overall_score is not None:
            report.append(f"  Overall Score: {agg.overall_score:.3f}")
        if agg.faithfulness is not None:
            report.append(f"  Faithfulness: {agg.faithfulness:.3f}")
        if agg.answer_relevance is not None:
            report.append(f"  Answer Relevance: {agg.answer_relevance:.3f}")
        if agg.context_precision is not None:
            report.append(f"  Context Precision: {agg.context_precision:.3f}")
        if agg.context_recall is not None:
            report.append(f"  Context Recall: {agg.context_recall:.3f}")

        # Metric Summary Statistics
        summary = batch_result.get_metric_summary()
        if summary:
            report.append(f"\nMETRIC STATISTICS:")
            for metric_name, stats in summary.items():
                report.append(f"  {metric_name.title()}:")
                report.append(f"    Mean: {stats['mean']:.3f}")
                report.append(f"    Min: {stats['min']:.3f}")
                report.append(f"    Max: {stats['max']:.3f}")
                report.append(f"    Std: {stats['std']:.3f}")

        # Error Summary
        error_count = sum(1 for r in batch_result.individual_results if r.errors)
        if error_count > 0:
            report.append(f"\nERRORS:")
            report.append(f"  Failed Evaluations: {error_count}")

        report.append("\n" + "=" * 60)
        return "\n".join(report)