"""Evaluation metrics and tools for reranking quality assessment."""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from .base import RerankingResult, RetrievedDocument


@dataclass
class RerankingMetrics:
    """Comprehensive metrics for reranking quality evaluation."""
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    map_score: float  # Mean Average Precision
    mrr_score: float  # Mean Reciprocal Rank
    reranking_effectiveness: float
    score_correlation: float
    processing_efficiency: float
    confidence_accuracy: float
    metadata: Dict[str, Any]


class RerankingEvaluator:
    """Comprehensive evaluator for reranking quality."""

    def __init__(self):
        self.evaluation_history: List[Dict[str, Any]] = []

    async def evaluate_reranking(
        self,
        result: RerankingResult,
        ground_truth_relevance: Optional[Dict[str, float]] = None,
        k_values: List[int] = None
    ) -> RerankingMetrics:
        """Evaluate a reranking result comprehensively."""

        k_values = k_values or [5, 10]
        k = max(k_values)  # Use the largest k for main evaluation

        # Calculate metrics
        precision_at_k = self._calculate_precision_at_k(result, k, ground_truth_relevance)
        recall_at_k = self._calculate_recall_at_k(result, k, ground_truth_relevance)
        ndcg_at_k = self._calculate_ndcg_at_k(result, k, ground_truth_relevance)
        map_score = self._calculate_map(result, ground_truth_relevance)
        mrr_score = self._calculate_mrr(result, ground_truth_relevance)
        reranking_effectiveness = self._calculate_reranking_effectiveness(result)
        score_correlation = self._calculate_score_correlation(result)
        processing_efficiency = self._calculate_processing_efficiency(result)
        confidence_accuracy = self._calculate_confidence_accuracy(result, ground_truth_relevance)

        metrics = RerankingMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            map_score=map_score,
            mrr_score=mrr_score,
            reranking_effectiveness=reranking_effectiveness,
            score_correlation=score_correlation,
            processing_efficiency=processing_efficiency,
            confidence_accuracy=confidence_accuracy,
            metadata={
                "k_value": k,
                "strategy": result.strategy.value,
                "document_count": len(result.reranked_documents),
                "evaluation_timestamp": datetime.now().isoformat(),
                "has_ground_truth": ground_truth_relevance is not None
            }
        )

        # Store evaluation in history
        self.evaluation_history.append({
            "result": result,
            "metrics": metrics,
            "ground_truth": ground_truth_relevance
        })

        return metrics

    def _calculate_precision_at_k(
        self,
        result: RerankingResult,
        k: int,
        ground_truth: Optional[Dict[str, float]]
    ) -> float:
        """Calculate Precision@K."""

        if not result.reranked_documents or not ground_truth:
            return 0.0

        top_k_docs = result.reranked_documents[:k]
        relevant_count = 0

        for doc in top_k_docs:
            doc_key = self._get_document_key(doc)
            if ground_truth.get(doc_key, 0.0) >= 0.5:  # Threshold for relevance
                relevant_count += 1

        return relevant_count / min(k, len(top_k_docs))

    def _calculate_recall_at_k(
        self,
        result: RerankingResult,
        k: int,
        ground_truth: Optional[Dict[str, float]]
    ) -> float:
        """Calculate Recall@K."""

        if not result.reranked_documents or not ground_truth:
            return 0.0

        # Count total relevant documents
        total_relevant = sum(1 for relevance in ground_truth.values() if relevance >= 0.5)

        if total_relevant == 0:
            return 0.0

        top_k_docs = result.reranked_documents[:k]
        relevant_retrieved = 0

        for doc in top_k_docs:
            doc_key = self._get_document_key(doc)
            if ground_truth.get(doc_key, 0.0) >= 0.5:
                relevant_retrieved += 1

        return relevant_retrieved / total_relevant

    def _calculate_ndcg_at_k(
        self,
        result: RerankingResult,
        k: int,
        ground_truth: Optional[Dict[str, float]]
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""

        if not result.reranked_documents or not ground_truth:
            return 0.0

        top_k_docs = result.reranked_documents[:k]

        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(top_k_docs):
            doc_key = self._get_document_key(doc)
            relevance = ground_truth.get(doc_key, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Calculate IDCG (Ideal DCG)
        ideal_relevances = sorted(ground_truth.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances):
            idcg += relevance / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_map(
        self,
        result: RerankingResult,
        ground_truth: Optional[Dict[str, float]]
    ) -> float:
        """Calculate Mean Average Precision."""

        if not result.reranked_documents or not ground_truth:
            return 0.0

        average_precisions = []
        relevant_count = 0

        for i, doc in enumerate(result.reranked_documents):
            doc_key = self._get_document_key(doc)
            if ground_truth.get(doc_key, 0.0) >= 0.5:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                average_precisions.append(precision_at_i)

        return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0

    def _calculate_mrr(
        self,
        result: RerankingResult,
        ground_truth: Optional[Dict[str, float]]
    ) -> float:
        """Calculate Mean Reciprocal Rank."""

        if not result.reranked_documents or not ground_truth:
            return 0.0

        for i, doc in enumerate(result.reranked_documents):
            doc_key = self._get_document_key(doc)
            if ground_truth.get(doc_key, 0.0) >= 0.5:
                return 1.0 / (i + 1)

        return 0.0

    def _calculate_reranking_effectiveness(self, result: RerankingResult) -> float:
        """Calculate how much reranking improved the ranking."""

        if not result.original_documents or not result.reranked_documents:
            return 0.0

        # Create mappings from content to original position
        original_positions = {doc.content: i for i, doc in enumerate(result.original_documents)}

        # Calculate rank changes
        position_improvements = []
        for new_pos, doc in enumerate(result.reranked_documents):
            original_pos = original_positions.get(doc.content, len(result.original_documents))
            improvement = original_pos - new_pos
            position_improvements.append(improvement)

        # Effectiveness is the average rank improvement
        avg_improvement = sum(position_improvements) / len(position_improvements)

        # Normalize by document count
        max_possible_improvement = len(result.original_documents) / 2
        effectiveness = max(0.0, avg_improvement / max_possible_improvement)

        return min(effectiveness, 1.0)

    def _calculate_score_correlation(self, result: RerankingResult) -> float:
        """Calculate correlation between original and reranked scores."""

        if len(result.reranked_documents) < 2:
            return 0.0

        # Get original scores in reranked order
        original_score_map = {doc.content: doc.similarity_score for doc in result.original_documents}
        reranked_original_scores = [
            original_score_map.get(doc.content, 0.0) for doc in result.reranked_documents
        ]
        reranked_scores = result.reranking_scores

        # Calculate Spearman correlation
        try:
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(reranked_original_scores, reranked_scores)
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except ImportError:
            # Fallback to simple correlation calculation
            return self._simple_correlation(reranked_original_scores, reranked_scores)

    def _simple_correlation(self, x: List[float], y: List[float]) -> float:
        """Simple correlation calculation fallback."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5

        return abs(numerator / denominator) if denominator != 0 else 0.0

    def _calculate_processing_efficiency(self, result: RerankingResult) -> float:
        """Calculate processing efficiency score."""

        # Efficiency based on processing time and document count
        docs_per_second = len(result.original_documents) / max(result.processing_time, 0.001)

        # Normalize (assume 50 docs/second is baseline efficient)
        efficiency = min(docs_per_second / 50.0, 1.0)

        return efficiency

    def _calculate_confidence_accuracy(
        self,
        result: RerankingResult,
        ground_truth: Optional[Dict[str, float]]
    ) -> float:
        """Calculate how accurate the confidence score is."""

        if not ground_truth:
            return 0.5  # Neutral when no ground truth

        # Calculate actual quality metrics
        actual_precision = self._calculate_precision_at_k(result, 5, ground_truth)

        # Compare with reported confidence
        confidence_error = abs(result.confidence_score - actual_precision)
        accuracy = 1.0 - confidence_error

        return max(0.0, accuracy)

    def _get_document_key(self, doc: RetrievedDocument) -> str:
        """Get a unique key for a document."""
        return doc.document_id or doc.content[:100]

    async def compare_strategies(
        self,
        results_by_strategy: Dict[str, List[RerankingResult]],
        ground_truth_data: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """Compare performance across different reranking strategies."""

        strategy_performance = {}

        for strategy_name, results in results_by_strategy.items():
            metrics_list = []
            for result in results:
                # Get ground truth for this specific result if available
                gt = ground_truth_data.get(result.query) if ground_truth_data else None
                metrics = await self.evaluate_reranking(result, gt)
                metrics_list.append(metrics)

            # Calculate aggregate metrics
            if metrics_list:
                avg_metrics = {
                    "precision_at_k": sum(m.precision_at_k for m in metrics_list) / len(metrics_list),
                    "recall_at_k": sum(m.recall_at_k for m in metrics_list) / len(metrics_list),
                    "ndcg_at_k": sum(m.ndcg_at_k for m in metrics_list) / len(metrics_list),
                    "map_score": sum(m.map_score for m in metrics_list) / len(metrics_list),
                    "mrr_score": sum(m.mrr_score for m in metrics_list) / len(metrics_list),
                    "reranking_effectiveness": sum(m.reranking_effectiveness for m in metrics_list) / len(metrics_list),
                    "processing_efficiency": sum(m.processing_efficiency for m in metrics_list) / len(metrics_list),
                    "confidence_accuracy": sum(m.confidence_accuracy for m in metrics_list) / len(metrics_list)
                }
                strategy_performance[strategy_name] = avg_metrics

        # Find best strategy for each metric
        best_strategies = {}
        for metric in ["precision_at_k", "recall_at_k", "ndcg_at_k", "processing_efficiency"]:
            if strategy_performance:
                best_strategy = max(strategy_performance.keys(),
                                  key=lambda s: strategy_performance[s][metric])
                best_strategies[metric] = best_strategy

        return {
            "strategy_performance": strategy_performance,
            "best_strategies": best_strategies,
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_evaluations": sum(len(results) for results in results_by_strategy.values())
        }

    def get_evaluation_analytics(self) -> Dict[str, Any]:
        """Get analytics from evaluation history."""

        if not self.evaluation_history:
            return {"total_evaluations": 0}

        total_evaluations = len(self.evaluation_history)
        strategy_performance = {}

        for eval_data in self.evaluation_history:
            strategy = eval_data["result"].strategy.value
            metrics = eval_data["metrics"]

            if strategy not in strategy_performance:
                strategy_performance[strategy] = []

            strategy_performance[strategy].append({
                "precision": metrics.precision_at_k,
                "recall": metrics.recall_at_k,
                "ndcg": metrics.ndcg_at_k,
                "effectiveness": metrics.reranking_effectiveness
            })

        # Calculate average performance per strategy
        avg_strategy_performance = {}
        for strategy, metrics_list in strategy_performance.items():
            avg_strategy_performance[strategy] = {
                "avg_precision": sum(m["precision"] for m in metrics_list) / len(metrics_list),
                "avg_recall": sum(m["recall"] for m in metrics_list) / len(metrics_list),
                "avg_ndcg": sum(m["ndcg"] for m in metrics_list) / len(metrics_list),
                "avg_effectiveness": sum(m["effectiveness"] for m in metrics_list) / len(metrics_list)
            }

        return {
            "total_evaluations": total_evaluations,
            "strategy_performance": avg_strategy_performance,
            "best_performing_strategy": max(avg_strategy_performance.keys(),
                                          key=lambda k: avg_strategy_performance[k]["avg_ndcg"]) if avg_strategy_performance else None,
            "recent_quality_trend": [eval_data["metrics"].ndcg_at_k
                                   for eval_data in self.evaluation_history[-10:]]
        }