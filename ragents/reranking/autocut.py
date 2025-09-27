"""Autocut algorithm for filtering irrelevant information from retrieval results."""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import statistics

from .base import RetrievedDocument


class CutoffStrategy(Enum):
    """Strategies for determining cutoff points."""
    PERCENTILE = "percentile"
    STANDARD_DEVIATION = "standard_deviation"
    ELBOW_METHOD = "elbow_method"
    GRADIENT_CHANGE = "gradient_change"
    ZSCORE = "zscore"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"


@dataclass
class CutoffResult:
    """Result of cutoff detection."""
    cutoff_index: int
    cutoff_score: float
    strategy: CutoffStrategy
    confidence: float
    removed_count: int
    kept_count: int
    score_gap: float
    metadata: Dict[str, Any]


class AutocutFilter:
    """Autocut algorithm for filtering irrelevant retrieved documents."""

    def __init__(self, strategy: CutoffStrategy = CutoffStrategy.ADAPTIVE_THRESHOLD):
        self.strategy = strategy
        self.cutoff_history: List[CutoffResult] = []

    def filter_documents(
        self,
        documents: List[RetrievedDocument],
        strategy: Optional[CutoffStrategy] = None,
        **kwargs
    ) -> Tuple[List[RetrievedDocument], CutoffResult]:
        """Filter documents using the Autocut algorithm."""

        if not documents:
            return documents, CutoffResult(
                cutoff_index=0,
                cutoff_score=0.0,
                strategy=strategy or self.strategy,
                confidence=0.0,
                removed_count=0,
                kept_count=0,
                score_gap=0.0,
                metadata={}
            )

        # Sort documents by similarity score (descending)
        sorted_docs = sorted(documents, key=lambda x: x.similarity_score, reverse=True)
        scores = [doc.similarity_score for doc in sorted_docs]

        # Apply cutoff strategy
        cutoff_strategy = strategy or self.strategy
        cutoff_result = self._apply_cutoff_strategy(scores, cutoff_strategy, **kwargs)

        # Filter documents based on cutoff
        filtered_docs = sorted_docs[:cutoff_result.cutoff_index + 1]

        # Update result with actual counts
        cutoff_result.kept_count = len(filtered_docs)
        cutoff_result.removed_count = len(documents) - len(filtered_docs)

        # Add to history
        self.cutoff_history.append(cutoff_result)

        return filtered_docs, cutoff_result

    def _apply_cutoff_strategy(
        self,
        scores: List[float],
        strategy: CutoffStrategy,
        **kwargs
    ) -> CutoffResult:
        """Apply the specified cutoff strategy."""

        if strategy == CutoffStrategy.PERCENTILE:
            return self._percentile_cutoff(scores, **kwargs)
        elif strategy == CutoffStrategy.STANDARD_DEVIATION:
            return self._std_deviation_cutoff(scores, **kwargs)
        elif strategy == CutoffStrategy.ELBOW_METHOD:
            return self._elbow_method_cutoff(scores, **kwargs)
        elif strategy == CutoffStrategy.GRADIENT_CHANGE:
            return self._gradient_change_cutoff(scores, **kwargs)
        elif strategy == CutoffStrategy.ZSCORE:
            return self._zscore_cutoff(scores, **kwargs)
        elif strategy == CutoffStrategy.ADAPTIVE_THRESHOLD:
            return self._adaptive_threshold_cutoff(scores, **kwargs)
        else:
            raise ValueError(f"Unknown cutoff strategy: {strategy}")

    def _percentile_cutoff(self, scores: List[float], percentile: float = 0.6, **kwargs) -> CutoffResult:
        """Cut off at a specified percentile of the score distribution."""

        if len(scores) <= 1:
            return self._create_cutoff_result(0, scores[0] if scores else 0.0, CutoffStrategy.PERCENTILE, 1.0)

        threshold = np.percentile(scores, percentile * 100)
        cutoff_index = 0

        for i, score in enumerate(scores):
            if score >= threshold:
                cutoff_index = i
            else:
                break

        confidence = min(1.0, (scores[0] - threshold) / max(scores[0], 0.001))
        score_gap = scores[cutoff_index] - scores[min(cutoff_index + 1, len(scores) - 1)]

        return self._create_cutoff_result(
            cutoff_index,
            scores[cutoff_index],
            CutoffStrategy.PERCENTILE,
            confidence,
            score_gap=score_gap,
            metadata={"percentile": percentile, "threshold": threshold}
        )

    def _std_deviation_cutoff(self, scores: List[float], std_multiplier: float = 1.0, **kwargs) -> CutoffResult:
        """Cut off based on standard deviation from mean."""

        if len(scores) <= 2:
            return self._create_cutoff_result(0, scores[0] if scores else 0.0, CutoffStrategy.STANDARD_DEVIATION, 1.0)

        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores)
        threshold = mean_score - (std_multiplier * std_score)

        cutoff_index = 0
        for i, score in enumerate(scores):
            if score >= threshold:
                cutoff_index = i
            else:
                break

        confidence = min(1.0, std_score / max(mean_score, 0.001))
        score_gap = scores[cutoff_index] - scores[min(cutoff_index + 1, len(scores) - 1)]

        return self._create_cutoff_result(
            cutoff_index,
            scores[cutoff_index],
            CutoffStrategy.STANDARD_DEVIATION,
            confidence,
            score_gap=score_gap,
            metadata={"mean": mean_score, "std": std_score, "threshold": threshold}
        )

    def _elbow_method_cutoff(self, scores: List[float], **kwargs) -> CutoffResult:
        """Use elbow method to find the optimal cutoff point."""

        if len(scores) <= 2:
            return self._create_cutoff_result(0, scores[0] if scores else 0.0, CutoffStrategy.ELBOW_METHOD, 1.0)

        # Calculate distances from line connecting first and last points
        distances = []
        first_point = (0, scores[0])
        last_point = (len(scores) - 1, scores[-1])

        for i, score in enumerate(scores):
            # Distance from point to line
            distance = abs(
                (last_point[1] - first_point[1]) * i -
                (last_point[0] - first_point[0]) * score +
                last_point[0] * first_point[1] -
                last_point[1] * first_point[0]
            ) / np.sqrt(
                (last_point[1] - first_point[1]) ** 2 +
                (last_point[0] - first_point[0]) ** 2
            )
            distances.append(distance)

        # Find elbow (maximum distance)
        cutoff_index = distances.index(max(distances))
        confidence = max(distances) / (scores[0] - scores[-1] + 0.001)
        score_gap = scores[cutoff_index] - scores[min(cutoff_index + 1, len(scores) - 1)]

        return self._create_cutoff_result(
            cutoff_index,
            scores[cutoff_index],
            CutoffStrategy.ELBOW_METHOD,
            confidence,
            score_gap=score_gap,
            metadata={"max_distance": max(distances), "distances": distances[:10]}  # Store first 10 for analysis
        )

    def _gradient_change_cutoff(self, scores: List[float], threshold_factor: float = 2.0, **kwargs) -> CutoffResult:
        """Cut off where gradient changes significantly."""

        if len(scores) <= 3:
            return self._create_cutoff_result(0, scores[0] if scores else 0.0, CutoffStrategy.GRADIENT_CHANGE, 1.0)

        # Calculate gradients (differences)
        gradients = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]

        if not gradients:
            return self._create_cutoff_result(0, scores[0], CutoffStrategy.GRADIENT_CHANGE, 1.0)

        # Find where gradient increases significantly
        mean_gradient = statistics.mean(gradients)
        cutoff_index = 0

        for i, gradient in enumerate(gradients):
            if gradient <= mean_gradient * threshold_factor:
                cutoff_index = i
            else:
                break

        confidence = min(1.0, gradients[cutoff_index] / max(mean_gradient, 0.001))
        score_gap = gradients[cutoff_index] if cutoff_index < len(gradients) else 0.0

        return self._create_cutoff_result(
            cutoff_index,
            scores[cutoff_index],
            CutoffStrategy.GRADIENT_CHANGE,
            confidence,
            score_gap=score_gap,
            metadata={"mean_gradient": mean_gradient, "threshold_factor": threshold_factor}
        )

    def _zscore_cutoff(self, scores: List[float], zscore_threshold: float = -1.0, **kwargs) -> CutoffResult:
        """Cut off based on Z-score analysis."""

        if len(scores) <= 2:
            return self._create_cutoff_result(0, scores[0] if scores else 0.0, CutoffStrategy.ZSCORE, 1.0)

        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.001

        cutoff_index = 0
        for i, score in enumerate(scores):
            zscore = (score - mean_score) / std_score
            if zscore >= zscore_threshold:
                cutoff_index = i
            else:
                break

        confidence = min(1.0, std_score / max(mean_score, 0.001))
        score_gap = scores[cutoff_index] - scores[min(cutoff_index + 1, len(scores) - 1)]

        return self._create_cutoff_result(
            cutoff_index,
            scores[cutoff_index],
            CutoffStrategy.ZSCORE,
            confidence,
            score_gap=score_gap,
            metadata={"mean": mean_score, "std": std_score, "zscore_threshold": zscore_threshold}
        )

    def _adaptive_threshold_cutoff(self, scores: List[float], **kwargs) -> CutoffResult:
        """Adaptive threshold combining multiple strategies."""

        if len(scores) <= 2:
            return self._create_cutoff_result(0, scores[0] if scores else 0.0, CutoffStrategy.ADAPTIVE_THRESHOLD, 1.0)

        # Apply multiple strategies and combine results
        strategies_results = []

        try:
            percentile_result = self._percentile_cutoff(scores, percentile=0.7)
            strategies_results.append(("percentile", percentile_result))
        except Exception:
            pass

        try:
            elbow_result = self._elbow_method_cutoff(scores)
            strategies_results.append(("elbow", elbow_result))
        except Exception:
            pass

        try:
            gradient_result = self._gradient_change_cutoff(scores)
            strategies_results.append(("gradient", gradient_result))
        except Exception:
            pass

        if not strategies_results:
            # Fallback to simple percentile
            return self._percentile_cutoff(scores, percentile=0.6)

        # Weight strategies based on confidence and score distribution
        weighted_cutoffs = []
        total_weight = 0

        for strategy_name, result in strategies_results:
            weight = result.confidence
            weighted_cutoffs.append(result.cutoff_index * weight)
            total_weight += weight

        if total_weight == 0:
            adaptive_cutoff = 0
        else:
            adaptive_cutoff = int(sum(weighted_cutoffs) / total_weight)

        # Ensure cutoff is within bounds
        adaptive_cutoff = max(0, min(adaptive_cutoff, len(scores) - 1))

        confidence = total_weight / len(strategies_results) if strategies_results else 0.0
        score_gap = scores[adaptive_cutoff] - scores[min(adaptive_cutoff + 1, len(scores) - 1)]

        return self._create_cutoff_result(
            adaptive_cutoff,
            scores[adaptive_cutoff],
            CutoffStrategy.ADAPTIVE_THRESHOLD,
            confidence,
            score_gap=score_gap,
            metadata={
                "strategies_used": [name for name, _ in strategies_results],
                "individual_cutoffs": {name: result.cutoff_index for name, result in strategies_results}
            }
        )

    def _create_cutoff_result(
        self,
        cutoff_index: int,
        cutoff_score: float,
        strategy: CutoffStrategy,
        confidence: float,
        score_gap: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CutoffResult:
        """Create a cutoff result object."""

        return CutoffResult(
            cutoff_index=cutoff_index,
            cutoff_score=cutoff_score,
            strategy=strategy,
            confidence=confidence,
            removed_count=0,  # Will be updated later
            kept_count=0,     # Will be updated later
            score_gap=score_gap,
            metadata=metadata or {}
        )

    def get_optimization_suggestions(self) -> Dict[str, Any]:
        """Get suggestions for optimizing cutoff parameters based on history."""

        if len(self.cutoff_history) < 5:
            return {"message": "Not enough history for optimization suggestions"}

        recent_results = self.cutoff_history[-10:]

        # Analysis metrics
        avg_confidence = sum(r.confidence for r in recent_results) / len(recent_results)
        avg_removed_ratio = sum(r.removed_count / (r.removed_count + r.kept_count)
                              for r in recent_results if (r.removed_count + r.kept_count) > 0) / len(recent_results)

        suggestions = {
            "average_confidence": avg_confidence,
            "average_removed_ratio": avg_removed_ratio,
            "recommendations": []
        }

        if avg_confidence < 0.5:
            suggestions["recommendations"].append("Consider using adaptive threshold strategy for better confidence")

        if avg_removed_ratio > 0.8:
            suggestions["recommendations"].append("Cutoff might be too aggressive, consider lowering thresholds")
        elif avg_removed_ratio < 0.2:
            suggestions["recommendations"].append("Cutoff might be too lenient, consider raising thresholds")

        return suggestions

    def get_cutoff_analytics(self) -> Dict[str, Any]:
        """Get analytics from cutoff history."""

        if not self.cutoff_history:
            return {"total_cutoffs": 0}

        strategy_performance = {}
        for result in self.cutoff_history:
            strategy = result.strategy.value
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(result.confidence)

        # Calculate average performance per strategy
        avg_strategy_performance = {}
        for strategy, confidences in strategy_performance.items():
            avg_strategy_performance[strategy] = sum(confidences) / len(confidences)

        return {
            "total_cutoffs": len(self.cutoff_history),
            "strategy_performance": avg_strategy_performance,
            "best_strategy": max(avg_strategy_performance.keys(),
                               key=lambda k: avg_strategy_performance[k]) if avg_strategy_performance else None,
            "average_removed_ratio": sum(r.removed_count / max(r.removed_count + r.kept_count, 1)
                                       for r in self.cutoff_history) / len(self.cutoff_history),
            "recent_confidence_trend": [r.confidence for r in self.cutoff_history[-5:]]
        }