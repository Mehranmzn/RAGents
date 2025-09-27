"""Evaluation metrics and tools for query rewriting quality assessment."""

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime

from pydantic import BaseModel

from ..llm.client import LLMClient
from .base import RewriteResult, RewriteStrategy


@dataclass
class RewriteMetrics:
    """Comprehensive metrics for rewrite quality evaluation."""
    clarity_score: float
    specificity_score: float
    completeness_score: float
    intent_preservation_score: float
    rag_optimization_score: float
    overall_quality_score: float
    metadata: Dict[str, Any]


class MetricConfig(BaseModel):
    """Configuration for metric calculation."""
    weight_clarity: float = 0.25
    weight_specificity: float = 0.20
    weight_completeness: float = 0.20
    weight_intent_preservation: float = 0.25
    weight_rag_optimization: float = 0.10
    use_llm_evaluation: bool = True
    evaluation_samples: int = 3


class RewriteEvaluator:
    """Comprehensive evaluator for query rewrite quality."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client
        self.evaluation_history: List[Dict[str, Any]] = []

    async def evaluate_rewrite(
        self,
        result: RewriteResult,
        config: MetricConfig = MetricConfig(),
        context: Optional[Dict[str, Any]] = None
    ) -> RewriteMetrics:
        """Evaluate a rewrite result comprehensively."""

        # Calculate individual metrics
        clarity = await self._evaluate_clarity(result, config)
        specificity = await self._evaluate_specificity(result, config)
        completeness = await self._evaluate_completeness(result, config)
        intent_preservation = await self._evaluate_intent_preservation(result, config)
        rag_optimization = await self._evaluate_rag_optimization(result, config)

        # Calculate weighted overall score
        overall_score = (
            clarity * config.weight_clarity +
            specificity * config.weight_specificity +
            completeness * config.weight_completeness +
            intent_preservation * config.weight_intent_preservation +
            rag_optimization * config.weight_rag_optimization
        )

        metrics = RewriteMetrics(
            clarity_score=clarity,
            specificity_score=specificity,
            completeness_score=completeness,
            intent_preservation_score=intent_preservation,
            rag_optimization_score=rag_optimization,
            overall_quality_score=overall_score,
            metadata={
                "strategy": result.strategy.value,
                "evaluation_timestamp": datetime.now().isoformat(),
                "config": config.dict(),
                "context_provided": bool(context)
            }
        )

        # Store evaluation in history
        self.evaluation_history.append({
            "result": result,
            "metrics": metrics,
            "context": context
        })

        return metrics

    async def batch_evaluate(
        self,
        results: List[RewriteResult],
        config: MetricConfig = MetricConfig()
    ) -> Dict[str, RewriteMetrics]:
        """Evaluate multiple rewrite results in batch."""

        evaluations = {}
        for i, result in enumerate(results):
            metrics = await self.evaluate_rewrite(result, config)
            evaluations[f"result_{i}"] = metrics

        return evaluations

    async def compare_strategies(
        self,
        results_by_strategy: Dict[RewriteStrategy, List[RewriteResult]],
        config: MetricConfig = MetricConfig()
    ) -> Dict[str, Any]:
        """Compare performance across different rewriting strategies."""

        strategy_performance = {}

        for strategy, results in results_by_strategy.items():
            metrics_list = []
            for result in results:
                metrics = await self.evaluate_rewrite(result, config)
                metrics_list.append(metrics)

            # Calculate aggregate metrics
            avg_metrics = self._calculate_average_metrics(metrics_list)
            strategy_performance[strategy.value] = avg_metrics

        # Find best strategy
        best_strategy = max(
            strategy_performance.keys(),
            key=lambda s: strategy_performance[s]["overall_quality_score"]
        )

        return {
            "strategy_performance": strategy_performance,
            "best_strategy": best_strategy,
            "performance_gap": strategy_performance[best_strategy]["overall_quality_score"] -
                            min(perf["overall_quality_score"] for perf in strategy_performance.values()),
            "evaluation_timestamp": datetime.now().isoformat()
        }

    async def _evaluate_clarity(
        self,
        result: RewriteResult,
        config: MetricConfig
    ) -> float:
        """Evaluate clarity of the rewritten query."""

        # Lexical metrics
        readability_score = self._calculate_readability(result.rewritten_query)
        structure_score = self._calculate_structure_score(result.rewritten_query)

        # LLM-based evaluation if available
        llm_clarity_score = 0.5  # Default
        if config.use_llm_evaluation and self.llm_client:
            llm_clarity_score = await self._llm_evaluate_clarity(result)

        # Combine scores
        clarity_score = (readability_score * 0.3 + structure_score * 0.3 + llm_clarity_score * 0.4)
        return min(max(clarity_score, 0.0), 1.0)

    async def _evaluate_specificity(
        self,
        result: RewriteResult,
        config: MetricConfig
    ) -> float:
        """Evaluate specificity and detail level of the rewritten query."""

        original_words = len(result.original_query.split())
        rewritten_words = len(result.rewritten_query.split())

        # Length-based specificity
        length_ratio = min(rewritten_words / max(original_words, 1), 3.0) / 3.0

        # Keyword-based specificity
        specificity_keywords = [
            "specifically", "exactly", "precisely", "detailed", "comprehensive",
            "include", "explain", "describe", "analyze", "compare", "contrast"
        ]
        keyword_score = sum(1 for word in specificity_keywords
                          if word in result.rewritten_query.lower()) / len(specificity_keywords)

        # Question complexity
        question_complexity = self._calculate_question_complexity(result.rewritten_query)

        specificity_score = (length_ratio * 0.4 + keyword_score * 0.3 + question_complexity * 0.3)
        return min(max(specificity_score, 0.0), 1.0)

    async def _evaluate_completeness(
        self,
        result: RewriteResult,
        config: MetricConfig
    ) -> float:
        """Evaluate completeness of the rewritten query."""

        # Check for comprehensive coverage indicators
        coverage_indicators = [
            "all aspects", "comprehensive", "complete", "thorough", "detailed",
            "including", "also", "additionally", "furthermore", "moreover"
        ]

        coverage_score = sum(1 for indicator in coverage_indicators
                           if indicator in result.rewritten_query.lower()) / len(coverage_indicators)

        # Multi-part question analysis
        multipart_score = self._analyze_multipart_structure(result.rewritten_query)

        # Context integration
        context_integration = 0.5  # Default
        if result.metadata.get("context_used"):
            context_integration = 0.8

        completeness_score = (coverage_score * 0.4 + multipart_score * 0.4 + context_integration * 0.2)
        return min(max(completeness_score, 0.0), 1.0)

    async def _evaluate_intent_preservation(
        self,
        result: RewriteResult,
        config: MetricConfig
    ) -> float:
        """Evaluate how well the original intent is preserved."""

        # Semantic similarity (simplified)
        semantic_similarity = self._calculate_semantic_similarity(
            result.original_query, result.rewritten_query
        )

        # Key concept preservation
        key_concepts_preserved = self._check_key_concepts_preservation(
            result.original_query, result.rewritten_query
        )

        # LLM-based evaluation if available
        llm_intent_score = 0.7  # Default
        if config.use_llm_evaluation and self.llm_client:
            llm_intent_score = await self._llm_evaluate_intent_preservation(result)

        intent_score = (semantic_similarity * 0.3 + key_concepts_preserved * 0.3 + llm_intent_score * 0.4)
        return min(max(intent_score, 0.0), 1.0)

    async def _evaluate_rag_optimization(
        self,
        result: RewriteResult,
        config: MetricConfig
    ) -> float:
        """Evaluate optimization for RAG systems."""

        rag_indicators = [
            "search", "find", "retrieve", "lookup", "information", "sources",
            "documents", "data", "evidence", "references", "based on"
        ]

        rag_optimization_score = sum(1 for indicator in rag_indicators
                                   if indicator in result.rewritten_query.lower()) / len(rag_indicators)

        # Query structure for RAG
        structure_bonus = 0.0
        if any(marker in result.rewritten_query for marker in ["What", "How", "Why", "When", "Where"]):
            structure_bonus += 0.2

        if "?" in result.rewritten_query:
            structure_bonus += 0.1

        total_rag_score = min(rag_optimization_score + structure_bonus, 1.0)
        return max(total_rag_score, 0.0)

    async def _llm_evaluate_clarity(self, result: RewriteResult) -> float:
        """Use LLM to evaluate clarity."""

        evaluation_prompt = f"""
        Evaluate the clarity of this rewritten query on a scale of 0.0 to 1.0:

        Original: {result.original_query}
        Rewritten: {result.rewritten_query}

        Consider:
        - Is the rewritten query easy to understand?
        - Are the instructions clear and unambiguous?
        - Is the language appropriate for the intended use?

        Provide only a numerical score (0.0 to 1.0):
        """

        try:
            response = await self.llm_client.complete(evaluation_prompt)
            score = float(re.search(r'0\.\d+|1\.0', response).group())
            return min(max(score, 0.0), 1.0)
        except (ValueError, AttributeError):
            return 0.5  # Default if parsing fails

    async def _llm_evaluate_intent_preservation(self, result: RewriteResult) -> float:
        """Use LLM to evaluate intent preservation."""

        evaluation_prompt = f"""
        Evaluate how well the rewritten query preserves the original intent on a scale of 0.0 to 1.0:

        Original: {result.original_query}
        Rewritten: {result.rewritten_query}

        Consider:
        - Does the rewritten query seek the same information?
        - Are the core objectives maintained?
        - Is the scope and focus preserved?

        Provide only a numerical score (0.0 to 1.0):
        """

        try:
            response = await self.llm_client.complete(evaluation_prompt)
            score = float(re.search(r'0\.\d+|1\.0', response).group())
            return min(max(score, 0.0), 1.0)
        except (ValueError, AttributeError):
            return 0.7  # Default if parsing fails

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified)."""

        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?') + 1

        if sentences == 0:
            return 0.5

        avg_words_per_sentence = len(words) / sentences

        # Prefer moderate sentence length (10-20 words)
        if 10 <= avg_words_per_sentence <= 20:
            readability = 1.0
        elif avg_words_per_sentence < 10:
            readability = avg_words_per_sentence / 10
        else:
            readability = max(0.0, 1.0 - (avg_words_per_sentence - 20) / 20)

        return readability

    def _calculate_structure_score(self, text: str) -> float:
        """Calculate structure quality score."""

        structure_elements = 0

        # Check for organizational elements
        if any(marker in text for marker in ["first", "second", "then", "next", "finally"]):
            structure_elements += 1

        if any(marker in text for marker in ["1.", "2.", "•", "-"]):
            structure_elements += 1

        if any(marker in text for marker in ["including", "such as", "for example"]):
            structure_elements += 1

        if text.count('?') > 0:
            structure_elements += 1

        return min(structure_elements / 4, 1.0)

    def _calculate_question_complexity(self, text: str) -> float:
        """Calculate question complexity score."""

        complexity_indicators = [
            "how", "why", "what", "when", "where", "which", "analyze", "compare",
            "evaluate", "explain", "describe", "discuss", "examine"
        ]

        complexity_score = sum(1 for indicator in complexity_indicators
                             if indicator in text.lower()) / len(complexity_indicators)

        return min(complexity_score, 1.0)

    def _analyze_multipart_structure(self, text: str) -> float:
        """Analyze multipart question structure."""

        multipart_indicators = [
            "and", "also", "additionally", "furthermore", "moreover",
            "including", "as well as", "; also", "what about"
        ]

        multipart_score = sum(1 for indicator in multipart_indicators
                            if indicator in text.lower()) / len(multipart_indicators)

        return min(multipart_score, 1.0)

    def _calculate_semantic_similarity(self, original: str, rewritten: str) -> float:
        """Calculate semantic similarity (simplified)."""

        original_words = set(original.lower().split())
        rewritten_words = set(rewritten.lower().split())

        if not original_words:
            return 0.0

        intersection = original_words.intersection(rewritten_words)
        similarity = len(intersection) / len(original_words)

        return min(similarity, 1.0)

    def _check_key_concepts_preservation(self, original: str, rewritten: str) -> float:
        """Check if key concepts are preserved."""

        # Extract potential key concepts (nouns and important adjectives)
        import re

        # Simple concept extraction (in real implementation, use NLP)
        concept_pattern = r'\b[A-Z][a-z]+|\b(?:machine learning|artificial intelligence|database|algorithm)\b'
        original_concepts = set(re.findall(concept_pattern, original, re.IGNORECASE))
        rewritten_concepts = set(re.findall(concept_pattern, rewritten, re.IGNORECASE))

        if not original_concepts:
            return 1.0

        preserved_concepts = original_concepts.intersection(rewritten_concepts)
        preservation_ratio = len(preserved_concepts) / len(original_concepts)

        return preservation_ratio

    def _calculate_average_metrics(self, metrics_list: List[RewriteMetrics]) -> Dict[str, float]:
        """Calculate average metrics from a list of metric objects."""

        if not metrics_list:
            return {}

        return {
            "clarity_score": sum(m.clarity_score for m in metrics_list) / len(metrics_list),
            "specificity_score": sum(m.specificity_score for m in metrics_list) / len(metrics_list),
            "completeness_score": sum(m.completeness_score for m in metrics_list) / len(metrics_list),
            "intent_preservation_score": sum(m.intent_preservation_score for m in metrics_list) / len(metrics_list),
            "rag_optimization_score": sum(m.rag_optimization_score for m in metrics_list) / len(metrics_list),
            "overall_quality_score": sum(m.overall_quality_score for m in metrics_list) / len(metrics_list)
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

            strategy_performance[strategy].append(metrics.overall_quality_score)

        # Calculate average performance per strategy
        avg_strategy_performance = {}
        for strategy, scores in strategy_performance.items():
            avg_strategy_performance[strategy] = sum(scores) / len(scores)

        return {
            "total_evaluations": total_evaluations,
            "average_strategy_performance": avg_strategy_performance,
            "best_performing_strategy": max(avg_strategy_performance.keys(),
                                          key=lambda k: avg_strategy_performance[k]) if avg_strategy_performance else None,
            "overall_quality_trend": [eval_data["metrics"].overall_quality_score
                                    for eval_data in self.evaluation_history[-10:]]
        }