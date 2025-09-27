"""Interactive prompt optimization tools for real-time experimentation."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime

from ..llm.client import LLMClient
from .base import QueryRewriter, RewriteResult, RewriteStrategy, PromptTemplate
from .optimizer import PromptOptimizer, OptimizationConfig
from .strategies import CoTRewriter, FewShotRewriter, ContextualRewriter


class InteractiveRewriter:
    """Interactive query rewriting system for real-time optimization."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.available_strategies = {
            RewriteStrategy.CHAIN_OF_THOUGHT: CoTRewriter(llm_client),
            RewriteStrategy.FEW_SHOT: FewShotRewriter(llm_client),
            RewriteStrategy.CONTEXTUAL: ContextualRewriter(llm_client),
        }
        self.optimizer = PromptOptimizer(llm_client)
        self.session_history: List[Dict[str, Any]] = []

    async def interactive_rewrite_session(
        self,
        initial_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Start an interactive rewriting session."""

        session = {
            "session_id": f"session_{datetime.now().isoformat()}",
            "initial_query": initial_query,
            "context": context or {},
            "iterations": [],
            "best_rewrite": None,
            "strategies_tested": [],
            "user_feedback": []
        }

        print(f"🎯 Starting Interactive Rewriting Session")
        print(f"Original Query: {initial_query}")
        print("-" * 60)

        # Test multiple strategies
        for strategy, rewriter in self.available_strategies.items():
            print(f"\n🔧 Testing {strategy.value}...")

            result = await rewriter.rewrite(initial_query, context)

            iteration = {
                "strategy": strategy.value,
                "result": result,
                "timestamp": datetime.now()
            }

            session["iterations"].append(iteration)
            session["strategies_tested"].append(strategy.value)

            print(f"📝 Rewritten Query: {result.rewritten_query}")
            print(f"📊 Confidence: {result.confidence_score:.2f}")

            # Simulate user feedback (in real implementation, this would be interactive)
            feedback = await self._get_user_feedback(result)
            iteration["user_feedback"] = feedback
            session["user_feedback"].append(feedback)

        # Find best rewrite based on feedback
        best_iteration = max(session["iterations"], key=lambda x: x["user_feedback"]["rating"])
        session["best_rewrite"] = best_iteration["result"]

        print(f"\n🏆 Best Rewrite: {session['best_rewrite'].rewritten_query}")
        print(f"📈 Strategy: {best_iteration['strategy']}")

        self.session_history.append(session)
        return session

    async def adaptive_rewrite(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        performance_feedback: Optional[Dict[str, float]] = None
    ) -> RewriteResult:
        """Adaptively select the best rewriting strategy based on context and feedback."""

        # Analyze query characteristics
        query_analysis = await self._analyze_query_characteristics(query, context)

        # Select strategy based on analysis
        best_strategy = self._select_optimal_strategy(query_analysis, performance_feedback)

        # Apply selected strategy
        rewriter = self.available_strategies[best_strategy]
        result = await rewriter.rewrite(query, context)

        # Enhance with adaptive optimization
        if performance_feedback and len(self.session_history) > 3:
            result = await self._apply_learned_optimizations(result, performance_feedback)

        return result

    async def collaborative_rewrite(
        self,
        query: str,
        strategies: List[RewriteStrategy],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, RewriteResult]:
        """Use multiple strategies collaboratively and combine their strengths."""

        results = {}

        # Apply each strategy
        for strategy in strategies:
            if strategy in self.available_strategies:
                rewriter = self.available_strategies[strategy]
                result = await rewriter.rewrite(query, context)
                results[strategy.value] = result

        # Create a collaborative rewrite
        if len(results) > 1:
            collaborative_result = await self._combine_strategies(query, results, context)
            results["collaborative"] = collaborative_result

        return results

    async def real_time_optimization(
        self,
        query: str,
        evaluation_function: Callable,
        max_iterations: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[RewriteResult, Dict[str, Any]]:
        """Real-time optimization with immediate feedback."""

        optimization_log = {
            "iterations": [],
            "improvements": [],
            "strategy_performance": {}
        }

        best_result = None
        best_score = float('-inf')

        for iteration in range(max_iterations):
            print(f"\n🔄 Optimization Iteration {iteration + 1}")

            # Try different strategies
            for strategy, rewriter in self.available_strategies.items():
                result = await rewriter.rewrite(query, context)

                # Evaluate result
                score = await evaluation_function(query, result.rewritten_query)

                iteration_data = {
                    "iteration": iteration,
                    "strategy": strategy.value,
                    "score": score,
                    "query": result.rewritten_query
                }

                optimization_log["iterations"].append(iteration_data)

                # Update strategy performance
                if strategy.value not in optimization_log["strategy_performance"]:
                    optimization_log["strategy_performance"][strategy.value] = []
                optimization_log["strategy_performance"][strategy.value].append(score)

                # Track best result
                if score > best_score:
                    best_result = result
                    best_score = score
                    optimization_log["improvements"].append({
                        "iteration": iteration,
                        "strategy": strategy.value,
                        "score": score,
                        "improvement": score - best_score
                    })

                print(f"  {strategy.value}: {score:.3f}")

            # Early stopping if good enough
            if best_score > 0.9:
                break

        return best_result, optimization_log

    async def _analyze_query_characteristics(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze query characteristics to inform strategy selection."""

        analysis_prompt = f"""
        Analyze this query and provide characteristics:

        Query: {query}
        Context: {context or 'None'}

        Analyze the following aspects:
        1. Complexity level (simple/medium/complex)
        2. Domain (general/technical/scientific/creative)
        3. Query type (factual/analytical/creative/procedural)
        4. Specificity level (vague/specific/very_specific)
        5. Required reasoning type (direct/step-by-step/hypothesis-based)

        Provide analysis in this format:
        Complexity: [level]
        Domain: [domain]
        Type: [type]
        Specificity: [level]
        Reasoning: [type]
        """

        response = await self.llm_client.complete(analysis_prompt)

        # Parse response into structured analysis
        analysis = {}
        for line in response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                analysis[key.strip().lower()] = value.strip().lower()

        return analysis

    def _select_optimal_strategy(
        self,
        query_analysis: Dict[str, Any],
        performance_feedback: Optional[Dict[str, float]]
    ) -> RewriteStrategy:
        """Select the optimal strategy based on query analysis and feedback."""

        # Default strategy mapping based on query characteristics
        strategy_mapping = {
            "complex": RewriteStrategy.CHAIN_OF_THOUGHT,
            "step-by-step": RewriteStrategy.CHAIN_OF_THOUGHT,
            "technical": RewriteStrategy.CONTEXTUAL,
            "scientific": RewriteStrategy.CONTEXTUAL,
            "vague": RewriteStrategy.FEW_SHOT,
            "simple": RewriteStrategy.FEW_SHOT,
        }

        # Check query characteristics
        for characteristic, strategy in strategy_mapping.items():
            if any(characteristic in str(value) for value in query_analysis.values()):
                return strategy

        # Use performance feedback if available
        if performance_feedback:
            best_strategy = max(performance_feedback.keys(), key=lambda k: performance_feedback[k])
            try:
                return RewriteStrategy(best_strategy)
            except ValueError:
                pass

        # Default fallback
        return RewriteStrategy.CHAIN_OF_THOUGHT

    async def _get_user_feedback(self, result: RewriteResult) -> Dict[str, Any]:
        """Simulate user feedback (in real implementation, this would be interactive)."""

        # Simulate feedback based on result quality
        feedback = {
            "rating": min(result.confidence_score + random.uniform(-0.2, 0.2), 1.0),
            "clarity": random.uniform(0.6, 1.0),
            "specificity": random.uniform(0.5, 1.0),
            "usefulness": random.uniform(0.6, 1.0),
            "timestamp": datetime.now().isoformat()
        }

        return feedback

    async def _apply_learned_optimizations(
        self,
        result: RewriteResult,
        performance_feedback: Dict[str, float]
    ) -> RewriteResult:
        """Apply learned optimizations from previous sessions."""

        # Extract patterns from successful rewrites
        successful_patterns = self._extract_successful_patterns()

        if successful_patterns:
            optimization_prompt = f"""
            Improve this rewritten query using these successful patterns:

            Current query: {result.rewritten_query}
            Successful patterns: {successful_patterns}

            Enhanced query:
            """

            try:
                enhanced_query = await self.llm_client.complete(optimization_prompt)
                result.rewritten_query = enhanced_query.strip()
                result.confidence_score = min(result.confidence_score + 0.1, 1.0)
                result.metadata["applied_learning"] = True
            except Exception:
                pass  # Keep original if enhancement fails

        return result

    async def _combine_strategies(
        self,
        original_query: str,
        results: Dict[str, RewriteResult],
        context: Optional[Dict[str, Any]]
    ) -> RewriteResult:
        """Combine multiple strategy results into a collaborative rewrite."""

        rewritten_queries = [result.rewritten_query for result in results.values()]

        combination_prompt = f"""
        Combine the best elements of these rewritten queries:

        Original: {original_query}

        Rewrites:
        {chr(10).join([f"- {query}" for query in rewritten_queries])}

        Create an optimal combined version that incorporates the strengths of each:
        """

        combined_query = await self.llm_client.complete(combination_prompt)

        # Calculate average confidence
        avg_confidence = sum(result.confidence_score for result in results.values()) / len(results)

        return RewriteResult(
            original_query=original_query,
            rewritten_query=combined_query.strip(),
            strategy=RewriteStrategy.CONTEXTUAL,  # Use contextual as default for combinations
            confidence_score=min(avg_confidence + 0.1, 1.0),
            metadata={
                "method": "collaborative_combination",
                "strategies_combined": list(results.keys()),
                "source_results": len(results)
            },
            timestamp=datetime.now(),
            reasoning="Combined multiple strategies for optimal result"
        )

    def _extract_successful_patterns(self) -> List[str]:
        """Extract patterns from successful previous rewrites."""

        patterns = []
        for session in self.session_history[-5:]:  # Last 5 sessions
            if session.get("best_rewrite"):
                best_rewrite = session["best_rewrite"]
                if best_rewrite.confidence_score > 0.8:
                    patterns.append(best_rewrite.rewritten_query[:100])

        return patterns

    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics from all interactive sessions."""

        if not self.session_history:
            return {"total_sessions": 0}

        total_sessions = len(self.session_history)
        strategy_performance = {}
        avg_improvement = 0

        for session in self.session_history:
            for iteration in session["iterations"]:
                strategy = iteration["strategy"]
                rating = iteration.get("user_feedback", {}).get("rating", 0)

                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                strategy_performance[strategy].append(rating)

        # Calculate average performance per strategy
        for strategy, ratings in strategy_performance.items():
            strategy_performance[strategy] = sum(ratings) / len(ratings)

        return {
            "total_sessions": total_sessions,
            "strategy_performance": strategy_performance,
            "most_effective_strategy": max(strategy_performance.keys(), key=lambda k: strategy_performance[k]) if strategy_performance else None,
            "average_session_improvement": avg_improvement,
            "total_rewrites": sum(len(s["iterations"]) for s in self.session_history)
        }


# Utility for importing in real applications
import random