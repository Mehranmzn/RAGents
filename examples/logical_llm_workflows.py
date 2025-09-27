"""Advanced workflows demonstrating Logical LLM integration with different use cases."""

import asyncio
import os
from typing import Dict, List, Any

from ragents import (
    LLMClient,
    RAGEngine,
    RAGConfig,
    LogicalReasoner,
    QueryClarifier,
    ConstraintEngine,
)
from ragents.llm.types import ModelConfig, ModelProvider
from ragents.logical_llm.integration import LogicalLLMIntegration, LogicalAgent
from ragents.agents.base import AgentConfig


class FinancialQueryOptimizer:
    """Specialized workflow for optimizing financial queries."""

    def __init__(self, llm_client: LLMClient, rag_engine: RAGEngine):
        self.llm_client = llm_client
        self.rag_engine = rag_engine
        self.logical_integration = LogicalLLMIntegration(llm_client)

    async def process_financial_query(self, query: str) -> Dict[str, Any]:
        """Process a financial query with maximum optimization."""
        # Step 1: Logical analysis
        result = await self.logical_integration.process_query(query, interactive=True)

        # Step 2: Financial-specific optimizations
        if result.logical_query.domain == "financial":
            # Extract key financial parameters
            financial_params = self._extract_financial_parameters(result.logical_query.parameters)

            # Generate multiple focused search strategies
            search_strategies = await self._generate_search_strategies(financial_params)

            # Execute optimized searches
            search_results = await self._execute_optimized_searches(search_strategies)

            return {
                "original_query": query,
                "logical_analysis": result.logical_query.__dict__,
                "optimization_results": {
                    "token_reduction": result.estimated_token_reduction,
                    "confidence": result.processing_confidence,
                    "search_strategies": len(search_strategies),
                    "total_results": len(search_results)
                },
                "search_results": search_results,
                "clarifications_needed": [req.__dict__ for req in result.clarification_requests]
            }
        else:
            return await self._handle_non_financial_query(query, result)

    def _extract_financial_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate financial-specific parameters."""
        financial_params = {}

        # Core financial entities
        if "company" in parameters:
            financial_params["company"] = parameters["company"]

        # Time parameters
        for time_param in ["year", "quarter", "time_period"]:
            if time_param in parameters:
                financial_params["time"] = parameters[time_param]
                break

        # Metric parameters
        if "metric_type" in parameters:
            financial_params["metric"] = parameters["metric_type"]

        # Optional parameters
        for optional_param in ["currency", "segment", "division"]:
            if optional_param in parameters:
                financial_params[optional_param] = parameters[optional_param]

        return financial_params

    async def _generate_search_strategies(self, financial_params: Dict[str, Any]) -> List[str]:
        """Generate multiple search strategies for comprehensive results."""
        strategies = []

        # Strategy 1: Exact match strategy
        if all(key in financial_params for key in ["company", "time", "metric"]):
            exact_query = f"{financial_params['company']} {financial_params['metric']} {financial_params['time']}"
            strategies.append(exact_query)

        # Strategy 2: Company-focused strategy
        if "company" in financial_params:
            company_query = f"{financial_params['company']} financial data"
            if "time" in financial_params:
                company_query += f" {financial_params['time']}"
            strategies.append(company_query)

        # Strategy 3: Metric-focused strategy
        if "metric" in financial_params and "company" in financial_params:
            metric_query = f"{financial_params['metric']} {financial_params['company']}"
            strategies.append(metric_query)

        # Strategy 4: Temporal strategy
        if "time" in financial_params and "company" in financial_params:
            temporal_query = f"{financial_params['company']} earnings {financial_params['time']}"
            strategies.append(temporal_query)

        return strategies

    async def _execute_optimized_searches(self, strategies: List[str]) -> List[Dict[str, Any]]:
        """Execute multiple search strategies and aggregate results."""
        results = []

        for strategy in strategies:
            try:
                # Execute RAG search with strategy
                rag_response = await self.rag_engine.query(strategy)

                result = {
                    "strategy": strategy,
                    "answer": rag_response.answer,
                    "confidence": rag_response.confidence,
                    "sources": len(rag_response.sources),
                    "processing_time": rag_response.processing_time
                }
                results.append(result)

            except Exception as e:
                results.append({
                    "strategy": strategy,
                    "error": str(e),
                    "success": False
                })

        return results

    async def _handle_non_financial_query(self, query: str, result) -> Dict[str, Any]:
        """Handle non-financial queries with general optimization."""
        return {
            "original_query": query,
            "domain": result.logical_query.domain,
            "optimization_applied": False,
            "reason": "Query not in financial domain",
            "logical_analysis": result.logical_query.__dict__
        }


class InteractiveClarificationWorkflow:
    """Workflow for handling complex queries requiring multiple clarifications."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logical_integration = LogicalLLMIntegration(llm_client)
        self.conversation_history = []

    async def handle_complex_query(self, query: str) -> Dict[str, Any]:
        """Handle a complex query with interactive clarification."""
        workflow_result = {
            "original_query": query,
            "clarification_rounds": [],
            "final_result": None,
            "total_token_reduction": 0.0
        }

        current_query = query
        round_number = 1
        max_rounds = 3

        while round_number <= max_rounds:
            # Process current query state
            result = await self.logical_integration.process_query(current_query, interactive=True)

            round_data = {
                "round": round_number,
                "query": current_query,
                "confidence": result.processing_confidence,
                "should_proceed": result.should_proceed,
                "clarifications": [req.__dict__ for req in result.clarification_requests],
                "token_reduction": result.estimated_token_reduction
            }

            workflow_result["clarification_rounds"].append(round_data)
            workflow_result["total_token_reduction"] += result.estimated_token_reduction

            if result.should_proceed:
                # Query is ready for processing
                workflow_result["final_result"] = result.__dict__
                break

            elif result.clarification_requests:
                # Simulate user providing clarifications
                clarifications = await self._simulate_user_clarifications(result.clarification_requests)

                if clarifications:
                    # Process clarifications
                    refined_result = await self.logical_integration.handle_clarification_response(
                        result, clarifications
                    )

                    # Update current query with refined information
                    current_query = self._build_refined_query(query, refined_result.logical_query.parameters)
                    round_data["user_clarifications"] = clarifications
                    round_data["refined_query"] = current_query
                else:
                    # No more clarifications available
                    break

            round_number += 1

        return workflow_result

    async def _simulate_user_clarifications(self, requests) -> Dict[str, str]:
        """Simulate user providing clarifications (for demo purposes)."""
        clarifications = {}

        for request in requests[:2]:  # Handle up to 2 clarifications per round
            field_name = request.field_name

            # Simulate responses based on field type
            if "company" in field_name:
                clarifications[field_name] = "Apple Inc."
            elif "time" in field_name or "year" in field_name:
                clarifications[field_name] = "2023"
            elif "quarter" in field_name:
                clarifications[field_name] = "Q3"
            elif "metric" in field_name:
                clarifications[field_name] = "revenue"
            elif "currency" in field_name:
                clarifications[field_name] = "USD"
            else:
                clarifications[field_name] = f"example_{field_name}"

        return clarifications

    def _build_refined_query(self, original_query: str, parameters: Dict[str, Any]) -> str:
        """Build a refined query incorporating clarified parameters."""
        # Extract key parameters
        company = parameters.get("company", "")
        time_info = parameters.get("time_period") or parameters.get("year", "")
        metric = parameters.get("metric_type", "")

        # Build refined query
        parts = [part for part in [company, metric, time_info] if part]
        if parts:
            return " ".join(parts)
        else:
            return original_query


class TokenOptimizationBenchmark:
    """Benchmark workflow for measuring token optimization effectiveness."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logical_integration = LogicalLLMIntegration(llm_client)

    async def run_optimization_benchmark(self, test_queries: List[str]) -> Dict[str, Any]:
        """Run optimization benchmark on a set of test queries."""
        results = []
        total_original_tokens = 0
        total_optimized_tokens = 0

        for query in test_queries:
            result = await self._benchmark_single_query(query)
            results.append(result)

            total_original_tokens += result["original_tokens"]
            total_optimized_tokens += result["optimized_tokens"]

        overall_reduction = total_original_tokens - total_optimized_tokens
        overall_percentage = (overall_reduction / total_original_tokens * 100) if total_original_tokens > 0 else 0

        return {
            "total_queries": len(test_queries),
            "individual_results": results,
            "overall_metrics": {
                "total_original_tokens": total_original_tokens,
                "total_optimized_tokens": total_optimized_tokens,
                "total_reduction": overall_reduction,
                "reduction_percentage": overall_percentage,
                "average_confidence": sum(r["confidence"] for r in results) / len(results) if results else 0
            }
        }

    async def _benchmark_single_query(self, query: str) -> Dict[str, Any]:
        """Benchmark optimization for a single query."""
        # Estimate original tokens (rough approximation)
        original_tokens = len(query.split()) * 1.3

        try:
            # Process through logical LLM
            result = await self.logical_integration.process_query(query, interactive=False)

            # Estimate optimized tokens
            optimized_tokens = len(result.optimized_search_query.split()) * 1.3

            return {
                "query": query,
                "original_tokens": int(original_tokens),
                "optimized_tokens": int(optimized_tokens),
                "token_reduction": int(original_tokens - optimized_tokens),
                "reduction_percentage": ((original_tokens - optimized_tokens) / original_tokens * 100) if original_tokens > 0 else 0,
                "confidence": result.processing_confidence,
                "domain": result.logical_query.domain,
                "optimization_applied": result.should_proceed,
                "optimized_query": result.optimized_search_query
            }

        except Exception as e:
            return {
                "query": query,
                "original_tokens": int(original_tokens),
                "optimized_tokens": int(original_tokens),
                "token_reduction": 0,
                "reduction_percentage": 0,
                "confidence": 0.0,
                "error": str(e),
                "optimization_applied": False
            }


async def demo_financial_query_optimizer():
    """Demonstrate financial query optimization workflow."""
    print("=== Financial Query Optimizer Demo ===")

    # Setup
    llm_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY", "demo-key")
    )
    llm_client = LLMClient(llm_config)

    rag_config = RAGConfig(vector_store_type="memory", collection_name="financial_demo")
    rag_engine = RAGEngine(rag_config, llm_client)

    optimizer = FinancialQueryOptimizer(llm_client, rag_engine)

    # Test financial queries
    financial_queries = [
        "What was Apple's revenue in Q3 2023?",
        "Show me Microsoft's profit margins for fiscal year 2023",
        "Compare Tesla and Ford's quarterly earnings for 2023",
    ]

    for query in financial_queries:
        print(f"\nProcessing: {query}")

        try:
            result = await optimizer.process_financial_query(query)

            print(f"Domain: {result['logical_analysis']['domain']}")
            print(f"Token Reduction: {result['optimization_results']['token_reduction']:.1%}")
            print(f"Search Strategies: {result['optimization_results']['search_strategies']}")

            if result['clarifications_needed']:
                print(f"Clarifications: {len(result['clarifications_needed'])} needed")

        except Exception as e:
            print(f"Demo error (expected with mock setup): {e}")

        print("-" * 50)


async def demo_interactive_clarification():
    """Demonstrate interactive clarification workflow."""
    print("=== Interactive Clarification Demo ===")

    llm_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY", "demo-key")
    )
    llm_client = LLMClient(llm_config)

    workflow = InteractiveClarificationWorkflow(llm_client)

    # Test complex/ambiguous queries
    complex_queries = [
        "Show me the numbers",  # Very vague
        "How did the tech companies perform last quarter?",  # Multiple ambiguities
        "Compare the financial performance",  # Missing everything
    ]

    for query in complex_queries:
        print(f"\nProcessing complex query: {query}")

        try:
            result = await workflow.handle_complex_query(query)

            print(f"Clarification rounds: {len(result['clarification_rounds'])}")
            print(f"Total token reduction: {result['total_token_reduction']:.1%}")

            for round_data in result['clarification_rounds']:
                print(f"  Round {round_data['round']}: Confidence {round_data['confidence']:.2f}")
                if round_data.get('user_clarifications'):
                    print(f"    Clarifications: {round_data['user_clarifications']}")
                if round_data.get('refined_query'):
                    print(f"    Refined: {round_data['refined_query']}")

            if result['final_result']:
                print("✅ Query successfully refined and ready for processing")
            else:
                print("❌ Query could not be sufficiently clarified")

        except Exception as e:
            print(f"Demo error (expected with mock setup): {e}")

        print("-" * 50)


async def demo_token_optimization_benchmark():
    """Demonstrate token optimization benchmarking."""
    print("=== Token Optimization Benchmark Demo ===")

    llm_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY", "demo-key")
    )
    llm_client = LLMClient(llm_config)

    benchmark = TokenOptimizationBenchmark(llm_client)

    # Diverse test queries
    test_queries = [
        "What was Apple's revenue in Q3 2023?",  # Well-formed
        "I need detailed information about Apple Inc.'s quarterly revenue performance during the third quarter of fiscal year 2023",  # Verbose
        "Show me tech company data",  # Vague
        "Compare Apple vs Microsoft financial metrics for the last quarter of 2023",  # Complex
        "Apple earnings Q3",  # Abbreviated
        "How much money did Apple make last quarter compared to the same period last year?",  # Conversational
    ]

    print(f"Benchmarking {len(test_queries)} queries...")

    try:
        results = await benchmark.run_optimization_benchmark(test_queries)

        print(f"\nOverall Results:")
        print(f"Total Token Reduction: {results['overall_metrics']['total_reduction']:.0f} tokens")
        print(f"Reduction Percentage: {results['overall_metrics']['reduction_percentage']:.1f}%")
        print(f"Average Confidence: {results['overall_metrics']['average_confidence']:.2f}")

        print(f"\nTop Optimizations:")
        sorted_results = sorted(results['individual_results'],
                              key=lambda x: x.get('reduction_percentage', 0),
                              reverse=True)

        for result in sorted_results[:3]:
            if result.get('reduction_percentage', 0) > 0:
                print(f"  {result['reduction_percentage']:.1f}% reduction: {result['query'][:50]}...")
                print(f"    → {result['optimized_query']}")

    except Exception as e:
        print(f"Demo error (expected with mock setup): {e}")

    print("-" * 50)


async def main():
    """Run all logical LLM workflow demonstrations."""
    print("🧠 RAGents Logical LLM Advanced Workflows")
    print("=" * 60)
    print("Advanced workflows demonstrating production-ready logical LLM integration")
    print()

    workflows = [
        demo_financial_query_optimizer,
        demo_interactive_clarification,
        demo_token_optimization_benchmark,
    ]

    for workflow in workflows:
        try:
            await workflow()
        except Exception as e:
            print(f"Workflow {workflow.__name__} failed: {e}")

        print("\n" + "=" * 60 + "\n")

    print("✅ All logical LLM workflow demos completed!")
    print("\nProduction Benefits:")
    print("• Significant token reduction in LLM operations")
    print("• Intelligent query clarification reduces back-and-forth")
    print("• Domain-specific optimization for better results")
    print("• Automated constraint validation and reasoning")
    print("• Measurable performance improvements")


if __name__ == "__main__":
    asyncio.run(main())