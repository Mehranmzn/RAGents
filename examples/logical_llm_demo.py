"""Demonstration of Logical LLM module for token reduction and intelligent query processing."""

import asyncio
import os
from pathlib import Path

from ragents import (
    LLMClient,
    RAGEngine,
    RAGConfig,
    LogicalReasoner,
    QueryClarifier,
    ConstraintEngine,
    LogicPattern,
    PatternMatcher,
    BuiltinPatterns,
)
from ragents.llm.types import ModelConfig, ModelProvider
from ragents.logical_llm.integration import (
    LogicalLLMIntegration,
    LogicalAgent,
    analyze_token_reduction_potential,
)
from ragents.agents.base import AgentConfig


async def setup_demo_environment():
    """Set up demo environment with mock LLM and RAG."""
    # Create LLM client (would use real API key in production)
    llm_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY", "demo-key"),
        temperature=0.7
    )
    llm_client = LLMClient(llm_config)

    # Create RAG engine
    rag_config = RAGConfig(
        vector_store_type="memory",
        collection_name="logical_demo",
        chunk_size=300,
        top_k=3
    )
    rag_engine = RAGEngine(rag_config, llm_client)

    return llm_client, rag_engine


async def demo_logical_reasoner():
    """Demonstrate logical reasoning capabilities."""
    print("=== Logical Reasoner Demo ===")

    llm_client, _ = await setup_demo_environment()
    reasoner = LogicalReasoner(llm_client)

    # Test queries that would benefit from logical processing
    test_queries = [
        "What was Apple's revenue in Q3 2023?",
        "Show me Microsoft's financial numbers for 2023",
        "Compare Tesla and Ford profits last year",
        "When did Amazon report their earnings?",
        "How much did Google spend on R&D?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            logical_query = await reasoner.analyze_query(query)

            print(f"Domain: {logical_query.domain}")
            print(f"Intent: {logical_query.intent}")
            print(f"Parameters: {logical_query.parameters}")
            print(f"Missing: {logical_query.missing_parameters}")
            print(f"Confidence: {logical_query.confidence_score:.2f}")

            if logical_query.refinement_suggestions:
                print("Suggestions:")
                for suggestion in logical_query.refinement_suggestions[:2]:
                    print(f"  - {suggestion}")

            # Generate focused search query
            focused_query = reasoner.generate_focused_search_query(logical_query)
            print(f"Focused Query: '{focused_query}'")

        except Exception as e:
            print(f"Demo error (expected with mock setup): {e}")

        print("-" * 50)


async def demo_query_clarifier():
    """Demonstrate query clarification capabilities."""
    print("=== Query Clarifier Demo ===")

    llm_client, _ = await setup_demo_environment()
    reasoner = LogicalReasoner(llm_client)
    clarifier = QueryClarifier(llm_client, reasoner)

    # Test ambiguous queries
    ambiguous_queries = [
        "Show me Apple's numbers",  # Missing time period and metric
        "What about Tesla last quarter?",  # Missing metric type
        "Compare the two companies",  # Missing companies and metric
        "How did Meta perform?",  # Missing time and metric
    ]

    for query in ambiguous_queries:
        print(f"\nQuery: {query}")
        try:
            logical_query, clarifications = await clarifier.analyze_and_clarify(query)

            print(f"Completeness: {'✓ Complete' if logical_query.is_complete() else '✗ Incomplete'}")
            print(f"Confidence: {logical_query.confidence_score:.2f}")

            if clarifications:
                print(f"Clarifications needed ({len(clarifications)}):")
                for i, clarification in enumerate(clarifications[:3], 1):
                    print(f"  {i}. {clarification.question}")
                    if clarification.options:
                        print(f"     Options: {', '.join(clarification.options[:3])}")

                # Simulate user response to first clarification
                if clarifications:
                    first_clarification = clarifications[0]
                    print(f"\nSimulated user response to: {first_clarification.question}")

                    # Mock user response
                    if "company" in first_clarification.field_name:
                        mock_response = "Apple Inc."
                    elif "time" in first_clarification.field_name:
                        mock_response = "Q3 2023"
                    elif "metric" in first_clarification.field_name:
                        mock_response = "revenue"
                    else:
                        mock_response = "example response"

                    print(f"User: {mock_response}")

                    # Process the response
                    clarification_response = await clarifier.process_clarification_response(
                        first_clarification, mock_response
                    )
                    print(f"Extracted: {clarification_response.extracted_value}")
                    print(f"Confidence: {clarification_response.confidence:.2f}")

        except Exception as e:
            print(f"Demo error (expected with mock setup): {e}")

        print("-" * 50)


async def demo_pattern_matching():
    """Demonstrate logic pattern matching."""
    print("=== Pattern Matching Demo ===")

    # Create pattern matcher with built-in patterns
    matcher = BuiltinPatterns.create_pattern_matcher()

    # Test queries for pattern matching
    test_queries = [
        "What was Apple's revenue in Q3 2023?",  # Financial pattern
        "Compare Tesla vs Ford profit margins",  # Comparative pattern
        "Show me the trend in Netflix subscriber growth over the last 3 years",  # Trend pattern
        "Why did Amazon's stock price drop?",  # Causal pattern
        "How many users does Instagram have?",  # Quantitative pattern
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")

        # Find best matching pattern
        best_pattern, confidence = matcher.match_query(query)

        if best_pattern:
            print(f"Best Pattern: {best_pattern.name}")
            print(f"Type: {best_pattern.pattern_type.value}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Token Reduction Potential: {best_pattern.token_reduction_potential:.1%}")

            # Show required entities
            if best_pattern.required_entities:
                print(f"Required: {', '.join(best_pattern.required_entities)}")

            # Show refinement questions
            if best_pattern.refinement_questions:
                print("Potential clarifications:")
                for question in best_pattern.refinement_questions[:2]:
                    print(f"  - {question}")
        else:
            print("No pattern matched")

        # Get all matches above threshold
        all_matches = matcher.get_all_matches(query, min_confidence=0.3)
        if len(all_matches) > 1:
            print(f"Other matches: {', '.join([p.name for p, _ in all_matches[1:3]])}")

        print("-" * 50)


async def demo_token_reduction():
    """Demonstrate token reduction analysis."""
    print("=== Token Reduction Analysis Demo ===")

    llm_client, _ = await setup_demo_environment()

    # Test queries with different complexity levels
    test_cases = [
        {
            "query": "What was Apple's revenue in Q3 2023?",
            "description": "Specific, well-formed query"
        },
        {
            "query": "I'm looking for information about the financial performance and earnings reports of Apple Inc. during the third quarter of the fiscal year 2023, specifically interested in their revenue numbers and how they compare to previous quarters and analyst expectations",
            "description": "Verbose, redundant query"
        },
        {
            "query": "Show me some financial data about tech companies",
            "description": "Vague, under-specified query"
        },
        {
            "query": "Compare Apple, Microsoft, and Google revenue for 2023 Q1, Q2, Q3, and Q4, showing year-over-year growth rates and market share implications",
            "description": "Complex, multi-entity query"
        }
    ]

    for case in test_cases:
        query = case["query"]
        description = case["description"]

        print(f"\nCase: {description}")
        print(f"Original Query ({len(query)} chars): {query}")

        try:
            analysis = await analyze_token_reduction_potential(query, llm_client)

            print(f"Optimized Query: {analysis['optimized_query']}")
            print(f"Token Reduction: {analysis['token_reduction']:.1f} tokens ({analysis['token_reduction_percentage']:.1f}%)")
            print(f"Logical Domain: {analysis['logical_domain']}")
            print(f"Confidence: {analysis['confidence_score']:.2f}")

            if analysis['requires_clarification']:
                print("⚠️  Requires clarification before processing")
            else:
                print("✅ Ready for optimized processing")

        except Exception as e:
            print(f"Demo error (expected with mock setup): {e}")

        print("-" * 60)


async def demo_logical_agent():
    """Demonstrate LogicalAgent with LLM integration."""
    print("=== Logical Agent Demo ===")

    llm_client, rag_engine = await setup_demo_environment()

    # Create logical agent
    agent_config = AgentConfig(
        name="LogicalFinancialAgent",
        description="Financial agent with logical reasoning capabilities",
        enable_rag=True,
        enable_reasoning=True
    )

    from ragents.logical_llm.integration import LogicalAgent
    agent = LogicalAgent(agent_config, llm_client, rag_engine)

    # Test queries that benefit from logical processing
    test_conversations = [
        "What was Apple's revenue in Q3 2023?",  # Complete query
        "Show me Apple's numbers",  # Incomplete query requiring clarification
        "Compare tech companies performance",  # Very vague query
        "Apple revenue Q3 2023",  # Abbreviated but clear query
    ]

    for query in test_conversations:
        print(f"\nUser: {query}")

        try:
            response = await agent.process_message(query)
            print(f"Agent: {response}")

            # Show if logical processing was used
            if hasattr(agent, 'logical_integration'):
                print("✓ Enhanced with logical LLM processing")
            else:
                print("Standard processing")

        except Exception as e:
            print(f"Agent response: I need more specific information to help you effectively.")
            print(f"(Demo error: {e})")

        print("-" * 50)


async def demo_integration_workflow():
    """Demonstrate complete integration workflow."""
    print("=== Integration Workflow Demo ===")

    llm_client, rag_engine = await setup_demo_environment()

    # Create logical LLM integration
    integration = LogicalLLMIntegration(llm_client)

    # Complex workflow example
    original_query = "I need to understand how Apple's financial performance compared to their competitors in the technology sector during 2023, particularly focusing on revenue growth and profitability metrics"

    print(f"Original Query: {original_query}")
    print(f"Length: {len(original_query)} characters")

    try:
        # Step 1: Process through logical LLM
        result = await integration.process_query(original_query, interactive=True)

        print(f"\nLogical Analysis:")
        print(f"Domain: {result.logical_query.domain}")
        print(f"Confidence: {result.processing_confidence:.2f}")
        print(f"Should Proceed: {result.should_proceed}")

        # Step 2: Show optimization results
        print(f"\nOptimization Results:")
        print(f"Optimized Query: {result.optimized_search_query}")
        print(f"Estimated Token Reduction: {result.estimated_token_reduction:.1%}")

        # Step 3: Handle clarifications if needed
        if result.clarification_requests:
            print(f"\nClarifications Needed ({len(result.clarification_requests)}):")
            for req in result.clarification_requests[:2]:
                print(f"- {req.question}")

            # Simulate providing clarifications
            print("\nSimulating clarification responses...")
            clarification_responses = {
                "company": "Apple Inc.",
                "time_period": "2023",
                "metric_type": "revenue"
            }

            # Process clarifications
            refined_result = await integration.handle_clarification_response(
                result, clarification_responses
            )

            print(f"After clarification:")
            print(f"Refined Query: {refined_result.optimized_search_query}")
            print(f"New Confidence: {refined_result.processing_confidence:.2f}")

    except Exception as e:
        print(f"Demo error (expected with mock setup): {e}")

    print("-" * 60)


async def main():
    """Run all logical LLM demonstrations."""
    print("🧠 RAGents Logical LLM Module Demonstration")
    print("=" * 60)
    print("This module reduces token consumption through intelligent query refinement")
    print("and logical constraint solving, inspired by Logic-LLM research.")
    print()

    demos = [
        demo_logical_reasoner,
        demo_query_clarifier,
        demo_pattern_matching,
        demo_token_reduction,
        demo_logical_agent,
        demo_integration_workflow,
    ]

    for demo in demos:
        try:
            await demo()
        except Exception as e:
            print(f"Demo {demo.__name__} failed: {e}")

        print("\n" + "=" * 60 + "\n")

    print("✅ All Logical LLM demos completed!")
    print("\nKey Benefits Demonstrated:")
    print("• Intelligent query analysis and domain detection")
    print("• Automatic clarification for ambiguous queries")
    print("• Token reduction through focused search queries")
    print("• Logical constraint solving and validation")
    print("• Pattern-based query optimization")
    print("• Seamless integration with existing RAGents agents")
    print("\nInspired by Logic-LLM research for faithful logical reasoning")
    print("and optimized token consumption in LLM interactions.")


if __name__ == "__main__":
    asyncio.run(main())