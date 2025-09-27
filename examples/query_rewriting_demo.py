"""Demonstration of RAGents query rewriting capabilities."""

import asyncio
import os
from typing import Dict, Any

from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env
from ragents.query_rewriting import (
    CoTRewriter,
    FewShotRewriter,
    ContextualRewriter,
    InteractiveRewriter,
    PromptOptimizer,
    RewriteEvaluator,
    OptimizationConfig,
    OptimizationObjective,
)


async def demo_basic_rewriting():
    """Demonstrate basic query rewriting strategies."""
    print("🔧 RAGents Query Rewriting Demo")
    print("=" * 60)

    # Initialize LLM client
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Test queries
    test_queries = [
        "What is machine learning?",
        "How do I optimize my database?",
        "Explain climate change effects",
        "Best practices for web security",
        "Python vs JavaScript performance"
    ]

    # Initialize rewriters
    rewriters = {
        "Chain-of-Thought": CoTRewriter(llm_client),
        "Few-Shot": FewShotRewriter(llm_client),
        "Contextual": ContextualRewriter(llm_client),
    }

    for query in test_queries:
        print(f"\n📝 Original Query: {query}")
        print("-" * 40)

        for strategy_name, rewriter in rewriters.items():
            try:
                result = await rewriter.rewrite(query)
                print(f"\n🔧 {strategy_name} Rewrite:")
                print(f"   Query: {result.rewritten_query}")
                print(f"   Confidence: {result.confidence_score:.2f}")
                print(f"   Reasoning: {result.reasoning}")

            except Exception as e:
                print(f"❌ Error with {strategy_name}: {e}")

        print("\n" + "=" * 60)


async def demo_contextual_rewriting():
    """Demonstrate contextual query rewriting."""
    print("\n🎯 Contextual Rewriting Demo")
    print("-" * 40)

    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)
    contextual_rewriter = ContextualRewriter(llm_client)

    # Test with different contexts
    base_query = "How do I implement authentication?"

    contexts = [
        {
            "domain": "web_development",
            "situation": "building_api",
            "user_context": "beginner_developer"
        },
        {
            "domain": "mobile_development",
            "situation": "security_audit",
            "user_context": "security_expert"
        },
        {
            "domain": "enterprise_software",
            "situation": "compliance_requirement",
            "user_context": "enterprise_architect"
        }
    ]

    print(f"📝 Base Query: {base_query}")

    for i, context in enumerate(contexts, 1):
        print(f"\n🌐 Context {i}: {context}")
        try:
            result = await contextual_rewriter.rewrite(base_query, context)
            print(f"🔧 Rewritten: {result.rewritten_query}")
            print(f"📊 Confidence: {result.confidence_score:.2f}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_interactive_rewriting():
    """Demonstrate interactive rewriting session."""
    print("\n🎮 Interactive Rewriting Demo")
    print("-" * 40)

    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)
    interactive_rewriter = InteractiveRewriter(llm_client)

    test_query = "Explain how neural networks work"
    context = {
        "domain": "machine_learning",
        "situation": "educational",
        "user_context": "computer_science_student"
    }

    print(f"📝 Query: {test_query}")
    print(f"🌐 Context: {context}")

    try:
        session = await interactive_rewriter.interactive_rewrite_session(test_query, context)

        print(f"\n🏆 Session Results:")
        print(f"   Session ID: {session['session_id']}")
        print(f"   Strategies Tested: {', '.join(session['strategies_tested'])}")
        print(f"   Best Rewrite: {session['best_rewrite'].rewritten_query}")
        print(f"   Best Strategy: {session['best_rewrite'].strategy.value}")

    except Exception as e:
        print(f"❌ Interactive session error: {e}")


async def demo_prompt_optimization():
    """Demonstrate automatic prompt optimization."""
    print("\n🚀 Prompt Optimization Demo")
    print("-" * 40)

    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Create a simple evaluation function
    async def simple_evaluation(query: str, response: str) -> float:
        # Simple scoring based on response length and keywords
        score = min(len(response) / 200, 1.0)  # Prefer longer responses
        if "step" in response.lower() or "because" in response.lower():
            score += 0.2
        return min(score, 1.0)

    optimizer = PromptOptimizer(llm_client, simple_evaluation)
    base_rewriter = CoTRewriter(llm_client)

    test_queries = [
        "Explain photosynthesis",
        "How do computers work?",
        "What causes earthquakes?"
    ]

    objective = OptimizationObjective(
        primary_metric="response_quality",
        target_performance=0.8
    )

    config = OptimizationConfig(
        max_iterations=3,
        population_size=3,
        use_genetic_algorithm=False  # Use simpler gradient-free optimization
    )

    print(f"📝 Test Queries: {test_queries}")
    print(f"🎯 Optimization Target: {objective.target_performance}")

    try:
        optimized_template, log = await optimizer.optimize_prompt(
            base_rewriter, test_queries, objective, config
        )

        print(f"\n✅ Optimization Complete!")
        print(f"🔧 Original Template:")
        print(f"   {base_rewriter.get_prompt_template()[:100]}...")
        print(f"\n🎯 Optimized Template:")
        print(f"   {optimized_template[:100]}...")
        print(f"\n📊 Optimization Method: {log['method']}")
        print(f"📈 Final Score: {log.get('final_score', 'N/A')}")

    except Exception as e:
        print(f"❌ Optimization error: {e}")


async def demo_evaluation_metrics():
    """Demonstrate rewrite evaluation metrics."""
    print("\n📊 Evaluation Metrics Demo")
    print("-" * 40)

    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)
    evaluator = RewriteEvaluator(llm_client)

    # Test different rewriters
    rewriters = {
        "CoT": CoTRewriter(llm_client),
        "FewShot": FewShotRewriter(llm_client),
        "Contextual": ContextualRewriter(llm_client),
    }

    test_query = "What are the environmental impacts of renewable energy?"

    print(f"📝 Evaluating Query: {test_query}")
    print("-" * 40)

    results = {}
    for name, rewriter in rewriters.items():
        try:
            rewrite_result = await rewriter.rewrite(test_query)
            metrics = await evaluator.evaluate_rewrite(rewrite_result)

            results[name] = {
                "rewrite": rewrite_result,
                "metrics": metrics
            }

            print(f"\n🔧 {name} Strategy:")
            print(f"   Rewritten: {rewrite_result.rewritten_query[:80]}...")
            print(f"   📊 Metrics:")
            print(f"      Clarity: {metrics.clarity_score:.2f}")
            print(f"      Specificity: {metrics.specificity_score:.2f}")
            print(f"      Completeness: {metrics.completeness_score:.2f}")
            print(f"      Intent Preservation: {metrics.intent_preservation_score:.2f}")
            print(f"      RAG Optimization: {metrics.rag_optimization_score:.2f}")
            print(f"      Overall Quality: {metrics.overall_quality_score:.2f}")

        except Exception as e:
            print(f"❌ Error evaluating {name}: {e}")

    # Find best strategy
    if results:
        best_strategy = max(results.keys(),
                          key=lambda k: results[k]["metrics"].overall_quality_score)
        best_score = results[best_strategy]["metrics"].overall_quality_score

        print(f"\n🏆 Best Strategy: {best_strategy}")
        print(f"📈 Best Score: {best_score:.2f}")


async def main():
    """Run the complete query rewriting demonstration."""
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    print("🌟 Welcome to RAGents Query Rewriting Demo!")
    print("This demo showcases automatic query optimization and rewriting capabilities.")

    try:
        await demo_basic_rewriting()
        await demo_contextual_rewriting()
        await demo_interactive_rewriting()
        await demo_prompt_optimization()
        await demo_evaluation_metrics()

        print("\n🎉 Demo completed successfully!")
        print("💡 Query rewriting can significantly improve RAG system performance")
        print("   by making queries more specific and effective for information retrieval.")

    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user.")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())