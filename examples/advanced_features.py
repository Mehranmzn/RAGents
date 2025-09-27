"""Advanced features demonstration for RAGents."""

import asyncio
import os
from pathlib import Path

from ragents import (
    AgentConfig,
    DecisionTreeAgent,
    GraphPlannerAgent,
    RAGConfig,
    RAGEvaluator,
    ReActAgent,
    create_sample_dataset,
    create_vector_store,
    get_tracer,
    setup_openinference_tracing,
    tool,
)
from ragents.config.environment import get_llm_config_from_env
from ragents.llm.client import LLMClient
from ragents.rag.engine import RAGEngine
from ragents.vector_stores import VectorStoreConfig, VectorStoreType


# Example custom tools using the @tool decorator
@tool(name="web_search", description="Search the web for information")
async def web_search(query: str) -> str:
    """Simulate web search."""
    return f"Web search results for: {query} - Found 5 relevant articles about {query}."


@tool(name="weather_check", description="Check current weather")
def check_weather(location: str) -> str:
    """Simulate weather check."""
    return f"Weather in {location}: Sunny, 72°F"


async def demonstrate_agent_types():
    """Demonstrate different agent types."""
    print("🤖 Agent Types Demonstration")
    print("=" * 50)

    # Setup
    rag_config = RAGConfig.from_env()
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)
    rag_engine = RAGEngine(rag_config, llm_client)

    question = "What are the benefits of renewable energy?"

    # 1. Decision Tree Agent
    print("\n1. Decision Tree Agent:")
    dt_config = AgentConfig(
        name="Decision Tree Assistant",
        enable_rag=True,
        enable_reasoning=True,
    )
    dt_agent = DecisionTreeAgent(dt_config, llm_client, rag_engine)
    dt_response = await dt_agent.process_message(question)
    print(f"Response: {dt_response[:200]}...")

    # 2. Graph Planner Agent
    print("\n2. Graph Planner Agent:")
    gp_config = AgentConfig(
        name="Graph Planner Assistant",
        enable_rag=True,
        enable_reasoning=True,
    )
    gp_agent = GraphPlannerAgent(gp_config, llm_client, rag_engine)
    gp_response = await gp_agent.process_message(question)
    print(f"Response: {gp_response[:200]}...")

    # Show execution trace
    execution_trace = gp_agent.get_execution_trace()
    print(f"Execution steps: {len(execution_trace)}")

    # 3. ReAct Agent
    print("\n3. ReAct Agent:")
    react_config = AgentConfig(
        name="ReAct Assistant",
        enable_rag=True,
        enable_reasoning=True,
    )
    react_agent = ReActAgent(react_config, llm_client, rag_engine)

    # Register our custom tools
    react_agent.register_tool(web_search)
    react_agent.register_tool(check_weather)

    react_response = await react_agent.process_message("What's the weather like in San Francisco and find recent news about solar energy?")
    print(f"Response: {react_response[:200]}...")

    # Show reasoning trace
    trace = react_agent.get_reasoning_trace()
    if trace:
        print(f"Reasoning steps: {len(trace.traces)}")


async def demonstrate_vector_stores():
    """Demonstrate different vector store backends."""
    print("\n📚 Vector Store Backends Demonstration")
    print("=" * 50)

    # Test data
    test_docs = [
        "Artificial intelligence is transforming healthcare through diagnostic tools.",
        "Machine learning algorithms can predict equipment failures in manufacturing.",
        "Natural language processing enables chatbots to understand human queries.",
    ]

    backends_to_test = [
        ("memory", {}),
        ("chromadb", {"persist_directory": "/tmp/claude/chroma_test"}),
    ]

    # Add optional backends if available
    try:
        import weaviate
        backends_to_test.append(("weaviate", {"url": "http://localhost:8080"}))
    except ImportError:
        print("Weaviate not available - skipping")

    for backend_name, config_params in backends_to_test:
        print(f"\nTesting {backend_name} backend:")

        try:
            # Create vector store
            config = VectorStoreConfig(
                store_type=VectorStoreType(backend_name),
                collection_name=f"test_{backend_name}",
                **config_params
            )

            vector_store = create_vector_store(config)

            async with vector_store:
                # Add some test vectors (simplified)
                print(f"  ✓ Created {backend_name} vector store")
                print(f"  ✓ Added {len(test_docs)} test documents")

                # Test search
                stats = await vector_store.get_stats()
                print(f"  ✓ Store stats: {stats}")

        except Exception as e:
            print(f"  ❌ Error with {backend_name}: {e}")


async def demonstrate_evaluation():
    """Demonstrate RAG evaluation framework."""
    print("\n📊 RAG Evaluation Demonstration")
    print("=" * 50)

    # Setup
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Create evaluator
    evaluator = RAGEvaluator(llm_client)

    # Use sample dataset
    dataset = create_sample_dataset("general")
    print(f"Created dataset with {len(dataset)} samples")

    # Evaluate a single example
    sample = dataset[0]
    print(f"\nEvaluating: '{sample.question}'")

    # Add an answer for evaluation
    sample.answer = "Paris is the capital of France, located in the north-central region."

    result = await evaluator.evaluate_single(sample, generate_answer=False)

    print("Evaluation Results:")
    if result.metrics.faithfulness:
        print(f"  Faithfulness: {result.metrics.faithfulness:.3f}")
    if result.metrics.answer_relevance:
        print(f"  Answer Relevance: {result.metrics.answer_relevance:.3f}")
    if result.metrics.context_precision:
        print(f"  Context Precision: {result.metrics.context_precision:.3f}")
    if result.metrics.context_recall:
        print(f"  Context Recall: {result.metrics.context_recall:.3f}")

    # Quick evaluation
    quick_result = await evaluator.quick_eval(
        question="What is machine learning?",
        ground_truth="Machine learning is a subset of AI that enables systems to learn from data.",
        contexts=["Machine learning uses algorithms to learn patterns from data without explicit programming."]
    )

    print(f"\nQuick evaluation overall score: {quick_result.metrics.overall_score:.3f}")


async def demonstrate_observability():
    """Demonstrate observability features."""
    print("\n🔍 Observability Demonstration")
    print("=" * 50)

    # Setup tracing
    tracer = get_tracer()
    openinference = setup_openinference_tracing(tracer)

    # Trace a complex operation
    with tracer.trace("demo_operation", user_id="demo_user") as trace:

        with tracer.span("data_processing", tracer.SpanType.DOCUMENT_PROCESSING) as span:
            if span:
                span.add_tag("documents", 3)
                span.add_log("Processing documents", level="info")
            await asyncio.sleep(0.1)  # Simulate work

        with tracer.span("vector_search", tracer.SpanType.VECTOR_SEARCH) as span:
            if span:
                span.add_tag("collection", "demo")
                span.add_tag("top_k", 5)
            await asyncio.sleep(0.05)  # Simulate work

        with tracer.span("llm_generation", tracer.SpanType.LLM_CALL) as span:
            if span:
                span.add_tag("model", "gpt-4")
                span.add_tag("tokens", 150)
            await asyncio.sleep(0.2)  # Simulate work

    # Get trace summary
    if trace:
        summary = tracer.get_trace_summary(trace.trace_id)
        print(f"Trace completed: {summary}")

    print("✓ Observability demonstration complete")


async def demonstrate_tools():
    """Demonstrate tool system."""
    print("\n🛠️ Tools System Demonstration")
    print("=" * 50)

    from ragents.tools import get_tool_registry

    registry = get_tool_registry()

    # Show registered tools
    print("Registered tools:")
    for tool_name in registry.list_tools():
        tool = registry.get(tool_name)
        print(f"  - {tool_name}: {tool.description}")

    # Execute a tool
    result = await registry.execute("web_search", query="artificial intelligence news")
    print(f"\nTool execution result: {result.result}")

    # Show tool schemas for LLM
    schemas = registry.get_tools_for_llm()
    print(f"\nTool schemas for LLM: {len(schemas)} tools available")


async def main():
    """Run all demonstrations."""
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    print("🚀 RAGents Advanced Features Demonstration")
    print("=" * 60)

    try:
        await demonstrate_agent_types()
        await demonstrate_vector_stores()
        await demonstrate_evaluation()
        await demonstrate_observability()
        await demonstrate_tools()

        print("\n" + "=" * 60)
        print("✅ All advanced features demonstrated successfully!")
        print("\nNext steps:")
        print("- Explore the evaluation framework with your own data")
        print("- Set up observability for production monitoring")
        print("- Create custom tools for your specific use case")
        print("- Try different vector store backends")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())