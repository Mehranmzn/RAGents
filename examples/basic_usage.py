"""Basic usage example for RAGents."""

import asyncio
import os
from pathlib import Path

from ragents import AgentConfig, DecisionTreeAgent, RAGConfig
from ragents.config.environment import get_llm_config_from_env
from ragents.llm.client import LLMClient
from ragents.rag.engine import RAGEngine


async def basic_agent_example():
    """Demonstrate basic agent usage."""
    print("🤖 Basic Agent Example")
    print("=" * 50)

    # Initialize configuration
    rag_config = RAGConfig.from_env()
    llm_config = get_llm_config_from_env()

    # Create LLM client
    llm_client = LLMClient(llm_config)

    # Set up RAG engine
    rag_engine = RAGEngine(rag_config, llm_client)

    # Create agent
    agent_config = AgentConfig(
        name="Basic Assistant",
        description="A helpful assistant for answering questions",
        enable_rag=True,
        enable_reasoning=True,
    )

    agent = DecisionTreeAgent(
        config=agent_config,
        llm_client=llm_client,
        rag_engine=rag_engine,
    )

    # Example interactions
    questions = [
        "Hello! How are you?",
        "What can you help me with?",
        "Can you explain what artificial intelligence is?",
    ]

    for question in questions:
        print(f"\n👤 User: {question}")
        response = await agent.process_message(question)
        print(f"🤖 Agent: {response}")


async def rag_with_documents_example():
    """Demonstrate RAG with document processing."""
    print("\n📚 RAG with Documents Example")
    print("=" * 50)

    # Setup (same as basic example)
    rag_config = RAGConfig.from_env()
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)
    rag_engine = RAGEngine(rag_config, llm_client)

    # Create a sample document
    sample_doc_path = Path("sample_document.txt")
    sample_content = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create
    intelligent machines that work and react like humans. Some of the activities
    computers with artificial intelligence are designed for include:

    - Speech recognition
    - Learning
    - Planning
    - Problem solving

    Machine Learning is a subset of AI that provides systems the ability to automatically
    learn and improve from experience without being explicitly programmed. Machine
    learning focuses on the development of computer programs that can access data and
    use it to learn for themselves.

    Deep Learning is a subset of machine learning that uses neural networks with
    multiple layers to model and understand complex patterns in data.
    """

    # Write sample document
    sample_doc_path.write_text(sample_content)

    try:
        # Add document to RAG system
        print("📄 Adding sample document to knowledge base...")
        document = await rag_engine.add_document(
            str(sample_doc_path),
            title="AI Overview",
            category="technology"
        )
        print(f"✅ Added document: {document.title}")

        # Create agent with RAG enabled
        agent_config = AgentConfig(
            name="Knowledge Assistant",
            description="An assistant with access to uploaded documents",
            enable_rag=True,
        )

        agent = DecisionTreeAgent(
            config=agent_config,
            llm_client=llm_client,
            rag_engine=rag_engine,
        )

        # Ask questions about the document
        rag_questions = [
            "What is artificial intelligence?",
            "How is machine learning related to AI?",
            "What activities are AI computers designed for?",
            "Explain the difference between machine learning and deep learning.",
        ]

        for question in rag_questions:
            print(f"\n👤 User: {question}")
            response = await agent.process_message(question)
            print(f"🤖 Agent: {response}")

    finally:
        # Clean up
        if sample_doc_path.exists():
            sample_doc_path.unlink()


async def structured_output_example():
    """Demonstrate structured LLM outputs with instructor."""
    print("\n🏗️ Structured Output Example")
    print("=" * 50)

    from ragents.llm.types import DocumentSummary, StructuredThought
    from ragents.llm.client import LLMClient
    from ragents.llm.types import ChatMessage, MessageRole

    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Example 1: Document Summary
    print("📋 Generating structured document summary...")

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are an expert at analyzing and summarizing documents."
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="""
            Analyze this text about renewable energy:

            Solar energy is one of the most promising renewable energy sources.
            It harnesses the power of the sun through photovoltaic cells and solar thermal systems.
            Wind energy uses turbines to convert wind motion into electricity.
            Both technologies have seen significant cost reductions in recent years,
            making them competitive with fossil fuels in many markets.
            """
        )
    ]

    summary = await llm_client.acomplete(messages, response_model=DocumentSummary)

    print(f"Title: {summary.title}")
    print(f"Main Topics: {', '.join(summary.main_topics)}")
    print(f"Key Points: {summary.key_points}")
    print(f"Entities: {', '.join(summary.entities)}")
    print(f"Summary: {summary.summary}")
    print(f"Confidence: {summary.confidence:.2f}")

    # Example 2: Structured Thinking
    print("\n🧠 Generating structured thought process...")

    thinking_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are an expert problem solver. Break down complex problems into structured thinking steps."
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="How can we reduce carbon emissions in urban transportation?"
        )
    ]

    thought = await llm_client.acomplete(thinking_messages, response_model=StructuredThought)

    print(f"Query Analysis: {thought.query_analysis}")
    print(f"Reasoning Steps:")
    for step in thought.reasoning_steps:
        print(f"  Step {step.step}: {step.description}")
        print(f"    Reasoning: {step.reasoning}")
        print(f"    Confidence: {step.confidence:.2f}")
    print(f"Final Answer: {thought.final_answer}")
    print(f"Overall Confidence: {thought.confidence_score:.2f}")


async def main():
    """Run all examples."""
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    try:
        await basic_agent_example()
        await rag_with_documents_example()
        await structured_output_example()

        print("\n✅ All examples completed successfully!")

    except Exception as e:
        print(f"❌ Error running examples: {e}")


if __name__ == "__main__":
    asyncio.run(main())