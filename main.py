"""Main entry point for RAGents demo."""

import asyncio
import os
from pathlib import Path

from ragents import AgentConfig, DecisionTreeAgent, RAGConfig, RAGEngine
from ragents.config.environment import get_llm_config_from_env
from ragents.llm.client import LLMClient


async def main():
    """Demo of RAGents capabilities."""
    print("🤖 Welcome to RAGents - Advanced Agentic RAG Framework")

    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    try:
        # Initialize configuration
        print("⚙️  Initializing configuration...")
        rag_config = RAGConfig.from_env()
        llm_config = get_llm_config_from_env()

        # Initialize LLM client
        print(f"🧠 Connecting to {llm_config.provider} with model {llm_config.model_name}...")
        llm_client = LLMClient(llm_config)

        # Initialize RAG engine
        print("📚 Setting up RAG engine...")
        rag_engine = RAGEngine(rag_config, llm_client)

        # Initialize agent
        agent_config = AgentConfig(
            name="RAGent Assistant",
            description="An intelligent assistant with access to knowledge bases and reasoning capabilities",
            enable_rag=True,
            enable_reasoning=True,
        )

        print("🤖 Creating decision tree agent...")
        agent = DecisionTreeAgent(
            config=agent_config,
            llm_client=llm_client,
            rag_engine=rag_engine,
        )

        # Demo interaction
        print("\n✅ RAGents initialized successfully!")
        print("💬 You can now interact with the agent. Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                print("🤔 Agent is thinking...")
                response = await agent.process_message(user_input)
                print(f"🤖 Agent: {response}\n")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

    except Exception as e:
        print(f"❌ Failed to initialize RAGents: {e}")
        return


def demo_document_processing():
    """Demo document processing capabilities."""
    print("\n📄 Document Processing Demo")
    print("This demo shows how to add documents to the RAG system.")

    # Example of how to add documents
    example_docs = [
        "example_docs/sample.pdf",
        "example_docs/readme.md",
        "example_docs/data.csv"
    ]

    print("Example documents that could be processed:")
    for doc in example_docs:
        print(f"  - {doc}")

    print("\nTo add documents, use:")
    print("  await rag_engine.add_document('path/to/document.pdf')")
    print("  documents = await rag_engine.add_documents_batch(['doc1.pdf', 'doc2.md'])")


if __name__ == "__main__":
    print("🚀 Starting RAGents...")
    asyncio.run(main())
