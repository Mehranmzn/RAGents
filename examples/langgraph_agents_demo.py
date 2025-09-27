"""Demonstration of LangGraph-based agents in RAGents."""

import asyncio
import os
from pathlib import Path

from ragents import (
    LLMClient,
    RAGEngine,
    RAGConfig,
    LangGraphAgent,
    LangGraphReActAgent,
    LangGraphMultiAgent,
    AgentConfig,
    AgentRole,
    AgentDefinition,
    create_research_team,
    create_analysis_team,
)
from ragents.llm.types import ModelConfig, ModelProvider
from ragents.tools.base import ToolRegistry


async def create_mock_llm_client() -> LLMClient:
    """Create a mock LLM client for demonstration."""
    config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY", "demo-key"),
        temperature=0.7
    )
    return LLMClient(config)


async def create_mock_rag_engine(llm_client: LLMClient) -> RAGEngine:
    """Create a mock RAG engine for demonstration."""
    rag_config = RAGConfig(
        vector_store_type="memory",
        collection_name="demo_collection",
        chunk_size=500,
        chunk_overlap=50,
        top_k=3
    )
    return RAGEngine(rag_config, llm_client)


async def demo_langgraph_agent():
    """Demonstrate basic LangGraph agent functionality."""
    print("=== LangGraph Agent Demo ===")

    # Setup
    llm_client = await create_mock_llm_client()
    rag_engine = await create_mock_rag_engine(llm_client)

    # Create agent configuration
    agent_config = AgentConfig(
        name="DemoAgent",
        description="A demonstration agent using LangGraph workflows",
        enable_rag=True,
        enable_reasoning=True,
        enable_query_rewriting=True,
        max_iterations=5
    )

    # Create LangGraph agent
    agent = LangGraphAgent(
        config=agent_config,
        llm_client=llm_client,
        rag_engine=rag_engine
    )

    # Process a message
    try:
        result = await agent.process_message(
            "What are the key principles of machine learning?",
            thread_id="demo_thread"
        )

        print(f"Agent Response: {result.response}")
        print(f"Processing Steps: {result.execution_metadata['processing_steps']}")
        print(f"RAG Used: {result.execution_metadata.get('rag_used', False)}")
        print(f"Reasoning Used: {result.execution_metadata.get('reasoning_used', False)}")

    except Exception as e:
        print(f"Demo error (expected with mock setup): {e}")

    print()


async def demo_langgraph_react_agent():
    """Demonstrate LangGraph ReAct agent functionality."""
    print("=== LangGraph ReAct Agent Demo ===")

    # Setup
    llm_client = await create_mock_llm_client()
    rag_engine = await create_mock_rag_engine(llm_client)

    # Create tool registry
    tool_registry = ToolRegistry()

    # Register a simple demo tool
    async def calculator(expression: str) -> str:
        """Simple calculator tool."""
        try:
            # WARNING: eval is dangerous - only for demo purposes
            result = eval(expression)
            return f"The result is: {result}"
        except Exception as e:
            return f"Calculation error: {e}"

    tool_registry.register(calculator)

    # Create ReAct agent configuration
    react_config = AgentConfig(
        name="ReActAgent",
        description="A reasoning and acting agent using LangGraph",
        enable_rag=True,
        enable_tools=True,
        max_iterations=10
    )

    # Create LangGraph ReAct agent
    react_agent = LangGraphReActAgent(
        config=react_config,
        llm_client=llm_client,
        rag_engine=rag_engine,
        tool_registry=tool_registry
    )

    # Process a complex query requiring reasoning and action
    try:
        result = await react_agent.process_message(
            "I need to understand machine learning and then calculate 15 * 23",
            thread_id="react_demo"
        )

        print(f"ReAct Response: {result.response}")
        print(f"Total Iterations: {result.execution_metadata['total_iterations']}")
        print(f"Actions Taken: {result.execution_metadata['actions_taken']}")
        print(f"Max Iterations Reached: {result.execution_metadata['max_iterations_reached']}")

    except Exception as e:
        print(f"Demo error (expected with mock setup): {e}")

    print()


async def demo_multi_agent_system():
    """Demonstrate multi-agent system functionality."""
    print("=== Multi-Agent System Demo ===")

    # Setup
    llm_client = await create_mock_llm_client()
    rag_engine = await create_mock_rag_engine(llm_client)

    # Create research team
    research_team = create_research_team(llm_client, rag_engine)

    # Process a complex research query
    try:
        result = await research_team.process_message(
            "Conduct a comprehensive analysis of renewable energy technologies, "
            "their current state, and future prospects for replacing fossil fuels",
            thread_id="research_demo"
        )

        print(f"Multi-Agent Response: {result.response}")
        print(f"Routing Decision: {result.execution_metadata['routing_decision']}")
        print(f"Agents Used: {result.execution_metadata['agents_used']}")
        print(f"Collaboration Rounds: {result.execution_metadata['collaboration_rounds']}")

    except Exception as e:
        print(f"Demo error (expected with mock setup): {e}")

    print()


async def demo_custom_multi_agent():
    """Demonstrate creating a custom multi-agent setup."""
    print("=== Custom Multi-Agent Demo ===")

    # Setup
    llm_client = await create_mock_llm_client()
    rag_engine = await create_mock_rag_engine(llm_client)

    # Create coordinator configuration
    coordinator_config = AgentConfig(
        name="CustomCoordinator",
        description="Coordinates specialized agents for technical analysis"
    )

    # Create multi-agent system
    multi_agent = LangGraphMultiAgent(coordinator_config, llm_client)

    # Register specialized agents

    # Technical researcher
    tech_researcher_config = AgentConfig(
        name="TechResearcher",
        description="Specializes in technical research and data gathering",
        enable_rag=True,
        enable_tools=True
    )
    multi_agent.register_agent(AgentDefinition(
        name="tech_researcher",
        role=AgentRole.RESEARCHER,
        config=tech_researcher_config,
        llm_client=llm_client,
        rag_engine=rag_engine,
        specialization="technical_analysis"
    ))

    # Data analyst
    data_analyst_config = AgentConfig(
        name="DataAnalyst",
        description="Analyzes data patterns and trends",
        enable_reasoning=True,
        reasoning_depth=4
    )
    multi_agent.register_agent(AgentDefinition(
        name="data_analyst",
        role=AgentRole.ANALYST,
        config=data_analyst_config,
        llm_client=llm_client,
        specialization="data_analysis"
    ))

    # Technical writer
    tech_writer_config = AgentConfig(
        name="TechWriter",
        description="Creates technical documentation and reports"
    )
    multi_agent.register_agent(AgentDefinition(
        name="tech_writer",
        role=AgentRole.WRITER,
        config=tech_writer_config,
        llm_client=llm_client,
        specialization="technical_writing"
    ))

    # Process a technical analysis task
    try:
        result = await multi_agent.process_message(
            "Analyze the performance characteristics of different database systems "
            "and create a technical comparison report",
            thread_id="custom_demo"
        )

        print(f"Custom Multi-Agent Response: {result.response}")
        print(f"System Type: {result.execution_metadata['system_type']}")
        print(f"Processing Steps: {result.execution_metadata['processing_steps']}")

    except Exception as e:
        print(f"Demo error (expected with mock setup): {e}")

    print()


async def demo_workflow_state_management():
    """Demonstrate LangGraph state management features."""
    print("=== Workflow State Management Demo ===")

    # Setup
    llm_client = await create_mock_llm_client()

    agent_config = AgentConfig(
        name="StatefulAgent",
        description="Demonstrates stateful conversations",
        enable_memory=True,
        memory_window=10
    )

    agent = LangGraphAgent(
        config=agent_config,
        llm_client=llm_client
    )

    thread_id = "stateful_demo"

    # Simulate a multi-turn conversation
    conversation = [
        "Hello, I'm working on a machine learning project.",
        "What are some good algorithms for classification?",
        "Can you explain more about decision trees?",
        "How do I evaluate the performance of my model?"
    ]

    for i, message in enumerate(conversation):
        try:
            print(f"Turn {i+1}: {message}")
            result = await agent.process_message(message, thread_id=thread_id)
            print(f"Response: {result.response[:100]}...")

            # Show conversation history
            if i == len(conversation) - 1:  # Last turn
                history = await agent.get_conversation_history(thread_id)
                print(f"Conversation History Length: {len(history)} messages")

        except Exception as e:
            print(f"Demo error (expected with mock setup): {e}")

    # Reset conversation
    try:
        await agent.reset_conversation(thread_id)
        print("Conversation reset successfully")
    except Exception as e:
        print(f"Reset error (expected with mock setup): {e}")

    print()


async def demo_error_handling():
    """Demonstrate error handling in LangGraph workflows."""
    print("=== Error Handling Demo ===")

    # Setup with intentionally problematic configuration
    llm_client = await create_mock_llm_client()

    agent_config = AgentConfig(
        name="ErrorProneAgent",
        description="Demonstrates error handling",
        max_iterations=2  # Low limit to trigger max iterations
    )

    agent = LangGraphAgent(
        config=agent_config,
        llm_client=llm_client
    )

    # Test with various error scenarios
    test_cases = [
        "This is a normal query that should work",
        "",  # Empty query
        "A" * 10000,  # Very long query
    ]

    for i, test_query in enumerate(test_cases):
        try:
            print(f"Test {i+1}: Query length = {len(test_query)}")
            result = await agent.process_message(
                test_query or "Empty query test",
                thread_id=f"error_test_{i}"
            )
            print(f"Success: {result.response[:50]}...")
            print(f"Error occurred: {result.execution_metadata['error_occurred']}")

        except Exception as e:
            print(f"Exception caught: {type(e).__name__}: {e}")

    print()


async def main():
    """Run all LangGraph agent demonstrations."""
    print("🚀 RAGents LangGraph Agents Demonstration")
    print("=" * 50)

    # Note: These demos will show the structure and workflow
    # but may not produce actual LLM responses without proper API keys

    demos = [
        demo_langgraph_agent,
        demo_langgraph_react_agent,
        demo_multi_agent_system,
        demo_custom_multi_agent,
        demo_workflow_state_management,
        demo_error_handling,
    ]

    for demo in demos:
        try:
            await demo()
        except Exception as e:
            print(f"Demo {demo.__name__} failed: {e}")

        print("-" * 30)

    print("✅ All demos completed!")
    print("\nKey Features Demonstrated:")
    print("• LangGraph workflow orchestration")
    print("• ReAct reasoning and acting patterns")
    print("• Multi-agent collaboration")
    print("• Stateful conversation management")
    print("• Error handling and recovery")
    print("• Tool integration")
    print("• RAG integration")


if __name__ == "__main__":
    asyncio.run(main())