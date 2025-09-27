"""Integration tests for agent workflows."""

import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest

from ragents.agents.base import SimpleAgent, AgentConfig
from ragents.agents.react_agent import ReActAgent
from ragents.rag.engine import RAGEngine
from ragents.llm.types import ChatMessage, MessageRole


class TestAgentWorkflowIntegration:
    """Integration tests for agent workflows."""

    @pytest.fixture
    def workflow_agent_config(self):
        """Agent configuration for workflow testing."""
        return AgentConfig(
            name="WorkflowAgent",
            description="Agent for testing complex workflows",
            max_iterations=10,
            enable_memory=True,
            memory_window=15,
            enable_tools=True,
            enable_rag=True,
            enable_reasoning=True,
            reasoning_depth=3,
            temperature=0.7
        )

    @pytest.fixture
    def mock_rag_workflow(self, rag_config, mock_llm_client):
        """Mock RAG engine for workflow testing."""
        rag_engine = RAGEngine(rag_config, mock_llm_client)

        # Mock comprehensive RAG responses
        async def mock_query(query_text, **kwargs):
            # Simulate different responses based on query content
            if "machine learning" in query_text.lower():
                response = MagicMock()
                response.answer = "Machine learning is a subset of AI that enables computers to learn from data."
                response.sources = [
                    MagicMock(chunk=MagicMock(content="ML definition content"), score=0.9),
                    MagicMock(chunk=MagicMock(content="ML examples content"), score=0.8)
                ]
                response.confidence = 0.92
                response.reasoning = "Found comprehensive ML information"
                response.metadata = {"num_sources": 2}
                return response
            elif "python" in query_text.lower():
                response = MagicMock()
                response.answer = "Python is a high-level programming language known for its simplicity."
                response.sources = [
                    MagicMock(chunk=MagicMock(content="Python features content"), score=0.85)
                ]
                response.confidence = 0.87
                response.reasoning = "Found Python programming information"
                response.metadata = {"num_sources": 1}
                return response
            else:
                response = MagicMock()
                response.answer = "I found some general information about your question."
                response.sources = []
                response.confidence = 0.3
                response.reasoning = "Limited information available"
                response.metadata = {"num_sources": 0}
                return response

        rag_engine.query = AsyncMock(side_effect=mock_query)
        return rag_engine

    @pytest.mark.asyncio
    async def test_simple_agent_conversation_workflow(self, workflow_agent_config, mock_llm_client, mock_rag_workflow):
        """Test a complete conversation workflow with simple agent."""
        agent = SimpleAgent(workflow_agent_config, mock_llm_client, mock_rag_workflow)

        # Simulate a multi-turn conversation
        conversation_turns = [
            ("Hello, can you help me understand machine learning?", "machine learning"),
            ("What are the main types of machine learning?", "machine learning"),
            ("Can you give me an example of supervised learning?", "machine learning"),
            ("Thanks! Now tell me about Python programming.", "python"),
            ("How do I get started with Python?", "python"),
            ("That's very helpful, thank you!", "general")
        ]

        responses = []
        for question, expected_topic in conversation_turns:
            response = await agent.process_message(question)
            responses.append(response)

            # Verify response is generated
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0

        # Verify conversation state
        assert agent.state.turn_count == len(conversation_turns)
        assert len(agent.state.memory) == len(conversation_turns) * 2  # User + assistant messages

        # Verify RAG was called for questions
        assert mock_rag_workflow.query.call_count >= 5  # Should have queried RAG for most questions

        # Verify memory contains conversation history
        memory_contents = " ".join([msg.content for msg in agent.state.memory])
        assert "machine learning" in memory_contents
        assert "Python" in memory_contents

    @pytest.mark.asyncio
    async def test_agent_with_tools_workflow(self, workflow_agent_config, mock_llm_client):
        """Test agent workflow with tool usage."""
        agent = SimpleAgent(workflow_agent_config, mock_llm_client)

        # Mock tool registry and tools
        from ragents.tools.base import ToolRegistry

        mock_tool_registry = ToolRegistry()

        # Define a mock tool
        async def mock_calculator(expression: str) -> str:
            """Calculate mathematical expressions."""
            # Simple mock calculator
            if "2+2" in expression:
                return "4"
            elif "10*5" in expression:
                return "50"
            else:
                return "42"

        mock_tool_registry.register(mock_calculator)

        # Mock LLM to suggest tool usage
        def mock_llm_with_tools(*args, **kwargs):
            messages = args[0] if args else kwargs.get('messages', [])
            last_message = messages[-1].content if messages else ""

            if "calculate" in last_message.lower() or "math" in last_message.lower():
                # Simulate tool usage suggestion
                response = MagicMock()
                response.content = "I'll help you calculate that. Let me use the calculator tool."
                response.tool_calls = [
                    {
                        "id": "calc_1",
                        "type": "function",
                        "function": {
                            "name": "mock_calculator",
                            "arguments": '{"expression": "2+2"}'
                        }
                    }
                ]
                return response
            else:
                response = MagicMock()
                response.content = "I can help you with various tasks including calculations."
                return response

        mock_llm_client.acomplete = AsyncMock(side_effect=mock_llm_with_tools)

        # Test tool workflow
        response = await agent.process_message("Can you calculate 2+2 for me?")

        assert response is not None
        # Would verify tool was called if tool integration was complete

    @pytest.mark.asyncio
    async def test_react_agent_workflow(self, workflow_agent_config, mock_llm_client, mock_rag_workflow):
        """Test ReAct agent reasoning workflow."""
        try:
            react_agent = ReActAgent(workflow_agent_config, mock_llm_client, mock_rag_workflow)

            # Mock ReAct-style responses
            def mock_react_response(*args, **kwargs):
                messages = args[0] if args else kwargs.get('messages', [])
                last_message = messages[-1].content if messages else ""

                if "machine learning" in last_message.lower():
                    response = MagicMock()
                    response.content = """
                    Thought: The user is asking about machine learning. I should search for information about ML.
                    Action: search_knowledge_base
                    Action Input: machine learning definition and types
                    Observation: Machine learning is a subset of AI that enables computers to learn from data.
                    Thought: I have good information about machine learning. I can provide a comprehensive answer.
                    Final Answer: Machine learning is a subset of artificial intelligence that enables computers to learn and make predictions from data without being explicitly programmed for each task.
                    """
                    return response
                else:
                    response = MagicMock()
                    response.content = "I'll think through this step by step."
                    return response

            mock_llm_client.acomplete = AsyncMock(side_effect=mock_react_response)

            # Test ReAct workflow
            response = await react_agent.process_message("What is machine learning and how does it work?")

            assert response is not None
            assert "machine learning" in response.lower()

        except ImportError:
            # ReActAgent might not be fully implemented
            pytest.skip("ReActAgent not available")

    @pytest.mark.asyncio
    async def test_agent_error_recovery_workflow(self, workflow_agent_config, mock_llm_client, mock_rag_workflow):
        """Test agent error recovery in workflows."""
        agent = SimpleAgent(workflow_agent_config, mock_llm_client, mock_rag_workflow)

        # Test LLM failure recovery
        mock_llm_client.acomplete.side_effect = Exception("LLM service unavailable")

        with pytest.raises(Exception):
            await agent.process_message("Test message")

        # Reset and test RAG failure recovery
        mock_llm_client.acomplete.side_effect = None
        mock_llm_client.acomplete.return_value = MagicMock(content="Fallback response")
        mock_rag_workflow.query.side_effect = Exception("RAG service unavailable")

        # Agent should handle RAG failure gracefully
        response = await agent.process_message("What is machine learning?")
        assert response == "Fallback response"

    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self, mock_llm_client, mock_rag_workflow):
        """Test workflow with multiple specialized agents."""
        # Create specialized agents
        ml_agent_config = AgentConfig(
            name="MLExpert",
            description="Expert in machine learning topics",
            enable_rag=True,
            system_prompt="You are an expert in machine learning and data science."
        )

        programming_agent_config = AgentConfig(
            name="ProgrammingHelper",
            description="Expert in programming and software development",
            enable_rag=True,
            system_prompt="You are an expert programmer who helps with coding questions."
        )

        ml_agent = SimpleAgent(ml_agent_config, mock_llm_client, mock_rag_workflow)
        programming_agent = SimpleAgent(programming_agent_config, mock_llm_client, mock_rag_workflow)

        # Configure different responses for each agent
        def mock_ml_response(*args, **kwargs):
            response = MagicMock()
            response.content = "As an ML expert, I can tell you that machine learning involves..."
            return response

        def mock_programming_response(*args, **kwargs):
            response = MagicMock()
            response.content = "As a programming expert, I recommend starting with Python because..."
            return response

        # Test routing queries to appropriate agents
        ml_query = "What are the best machine learning algorithms for classification?"
        programming_query = "How do I write a Python function to process data?"

        # Mock different LLM clients for each agent
        ml_llm_client = MagicMock()
        ml_llm_client.acomplete = AsyncMock(side_effect=mock_ml_response)
        ml_agent.llm_client = ml_llm_client

        programming_llm_client = MagicMock()
        programming_llm_client.acomplete = AsyncMock(side_effect=mock_programming_response)
        programming_agent.llm_client = programming_llm_client

        # Process queries with appropriate agents
        ml_response = await ml_agent.process_message(ml_query)
        programming_response = await programming_agent.process_message(programming_query)

        assert "ML expert" in ml_response
        assert "programming expert" in programming_response

    @pytest.mark.asyncio
    async def test_agent_memory_and_context_workflow(self, workflow_agent_config, mock_llm_client):
        """Test agent memory and context management in workflows."""
        agent = SimpleAgent(workflow_agent_config, mock_llm_client)

        # Mock LLM to reference previous context
        call_count = 0

        def mock_contextual_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            messages = args[0] if args else kwargs.get('messages', [])

            # Check if previous context is included
            if call_count == 1:
                response = MagicMock()
                response.content = "I understand you're asking about Python. It's a versatile programming language."
                return response
            elif call_count == 2:
                # Should have context from previous turn
                full_context = " ".join([msg.content for msg in messages])
                if "Python" in full_context:
                    response = MagicMock()
                    response.content = "Given our previous discussion about Python, I can tell you about its data science libraries."
                    return response
                else:
                    response = MagicMock()
                    response.content = "Let me help with data science libraries."
                    return response
            else:
                response = MagicMock()
                response.content = "I'm here to help with any questions."
                return response

        mock_llm_client.acomplete = AsyncMock(side_effect=mock_contextual_response)

        # Test contextual conversation
        response1 = await agent.process_message("Tell me about Python programming.")
        response2 = await agent.process_message("What are its main data science libraries?")

        assert "Python" in response1
        assert "previous discussion" in response2 or "data science" in response2

        # Verify memory contains both turns
        assert len(agent.state.memory) == 4  # 2 user + 2 assistant messages
        assert agent.state.turn_count == 2

    @pytest.mark.asyncio
    async def test_agent_reasoning_workflow(self, mock_llm_client):
        """Test agent reasoning capabilities in workflows."""
        reasoning_config = AgentConfig(
            name="ReasoningAgent",
            enable_reasoning=True,
            reasoning_depth=3,
            enable_rag=False  # Focus on reasoning without RAG
        )

        agent = SimpleAgent(reasoning_config, mock_llm_client)

        # Mock structured reasoning response
        def mock_reasoning_response(*args, **kwargs):
            messages = args[0] if args else kwargs.get('messages', [])
            last_message = messages[-1].content if messages else ""

            if "complex" in last_message.lower() or "analyze" in last_message.lower():
                response = MagicMock()
                response.content = """
                Let me analyze this step by step:
                1. First, I need to understand the problem
                2. Then, I'll consider different approaches
                3. Finally, I'll provide a reasoned conclusion

                Based on this analysis, the best approach would be...
                """
                return response
            else:
                response = MagicMock()
                response.content = "I'll help you think through this problem."
                return response

        mock_llm_client.acomplete = AsyncMock(side_effect=mock_reasoning_response)

        # Test reasoning workflow
        response = await agent.process_message("Can you analyze this complex problem for me?")

        assert "step by step" in response or "analyze" in response
        assert "1." in response  # Should show structured reasoning

    @pytest.mark.asyncio
    async def test_concurrent_agent_workflows(self, workflow_agent_config, mock_llm_client, mock_rag_workflow):
        """Test concurrent agent operations."""
        # Create multiple agents
        agents = [
            SimpleAgent(workflow_agent_config, mock_llm_client, mock_rag_workflow)
            for _ in range(3)
        ]

        # Mock responses for concurrent testing
        response_counter = 0

        def mock_concurrent_response(*args, **kwargs):
            nonlocal response_counter
            response_counter += 1
            response = MagicMock()
            response.content = f"Response from agent operation {response_counter}"
            return response

        mock_llm_client.acomplete = AsyncMock(side_effect=mock_concurrent_response)

        # Test concurrent message processing
        questions = [
            "What is machine learning?",
            "How does Python work?",
            "Explain artificial intelligence."
        ]

        # Process messages concurrently
        tasks = [
            agent.process_message(question)
            for agent, question in zip(agents, questions)
        ]

        responses = await asyncio.gather(*tasks)

        # Verify all responses were generated
        assert len(responses) == 3
        for response in responses:
            assert response is not None
            assert "Response from agent operation" in response

        # Verify each agent maintained its own state
        for i, agent in enumerate(agents):
            assert agent.state.turn_count == 1
            assert len(agent.state.memory) == 2  # User + assistant message

    @pytest.mark.asyncio
    async def test_agent_workflow_performance(self, workflow_agent_config, mock_llm_client, mock_rag_workflow):
        """Test agent workflow performance characteristics."""
        agent = SimpleAgent(workflow_agent_config, mock_llm_client, mock_rag_workflow)

        # Mock fast responses
        mock_llm_client.acomplete = AsyncMock(return_value=MagicMock(content="Quick response"))

        # Test processing speed
        import time

        questions = [f"Question {i}" for i in range(10)]

        start_time = time.time()

        for question in questions:
            await agent.process_message(question)

        total_time = time.time() - start_time

        # Verify reasonable performance (with mocking, should be very fast)
        assert total_time < 2.0  # Should process 10 messages in under 2 seconds
        assert agent.state.turn_count == 10

    @pytest.mark.asyncio
    async def test_agent_workflow_with_query_rewriting(self, workflow_agent_config, mock_llm_client, mock_rag_workflow):
        """Test agent workflow with query rewriting enabled."""
        workflow_agent_config.enable_query_rewriting = True
        agent = SimpleAgent(workflow_agent_config, mock_llm_client, mock_rag_workflow)

        # Mock query rewriter
        from ragents.query_rewriting.base import RewriteResult

        mock_rewrite_result = RewriteResult(
            original_query="What is ML?",
            rewritten_query="What is machine learning and how does it work?",
            confidence_score=0.9,
            reasoning="Expanded abbreviation and added context",
            metadata={"strategy": "expansion"}
        )

        agent.query_rewriter.rewrite = AsyncMock(return_value=mock_rewrite_result)

        # Test workflow with query rewriting
        response = await agent.process_message("What is ML?")

        assert response is not None
        # Verify rewriter was called
        agent.query_rewriter.rewrite.assert_called_once()

        # Verify RAG was called with rewritten query
        mock_rag_workflow.query.assert_called()
        call_args = mock_rag_workflow.query.call_args[0][0]
        assert "machine learning" in call_args.lower()  # Should use expanded form