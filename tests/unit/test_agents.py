"""Unit tests for agent classes."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from ragents.agents.base import Agent, AgentConfig, AgentState, SimpleAgent
from ragents.llm.types import ChatMessage, MessageRole


class TestAgentConfig:
    """Test cases for AgentConfig."""

    def test_agent_config_defaults(self):
        """Test agent configuration with defaults."""
        config = AgentConfig(name="TestAgent")

        assert config.name == "TestAgent"
        assert config.description == ""
        assert config.max_iterations == 10
        assert config.enable_memory is True
        assert config.memory_window == 20
        assert config.temperature == 0.7

    def test_agent_config_custom_values(self):
        """Test agent configuration with custom values."""
        config = AgentConfig(
            name="CustomAgent",
            description="A custom test agent",
            max_iterations=5,
            enable_memory=False,
            temperature=0.3
        )

        assert config.name == "CustomAgent"
        assert config.description == "A custom test agent"
        assert config.max_iterations == 5
        assert config.enable_memory is False
        assert config.temperature == 0.3

    def test_agent_config_validation(self):
        """Test agent configuration validation."""
        # Temperature should be within valid range
        with pytest.raises(ValueError):
            AgentConfig(name="TestAgent", temperature=3.0)  # Too high

        with pytest.raises(ValueError):
            AgentConfig(name="TestAgent", temperature=-1.0)  # Too low


class TestAgentState:
    """Test cases for AgentState."""

    def test_agent_state_defaults(self):
        """Test agent state with defaults."""
        state = AgentState(conversation_id="test-123")

        assert state.conversation_id == "test-123"
        assert state.turn_count == 0
        assert state.context == {}
        assert state.memory == []
        assert state.active_tools == []
        assert state.last_action is None
        assert state.is_thinking is False

    def test_agent_state_with_data(self):
        """Test agent state with initial data."""
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        state = AgentState(
            conversation_id="test-456",
            turn_count=5,
            context={"key": "value"},
            memory=messages,
            active_tools=["tool1", "tool2"]
        )

        assert state.conversation_id == "test-456"
        assert state.turn_count == 5
        assert state.context["key"] == "value"
        assert len(state.memory) == 1
        assert state.active_tools == ["tool1", "tool2"]


class TestSimpleAgent:
    """Test cases for SimpleAgent."""

    def test_simple_agent_initialization(self, agent_config, mock_llm_client):
        """Test simple agent initialization."""
        agent = SimpleAgent(agent_config, mock_llm_client)

        assert agent.config == agent_config
        assert agent.llm_client == mock_llm_client
        assert agent.state.conversation_id == "default"
        assert agent.query_rewriter is not None  # Should be initialized

    def test_simple_agent_initialization_with_rag(self, agent_config, mock_llm_client, mock_rag_engine):
        """Test simple agent initialization with RAG."""
        agent = SimpleAgent(agent_config, mock_llm_client, mock_rag_engine)

        assert agent.rag_engine == mock_rag_engine

    @pytest.mark.asyncio
    async def test_add_to_memory(self, agent_config, mock_llm_client):
        """Test adding messages to memory."""
        agent = SimpleAgent(agent_config, mock_llm_client)
        message = ChatMessage(role=MessageRole.USER, content="Test message")

        await agent._add_to_memory(message)

        assert len(agent.state.memory) == 1
        assert agent.state.memory[0] == message

    @pytest.mark.asyncio
    async def test_memory_window_truncation(self, mock_llm_client):
        """Test memory truncation when exceeding window size."""
        config = AgentConfig(name="TestAgent", memory_window=3)
        agent = SimpleAgent(config, mock_llm_client)

        # Add more messages than the window size
        for i in range(5):
            message = ChatMessage(role=MessageRole.USER, content=f"Message {i}")
            await agent._add_to_memory(message)

        # Should only keep the last 3 messages
        assert len(agent.state.memory) == 3
        assert agent.state.memory[-1].content == "Message 4"
        assert agent.state.memory[0].content == "Message 2"

    @pytest.mark.asyncio
    async def test_memory_disabled(self, mock_llm_client):
        """Test agent with memory disabled."""
        config = AgentConfig(name="TestAgent", enable_memory=False)
        agent = SimpleAgent(config, mock_llm_client)
        message = ChatMessage(role=MessageRole.USER, content="Test message")

        await agent._add_to_memory(message)

        # Memory should remain empty when disabled
        assert len(agent.state.memory) == 0

    @pytest.mark.asyncio
    async def test_get_system_prompt_basic(self, agent_config, mock_llm_client):
        """Test basic system prompt generation."""
        agent = SimpleAgent(agent_config, mock_llm_client)
        prompt = await agent._get_system_prompt()

        assert "TestAgent" in prompt
        assert "intelligent assistant" in prompt

    @pytest.mark.asyncio
    async def test_get_system_prompt_with_capabilities(self, agent_config, mock_llm_client, mock_rag_engine):
        """Test system prompt with capabilities."""
        agent = SimpleAgent(agent_config, mock_llm_client, mock_rag_engine)
        prompt = await agent._get_system_prompt()

        assert "knowledge base" in prompt
        assert "tools and functions" in prompt
        assert "reasoning capabilities" in prompt

    @pytest.mark.asyncio
    async def test_get_system_prompt_custom(self, mock_llm_client):
        """Test custom system prompt."""
        config = AgentConfig(
            name="CustomAgent",
            system_prompt="You are a specialized assistant for testing."
        )
        agent = SimpleAgent(config, mock_llm_client)
        prompt = await agent._get_system_prompt()

        assert "specialized assistant for testing" in prompt

    @pytest.mark.asyncio
    async def test_reset_conversation(self, agent_config, mock_llm_client):
        """Test conversation reset."""
        agent = SimpleAgent(agent_config, mock_llm_client)

        # Add some state
        agent.state.turn_count = 5
        agent.state.context["key"] = "value"
        await agent._add_to_memory(ChatMessage(role=MessageRole.USER, content="Test"))

        await agent.reset_conversation()

        assert agent.state.turn_count == 0
        assert agent.state.context == {}
        assert len(agent.state.memory) == 0

    def test_get_conversation_history(self, agent_config, mock_llm_client):
        """Test getting conversation history."""
        agent = SimpleAgent(agent_config, mock_llm_client)
        message = ChatMessage(role=MessageRole.USER, content="Test")
        agent.state.memory.append(message)

        history = agent.get_conversation_history()

        assert len(history) == 1
        assert history[0] == message
        # Should return a copy, not the original
        assert history is not agent.state.memory

    @pytest.mark.asyncio
    async def test_should_query_rag_question_words(self, agent_config, mock_llm_client):
        """Test RAG query decision with question words."""
        agent = SimpleAgent(agent_config, mock_llm_client)

        assert await agent._should_query_rag("What is machine learning?") is True
        assert await agent._should_query_rag("How does this work?") is True
        assert await agent._should_query_rag("Why is the sky blue?") is True

    @pytest.mark.asyncio
    async def test_should_query_rag_question_mark(self, agent_config, mock_llm_client):
        """Test RAG query decision with question mark."""
        agent = SimpleAgent(agent_config, mock_llm_client)

        assert await agent._should_query_rag("Is this correct?") is True
        assert await agent._should_query_rag("Can you help me?") is True

    @pytest.mark.asyncio
    async def test_should_query_rag_statement(self, agent_config, mock_llm_client):
        """Test RAG query decision with statements."""
        agent = SimpleAgent(agent_config, mock_llm_client)

        assert await agent._should_query_rag("This is a statement.") is False
        assert await agent._should_query_rag("Hello there.") is False

    @pytest.mark.asyncio
    async def test_process_message_basic(self, agent_config, mock_llm_client):
        """Test basic message processing."""
        agent = SimpleAgent(agent_config, mock_llm_client)

        response = await agent.process_message("Hello, how are you?")

        assert response == "Test response"
        assert agent.state.turn_count == 1
        assert len(agent.state.memory) == 2  # User message + assistant response

    @pytest.mark.asyncio
    async def test_process_message_with_rag(self, agent_config, mock_llm_client, mock_rag_engine):
        """Test message processing with RAG."""
        agent = SimpleAgent(agent_config, mock_llm_client, mock_rag_engine)

        # Mock RAG response
        mock_rag_response = MagicMock()
        mock_rag_response.answer = "RAG context information"
        mock_rag_engine.query = AsyncMock(return_value=mock_rag_response)

        response = await agent.process_message("What is machine learning?")

        assert response == "Test response"
        mock_rag_engine.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_without_rag(self, mock_llm_client):
        """Test message processing without RAG."""
        config = AgentConfig(name="TestAgent", enable_rag=False)
        agent = SimpleAgent(config, mock_llm_client)

        response = await agent.process_message("What is machine learning?")

        assert response == "Test response"
        # RAG should not be called

    @pytest.mark.asyncio
    async def test_rewrite_query_enabled(self, agent_config, mock_llm_client):
        """Test query rewriting when enabled."""
        agent = SimpleAgent(agent_config, mock_llm_client)

        # Mock the query rewriter
        mock_rewrite_result = MagicMock()
        mock_rewrite_result.rewritten_query = "rewritten query"
        mock_rewrite_result.confidence_score = 0.8
        agent.query_rewriter = MagicMock()
        agent.query_rewriter.rewrite = AsyncMock(return_value=mock_rewrite_result)

        result = await agent._rewrite_query_if_needed("original query")

        assert result == "rewritten query"
        agent.query_rewriter.rewrite.assert_called_once()

    @pytest.mark.asyncio
    async def test_rewrite_query_low_confidence(self, agent_config, mock_llm_client):
        """Test query rewriting with low confidence."""
        agent = SimpleAgent(agent_config, mock_llm_client)

        # Mock low confidence rewrite result
        mock_rewrite_result = MagicMock()
        mock_rewrite_result.rewritten_query = "rewritten query"
        mock_rewrite_result.confidence_score = 0.3  # Low confidence
        agent.query_rewriter = MagicMock()
        agent.query_rewriter.rewrite = AsyncMock(return_value=mock_rewrite_result)

        result = await agent._rewrite_query_if_needed("original query")

        assert result == "original query"  # Should fall back to original

    @pytest.mark.asyncio
    async def test_rewrite_query_disabled(self, mock_llm_client):
        """Test query rewriting when disabled."""
        config = AgentConfig(name="TestAgent", enable_query_rewriting=False)
        agent = SimpleAgent(config, mock_llm_client)

        result = await agent._rewrite_query_if_needed("original query")

        assert result == "original query"

    @pytest.mark.asyncio
    async def test_rewrite_query_exception(self, agent_config, mock_llm_client):
        """Test query rewriting with exception handling."""
        agent = SimpleAgent(agent_config, mock_llm_client)

        # Mock exception in query rewriter
        agent.query_rewriter = MagicMock()
        agent.query_rewriter.rewrite = AsyncMock(side_effect=Exception("Rewrite error"))

        result = await agent._rewrite_query_if_needed("original query")

        assert result == "original query"  # Should fall back to original on error