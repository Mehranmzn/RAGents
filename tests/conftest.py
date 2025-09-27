"""Pytest configuration and fixtures."""

import os
import tempfile
from typing import Dict, Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from pydantic import BaseModel

from ragents.llm.client import LLMClient
from ragents.llm.types import ModelConfig, ModelProvider, ChatMessage, MessageRole, ModelResponse
from ragents.agents.base import AgentConfig
from ragents.config.rag_config import RAGConfig
from ragents.rag.engine import RAGEngine


class MockResponse(BaseModel):
    """Mock response for testing."""
    content: str = "Test response"


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def openai_model_config(mock_api_key):
    """OpenAI model configuration for testing."""
    return ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key=mock_api_key,
        temperature=0.7,
        max_tokens=1000
    )


@pytest.fixture
def anthropic_model_config(mock_api_key):
    """Anthropic model configuration for testing."""
    return ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-sonnet-20240229",
        api_key=mock_api_key,
        temperature=0.7,
        max_tokens=1000
    )


@pytest.fixture
def mock_llm_client(openai_model_config):
    """Mock LLM client for testing."""
    client = LLMClient(openai_model_config)

    # Mock the actual API calls
    client._sync_client = MagicMock()
    client._async_client = MagicMock()
    client._instructor_sync = MagicMock()
    client._instructor_async = MagicMock()

    # Configure mock responses
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.model = "gpt-3.5-turbo"
    mock_response.usage = MagicMock()
    mock_response.usage.model_dump.return_value = {"total_tokens": 50}

    client._sync_client.chat.completions.create.return_value = mock_response
    client._async_client.chat.completions.create.return_value = mock_response
    client._instructor_sync.chat.completions.create.return_value = MockResponse()
    client._instructor_async.chat.completions.create.return_value = MockResponse()

    return client


@pytest.fixture
def agent_config():
    """Basic agent configuration for testing."""
    return AgentConfig(
        name="TestAgent",
        description="A test agent",
        max_iterations=5,
        enable_memory=True,
        memory_window=10,
        enable_tools=True,
        enable_rag=True,
        temperature=0.7
    )


@pytest.fixture
def rag_config():
    """RAG configuration for testing."""
    return RAGConfig(
        vector_store_type="chroma",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50,
        top_k=5
    )


@pytest.fixture
def mock_rag_engine(rag_config, mock_llm_client):
    """Mock RAG engine for testing."""
    engine = RAGEngine(rag_config, mock_llm_client)

    # Mock the vector store
    engine.vector_store = MagicMock()
    engine.vector_store.search.return_value = [
        {
            "id": "doc1",
            "content": "Sample document content",
            "metadata": {"source": "test.txt"},
            "score": 0.9
        }
    ]

    return engine


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="I'm doing well, thank you!"),
        ChatMessage(role=MessageRole.USER, content="Can you help me with a question?")
    ]


@pytest.fixture
def temp_directory():
    """Temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "This is a sample document about machine learning.",
            "metadata": {"title": "ML Basics", "author": "Test Author"}
        },
        {
            "id": "doc2",
            "content": "Python is a versatile programming language.",
            "metadata": {"title": "Python Guide", "author": "Test Author"}
        },
        {
            "id": "doc3",
            "content": "Natural language processing enables computers to understand text.",
            "metadata": {"title": "NLP Overview", "author": "Test Author"}
        }
    ]


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    yield
    # Cleanup is handled by pytest automatically


@pytest.fixture
def mock_embedding_function():
    """Mock embedding function for testing."""
    def mock_embed(texts):
        # Return simple mock embeddings (dimension=384 for all-MiniLM-L6-v2)
        import numpy as np
        return np.random.rand(len(texts), 384).tolist()

    return mock_embed