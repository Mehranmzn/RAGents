"""Unit tests for LLM client."""

import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from ragents.llm.client import LLMClient
from ragents.llm.types import ModelConfig, ModelProvider, ChatMessage, MessageRole


class TestResponse(BaseModel):
    """Test response model."""
    answer: str
    confidence: float


class TestLLMClient:
    """Test cases for LLMClient."""

    def test_init_openai_client(self, openai_model_config):
        """Test initialization with OpenAI provider."""
        client = LLMClient(openai_model_config)
        assert client.config.provider == ModelProvider.OPENAI
        assert client.config.model_name == "gpt-3.5-turbo"

    def test_init_anthropic_client(self, anthropic_model_config):
        """Test initialization with Anthropic provider."""
        client = LLMClient(anthropic_model_config)
        assert client.config.provider == ModelProvider.ANTHROPIC
        assert client.config.model_name == "claude-3-sonnet-20240229"

    def test_init_invalid_provider(self, mock_api_key):
        """Test initialization with invalid provider."""
        config = ModelConfig(
            provider="invalid_provider",  # This should cause an error
            model_name="test-model",
            api_key=mock_api_key
        )
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMClient(config)

    def test_prepare_messages(self, mock_llm_client, sample_messages):
        """Test message preparation for API."""
        api_messages = mock_llm_client._prepare_messages(sample_messages)

        assert len(api_messages) == 3
        assert api_messages[0]["role"] == "user"
        assert api_messages[0]["content"] == "Hello, how are you?"
        assert api_messages[1]["role"] == "assistant"
        assert api_messages[2]["role"] == "user"

    def test_get_completion_kwargs(self, mock_llm_client):
        """Test completion parameters generation."""
        kwargs = mock_llm_client._get_completion_kwargs()

        assert kwargs["model"] == "gpt-3.5-turbo"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1000
        assert "top_p" in kwargs
        assert "frequency_penalty" in kwargs

    def test_complete_without_response_model(self, mock_llm_client, sample_messages):
        """Test completion without structured output."""
        response = mock_llm_client.complete(sample_messages)

        assert hasattr(response, 'content')
        assert response.content == "Test response"
        assert response.model == "gpt-3.5-turbo"
        mock_llm_client._sync_client.chat.completions.create.assert_called_once()

    def test_complete_with_response_model(self, mock_llm_client, sample_messages):
        """Test completion with structured output."""
        response = mock_llm_client.complete(sample_messages, response_model=TestResponse)

        assert isinstance(response, TestResponse)
        assert response.content == "Test response"
        mock_llm_client._instructor_sync.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_acomplete_without_response_model(self, mock_llm_client, sample_messages):
        """Test async completion without structured output."""
        response = await mock_llm_client.acomplete(sample_messages)

        assert hasattr(response, 'content')
        assert response.content == "Test response"
        mock_llm_client._async_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_acomplete_with_response_model(self, mock_llm_client, sample_messages):
        """Test async completion with structured output."""
        response = await mock_llm_client.acomplete(sample_messages, response_model=TestResponse)

        assert isinstance(response, TestResponse)
        assert response.content == "Test response"
        mock_llm_client._instructor_async.chat.completions.create.assert_called_once()

    def test_complete_with_retries_success(self, mock_llm_client, sample_messages):
        """Test completion with retries when successful."""
        response = mock_llm_client.complete_with_retries(sample_messages, max_retries=3)

        assert hasattr(response, 'content')
        assert response.content == "Test response"

    def test_complete_with_retries_failure(self, mock_llm_client, sample_messages):
        """Test completion with retries when all attempts fail."""
        mock_llm_client._sync_client.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            mock_llm_client.complete_with_retries(sample_messages, max_retries=2)

    @pytest.mark.asyncio
    async def test_acomplete_with_retries_success(self, mock_llm_client, sample_messages):
        """Test async completion with retries when successful."""
        response = await mock_llm_client.acomplete_with_retries(sample_messages, max_retries=3)

        assert hasattr(response, 'content')
        assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_acomplete_with_retries_failure(self, mock_llm_client, sample_messages):
        """Test async completion with retries when all attempts fail."""
        mock_llm_client._async_client.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await mock_llm_client.acomplete_with_retries(sample_messages, max_retries=2)

    def test_parse_json_with_fallback_valid_json(self, mock_llm_client):
        """Test JSON parsing with valid JSON."""
        json_content = '{"answer": "test", "confidence": 0.9}'
        result = mock_llm_client.parse_json_with_fallback(json_content, TestResponse)

        assert isinstance(result, TestResponse)
        assert result.answer == "test"
        assert result.confidence == 0.9

    def test_parse_json_with_fallback_markdown(self, mock_llm_client):
        """Test JSON parsing from markdown code block."""
        markdown_content = '''
        Here's the response:
        ```json
        {"answer": "test", "confidence": 0.8}
        ```
        '''
        result = mock_llm_client.parse_json_with_fallback(markdown_content, TestResponse)

        assert isinstance(result, TestResponse)
        assert result.answer == "test"
        assert result.confidence == 0.8

    def test_parse_json_with_fallback_invalid(self, mock_llm_client):
        """Test JSON parsing with invalid JSON falls back to default."""
        invalid_content = "This is not JSON at all"
        result = mock_llm_client.parse_json_with_fallback(invalid_content, TestResponse)

        assert isinstance(result, TestResponse)
        # Should return default instance when parsing fails

    def test_fix_json_quotes(self, mock_llm_client):
        """Test JSON quote fixing."""
        content_with_smart_quotes = '{"answer": "test", "confidence": 0.9}'
        fixed = mock_llm_client._fix_json_quotes(content_with_smart_quotes)

        assert '"' in fixed
        assert '"' not in fixed
        assert '"' not in fixed