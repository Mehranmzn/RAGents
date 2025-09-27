"""Type-safe LLM client using instructor."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import instructor
from anthropic import Anthropic, AsyncAnthropic
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

from .types import ChatMessage, ModelConfig, ModelProvider, ModelResponse

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Type-safe LLM client with instructor integration."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._sync_client = None
        self._async_client = None
        self._instructor_sync = None
        self._instructor_async = None
        self._setup_clients()

    def _setup_clients(self) -> None:
        """Initialize the underlying LLM clients."""
        if self.config.provider == ModelProvider.OPENAI:
            self._sync_client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            self._async_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        elif self.config.provider == ModelProvider.ANTHROPIC:
            self._sync_client = Anthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            self._async_client = AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

        # Setup instructor clients
        self._instructor_sync = instructor.from_openai(self._sync_client)
        self._instructor_async = instructor.from_openai(self._async_client)

    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert ChatMessage objects to API format."""
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
                **({"tool_call_id": msg.tool_call_id} if msg.tool_call_id else {}),
                **({"name": msg.name} if msg.name else {}),
            }
            for msg in messages
        ]

    def _get_completion_kwargs(self) -> Dict[str, Any]:
        """Get completion parameters."""
        kwargs = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
        }
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens
        return kwargs

    def complete(
        self,
        messages: List[ChatMessage],
        response_model: Optional[Type[T]] = None,
        **kwargs: Any,
    ) -> Union[ModelResponse, T]:
        """Complete a chat conversation with optional structured output."""
        api_messages = self._prepare_messages(messages)
        completion_kwargs = {**self._get_completion_kwargs(), **kwargs}

        if response_model:
            # Use instructor for structured output
            response = self._instructor_sync.chat.completions.create(
                messages=api_messages,
                response_model=response_model,
                **completion_kwargs,
            )
            return response
        else:
            # Standard completion
            response = self._sync_client.chat.completions.create(
                messages=api_messages, **completion_kwargs
            )
            return ModelResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage=response.usage.model_dump() if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                tool_calls=getattr(response.choices[0].message, "tool_calls", None),
            )

    async def acomplete(
        self,
        messages: List[ChatMessage],
        response_model: Optional[Type[T]] = None,
        **kwargs: Any,
    ) -> Union[ModelResponse, T]:
        """Async completion with optional structured output."""
        api_messages = self._prepare_messages(messages)
        completion_kwargs = {**self._get_completion_kwargs(), **kwargs}

        if response_model:
            # Use instructor for structured output
            response = await self._instructor_async.chat.completions.create(
                messages=api_messages,
                response_model=response_model,
                **completion_kwargs,
            )
            return response
        else:
            # Standard completion
            response = await self._async_client.chat.completions.create(
                messages=api_messages, **completion_kwargs
            )
            return ModelResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage=response.usage.model_dump() if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                tool_calls=getattr(response.choices[0].message, "tool_calls", None),
            )

    def complete_with_retries(
        self,
        messages: List[ChatMessage],
        response_model: Optional[Type[T]] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Union[ModelResponse, T]:
        """Complete with retry logic for structured outputs."""
        for attempt in range(max_retries):
            try:
                return self.complete(messages, response_model, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                # Add exponential backoff if needed
                continue

    async def acomplete_with_retries(
        self,
        messages: List[ChatMessage],
        response_model: Optional[Type[T]] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Union[ModelResponse, T]:
        """Async complete with retry logic."""
        for attempt in range(max_retries):
            try:
                return await self.acomplete(messages, response_model, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                # Add exponential backoff
                await asyncio.sleep(2**attempt)
                continue

    def parse_json_with_fallback(self, content: str, expected_type: Type[T]) -> T:
        """Parse JSON content with multiple fallback strategies."""
        # Try direct parsing first
        try:
            data = json.loads(content)
            return expected_type.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting JSON from markdown code blocks
        import re

        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return expected_type.model_validate(data)
            except (json.JSONDecodeError, ValueError):
                pass

        # Try fixing common JSON issues
        fixed_content = self._fix_json_quotes(content)
        try:
            data = json.loads(fixed_content)
            return expected_type.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Last resort: return default instance
        return expected_type()

    def _fix_json_quotes(self, content: str) -> str:
        """Fix common JSON quote issues."""
        # Replace smart quotes with regular quotes
        content = content.replace(""", '"').replace(""", '"')
        content = content.replace("'", '"')

        # Try to fix unescaped quotes in strings
        import re

        # This is a simplified fix - a full implementation would be more robust
        return content