"""Type definitions for LLM interactions."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class MessageRole(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """Structured chat message."""
    role: MessageRole
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ModelConfig(BaseModel):
    """Configuration for LLM models."""
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class ModelResponse(BaseModel):
    """Structured response from LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ReasoningStep(BaseModel):
    """A single step in the reasoning process."""
    step: int
    description: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class StructuredThought(BaseModel):
    """Structured thinking process for complex queries."""
    query_analysis: str
    reasoning_steps: List[ReasoningStep]
    final_answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources_needed: List[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    """Structured document summary."""
    title: str
    main_topics: List[str]
    key_points: List[str]
    entities: List[str]
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)


class QueryPlan(BaseModel):
    """Plan for executing a complex query."""
    original_query: str
    sub_queries: List[str]
    search_strategy: Literal["sequential", "parallel", "hybrid"]
    expected_sources: List[str]
    reasoning: str