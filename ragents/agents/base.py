"""Base agent classes and interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..llm.client import LLMClient
from ..llm.types import ChatMessage, MessageRole
from ..rag.engine import RAGEngine
from ..query_rewriting.base import QueryRewriter, RewriteResult
from ..query_rewriting.strategies import CoTRewriter, ContextualRewriter


class AgentConfig(BaseModel):
    """Configuration for agents."""

    name: str
    description: str = ""
    max_iterations: int = 10
    enable_memory: bool = True
    memory_window: int = 20
    enable_tools: bool = True
    enable_rag: bool = True
    system_prompt: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    enable_reasoning: bool = True
    reasoning_depth: int = 3
    enable_query_rewriting: bool = True
    query_rewriter_strategy: Optional[str] = "contextual"
    adaptive_rewriting: bool = True


class AgentState(BaseModel):
    """Current state of an agent."""

    conversation_id: str
    turn_count: int = 0
    context: Dict[str, Any] = Field(default_factory=dict)
    memory: List[ChatMessage] = Field(default_factory=list)
    active_tools: List[str] = Field(default_factory=list)
    last_action: Optional[str] = None
    is_thinking: bool = False


class Agent(ABC):
    """Base agent class."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        rag_engine: Optional[RAGEngine] = None,
        query_rewriter: Optional[QueryRewriter] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.rag_engine = rag_engine
        self.state = AgentState(conversation_id="default")

        # Initialize query rewriter
        self.query_rewriter = query_rewriter
        if config.enable_query_rewriting and not query_rewriter:
            if config.query_rewriter_strategy == "cot":
                self.query_rewriter = CoTRewriter(llm_client)
            else:
                self.query_rewriter = ContextualRewriter(llm_client)

    @abstractmethod
    async def process_message(self, message: str) -> str:
        """Process a user message and return a response."""
        pass

    async def _add_to_memory(self, message: ChatMessage) -> None:
        """Add a message to conversation memory."""
        if self.config.enable_memory:
            self.state.memory.append(message)
            # Truncate memory if it exceeds window
            if len(self.state.memory) > self.config.memory_window:
                self.state.memory = self.state.memory[-self.config.memory_window :]

    async def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        base_prompt = (
            self.config.system_prompt
            or f"You are {self.config.name}, an intelligent assistant. {self.config.description}"
        )

        capabilities = []
        if self.config.enable_rag and self.rag_engine:
            capabilities.append("access to a knowledge base for information retrieval")
        if self.config.enable_tools:
            capabilities.append("access to various tools and functions")
        if self.config.enable_reasoning:
            capabilities.append("structured reasoning capabilities")

        if capabilities:
            base_prompt += f"\n\nYou have {', '.join(capabilities)}."

        return base_prompt

    async def reset_conversation(self) -> None:
        """Reset the conversation state."""
        self.state.memory.clear()
        self.state.turn_count = 0
        self.state.context.clear()

    def get_conversation_history(self) -> List[ChatMessage]:
        """Get the current conversation history."""
        return self.state.memory.copy()


class SimpleAgent(Agent):
    """Simple agent implementation without decision trees."""

    async def process_message(self, message: str) -> str:
        """Process a message with optional RAG integration."""
        await self._add_to_memory(ChatMessage(role=MessageRole.USER, content=message))

        # Rewrite query if needed
        processed_message = await self._rewrite_query_if_needed(message)

        # Check if we should use RAG
        should_use_rag = (
            self.config.enable_rag
            and self.rag_engine
            and await self._should_query_rag(processed_message)
        )

        context_info = ""
        if should_use_rag:
            rag_response = await self.rag_engine.query(processed_message)
            context_info = f"\n\nRelevant Information:\n{rag_response.answer}"

        # Build messages for LLM
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=await self._get_system_prompt())
        ]

        # Add conversation history
        messages.extend(self.state.memory[-self.config.memory_window :])

        # Add context if available
        if context_info:
            messages.append(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"Additional context: {context_info}",
                )
            )

        # Generate response
        response = await self.llm_client.acomplete(messages)
        response_content = response.content

        # Add response to memory
        await self._add_to_memory(
            ChatMessage(role=MessageRole.ASSISTANT, content=response_content)
        )

        self.state.turn_count += 1
        return response_content

    async def _should_query_rag(self, message: str) -> bool:
        """Determine if we should query the RAG system for this message."""
        # Simple heuristic - could be made more sophisticated
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        return any(word in message.lower() for word in question_words) or "?" in message

    async def _rewrite_query_if_needed(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Rewrite query using the configured rewriter if enabled."""
        if not self.config.enable_query_rewriting or not self.query_rewriter:
            return query

        try:
            # Prepare context for rewriting
            rewrite_context = context or {}
            if hasattr(self, 'state') and self.state.context:
                rewrite_context.update(self.state.context)

            # Add agent-specific context
            rewrite_context.update({
                "agent_type": self.config.name,
                "domain": "general",
                "situation": "information_seeking",
                "user_context": "assistant_interaction"
            })

            # Perform rewriting
            rewrite_result = await self.query_rewriter.rewrite(query, rewrite_context)

            # Use rewritten query if confidence is high enough
            if rewrite_result.confidence_score >= 0.6:
                return rewrite_result.rewritten_query

        except Exception:
            # Fall back to original query if rewriting fails
            pass

        return query