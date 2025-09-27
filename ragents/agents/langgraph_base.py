"""LangGraph-based agent implementations."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TypedDict, Union, Annotated, Literal
from dataclasses import dataclass

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .base import AgentConfig
from ..llm.client import LLMClient
from ..llm.types import ChatMessage, MessageRole
from ..rag.engine import RAGEngine
from ..query_rewriting.base import QueryRewriter


class LangGraphAgentState(TypedDict):
    """State schema for LangGraph agents."""

    messages: Annotated[List[BaseMessage], add_messages]
    current_query: str
    processed_query: str
    context_info: str
    rag_results: Optional[Dict[str, Any]]
    reasoning_steps: List[str]
    agent_metadata: Dict[str, Any]
    should_continue: bool
    error_message: Optional[str]


@dataclass
class LangGraphAgentResult:
    """Result from LangGraph agent execution."""

    response: str
    state: LangGraphAgentState
    execution_metadata: Dict[str, Any]


class LangGraphAgent:
    """Base LangGraph agent with workflow orchestration."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        rag_engine: Optional[RAGEngine] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.rag_engine = rag_engine
        self.query_rewriter = query_rewriter
        self.checkpointer = checkpointer or MemorySaver()

        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(LangGraphAgentState)

        # Add nodes
        workflow.add_node("process_input", self._process_input_node)
        workflow.add_node("query_rewriting", self._query_rewriting_node)
        workflow.add_node("rag_retrieval", self._rag_retrieval_node)
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Define the flow
        workflow.add_edge(START, "process_input")
        workflow.add_conditional_edges(
            "process_input",
            self._should_rewrite_query,
            {
                "rewrite": "query_rewriting",
                "skip_rewrite": "rag_retrieval"
            }
        )
        workflow.add_edge("query_rewriting", "rag_retrieval")
        workflow.add_conditional_edges(
            "rag_retrieval",
            self._should_use_reasoning,
            {
                "reasoning": "reasoning",
                "direct_response": "generate_response"
            }
        )
        workflow.add_edge("reasoning", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("error_handler", END)

        return workflow

    async def _process_input_node(self, state: LangGraphAgentState) -> LangGraphAgentState:
        """Process the input message."""
        try:
            # Get the latest human message
            human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            if human_messages:
                current_query = human_messages[-1].content
                state["current_query"] = current_query
                state["processed_query"] = current_query

            # Initialize metadata
            state["agent_metadata"] = {
                "agent_name": self.config.name,
                "enable_rag": self.config.enable_rag,
                "enable_reasoning": self.config.enable_reasoning,
                "processing_steps": []
            }

            state["should_continue"] = True
            state["error_message"] = None

        except Exception as e:
            state["error_message"] = str(e)
            state["should_continue"] = False

        return state

    async def _query_rewriting_node(self, state: LangGraphAgentState) -> LangGraphAgentState:
        """Rewrite the query if needed."""
        try:
            if self.query_rewriter and self.config.enable_query_rewriting:
                rewrite_result = await self.query_rewriter.rewrite(
                    state["current_query"],
                    context=state["agent_metadata"]
                )

                if rewrite_result.confidence_score >= 0.6:
                    state["processed_query"] = rewrite_result.rewritten_query
                    state["agent_metadata"]["query_rewritten"] = True
                    state["agent_metadata"]["rewrite_confidence"] = rewrite_result.confidence_score
                else:
                    state["agent_metadata"]["query_rewritten"] = False

                state["agent_metadata"]["processing_steps"].append("query_rewriting")

        except Exception as e:
            state["error_message"] = f"Query rewriting failed: {str(e)}"

        return state

    async def _rag_retrieval_node(self, state: LangGraphAgentState) -> LangGraphAgentState:
        """Perform RAG retrieval."""
        try:
            state["context_info"] = ""
            state["rag_results"] = None

            if (self.config.enable_rag and
                self.rag_engine and
                self._should_query_rag(state["processed_query"])):

                rag_response = await self.rag_engine.query(state["processed_query"])

                state["context_info"] = f"Retrieved Information:\n{rag_response.answer}"
                state["rag_results"] = {
                    "answer": rag_response.answer,
                    "sources": [source.__dict__ if hasattr(source, '__dict__') else str(source)
                              for source in rag_response.sources],
                    "confidence": rag_response.confidence,
                    "reasoning": rag_response.reasoning
                }

                state["agent_metadata"]["rag_used"] = True
                state["agent_metadata"]["rag_confidence"] = rag_response.confidence
                state["agent_metadata"]["processing_steps"].append("rag_retrieval")
            else:
                state["agent_metadata"]["rag_used"] = False

        except Exception as e:
            state["error_message"] = f"RAG retrieval failed: {str(e)}"
            state["agent_metadata"]["rag_used"] = False

        return state

    async def _reasoning_node(self, state: LangGraphAgentState) -> LangGraphAgentState:
        """Perform structured reasoning."""
        try:
            state["reasoning_steps"] = []

            if self.config.enable_reasoning:
                # Create reasoning prompt
                reasoning_prompt = self._build_reasoning_prompt(
                    state["processed_query"],
                    state["context_info"]
                )

                # Get reasoning from LLM
                reasoning_messages = [
                    SystemMessage(content="You are an expert at structured reasoning. Break down complex problems step by step."),
                    HumanMessage(content=reasoning_prompt)
                ]

                # Convert to our message format
                chat_messages = self._convert_to_chat_messages(reasoning_messages)
                response = await self.llm_client.acomplete(chat_messages)

                # Parse reasoning steps
                reasoning_text = response.content
                steps = self._parse_reasoning_steps(reasoning_text)
                state["reasoning_steps"] = steps

                state["agent_metadata"]["reasoning_used"] = True
                state["agent_metadata"]["processing_steps"].append("reasoning")
            else:
                state["agent_metadata"]["reasoning_used"] = False

        except Exception as e:
            state["error_message"] = f"Reasoning failed: {str(e)}"
            state["reasoning_steps"] = []

        return state

    async def _generate_response_node(self, state: LangGraphAgentState) -> LangGraphAgentState:
        """Generate the final response."""
        try:
            # Build the response prompt
            response_prompt = self._build_response_prompt(state)

            # Create messages for LLM
            messages = [
                SystemMessage(content=await self._get_system_prompt()),
            ]

            # Add conversation history (limit to recent messages)
            recent_messages = state["messages"][-self.config.memory_window:]
            messages.extend(recent_messages)

            # Add context if available
            if state["context_info"]:
                messages.append(SystemMessage(content=f"Additional context: {state['context_info']}"))

            # Add reasoning if available
            if state["reasoning_steps"]:
                reasoning_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(state["reasoning_steps"])])
                messages.append(SystemMessage(content=f"Reasoning steps:\n{reasoning_text}"))

            # Generate response
            chat_messages = self._convert_to_chat_messages(messages)
            response = await self.llm_client.acomplete(chat_messages)

            # Add AI response to messages
            ai_message = AIMessage(content=response.content)
            state["messages"].append(ai_message)

            state["agent_metadata"]["processing_steps"].append("generate_response")
            state["agent_metadata"]["response_generated"] = True

        except Exception as e:
            state["error_message"] = f"Response generation failed: {str(e)}"
            # Add fallback response
            fallback_response = "I apologize, but I encountered an error while processing your request."
            state["messages"].append(AIMessage(content=fallback_response))

        return state

    async def _error_handler_node(self, state: LangGraphAgentState) -> LangGraphAgentState:
        """Handle errors in the workflow."""
        error_response = f"An error occurred: {state.get('error_message', 'Unknown error')}"
        state["messages"].append(AIMessage(content=error_response))
        state["agent_metadata"]["error_handled"] = True
        return state

    def _should_rewrite_query(self, state: LangGraphAgentState) -> Literal["rewrite", "skip_rewrite"]:
        """Determine if query should be rewritten."""
        if (self.config.enable_query_rewriting and
            self.query_rewriter and
            not state.get("error_message")):
            return "rewrite"
        return "skip_rewrite"

    def _should_use_reasoning(self, state: LangGraphAgentState) -> Literal["reasoning", "direct_response"]:
        """Determine if reasoning should be used."""
        if (self.config.enable_reasoning and
            not state.get("error_message") and
            self._is_complex_query(state["processed_query"])):
            return "reasoning"
        return "direct_response"

    def _should_query_rag(self, message: str) -> bool:
        """Determine if we should query the RAG system."""
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        return any(word in message.lower() for word in question_words) or "?" in message

    def _is_complex_query(self, query: str) -> bool:
        """Determine if a query is complex and needs reasoning."""
        complexity_indicators = [
            "analyze", "compare", "explain why", "what are the differences",
            "how does", "what causes", "evaluate", "assess", "determine"
        ]
        return any(indicator in query.lower() for indicator in complexity_indicators)

    def _build_reasoning_prompt(self, query: str, context: str) -> str:
        """Build a prompt for structured reasoning."""
        prompt = f"""
        Please analyze the following query and provide structured reasoning:

        Query: {query}

        Available Context:
        {context if context else "No additional context available"}

        Provide your analysis in clear, numbered steps that break down:
        1. What the query is asking
        2. Key concepts or information needed
        3. How to approach finding an answer
        4. Any assumptions or considerations

        Format your response as numbered steps.
        """
        return prompt

    def _build_response_prompt(self, state: LangGraphAgentState) -> str:
        """Build the final response prompt."""
        query = state["processed_query"]
        context = state.get("context_info", "")
        reasoning = state.get("reasoning_steps", [])

        prompt = f"User query: {query}"

        if context:
            prompt += f"\n\nRelevant context: {context}"

        if reasoning:
            prompt += f"\n\nReasoning steps: {'; '.join(reasoning)}"

        return prompt

    def _parse_reasoning_steps(self, reasoning_text: str) -> List[str]:
        """Parse reasoning steps from text."""
        lines = reasoning_text.strip().split('\n')
        steps = []

        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering and clean up
                cleaned = line.lstrip('0123456789.-• ').strip()
                if cleaned:
                    steps.append(cleaned)

        return steps

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

    def _convert_to_chat_messages(self, messages: List[BaseMessage]) -> List[ChatMessage]:
        """Convert LangChain messages to our ChatMessage format."""
        chat_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = MessageRole.USER
            elif isinstance(msg, AIMessage):
                role = MessageRole.ASSISTANT
            elif isinstance(msg, SystemMessage):
                role = MessageRole.SYSTEM
            else:
                role = MessageRole.USER  # fallback

            chat_messages.append(ChatMessage(role=role, content=msg.content))

        return chat_messages

    async def process_message(
        self,
        message: str,
        thread_id: str = "default",
        config: Optional[Dict[str, Any]] = None
    ) -> LangGraphAgentResult:
        """Process a message using the LangGraph workflow."""
        # Create initial state
        initial_state = LangGraphAgentState(
            messages=[HumanMessage(content=message)],
            current_query="",
            processed_query="",
            context_info="",
            rag_results=None,
            reasoning_steps=[],
            agent_metadata={},
            should_continue=True,
            error_message=None
        )

        # Run the workflow
        result = await self.app.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}, **(config or {})}
        )

        # Extract the response
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        response = ai_messages[-1].content if ai_messages else "No response generated"

        return LangGraphAgentResult(
            response=response,
            state=result,
            execution_metadata={
                "workflow_completed": True,
                "processing_steps": result["agent_metadata"].get("processing_steps", []),
                "rag_used": result["agent_metadata"].get("rag_used", False),
                "reasoning_used": result["agent_metadata"].get("reasoning_used", False),
                "error_occurred": result.get("error_message") is not None
            }
        )

    async def get_conversation_history(self, thread_id: str = "default") -> List[BaseMessage]:
        """Get conversation history for a thread."""
        try:
            # Get the latest state from checkpointer
            checkpoint = await self.checkpointer.aget({"configurable": {"thread_id": thread_id}})
            if checkpoint and checkpoint.get("channel_values"):
                state = checkpoint["channel_values"]
                return state.get("messages", [])
        except:
            pass
        return []

    async def reset_conversation(self, thread_id: str = "default") -> None:
        """Reset conversation for a thread."""
        # This would depend on the checkpointer implementation
        # For MemorySaver, we can clear the specific thread
        if hasattr(self.checkpointer, 'storage'):
            thread_key = {"configurable": {"thread_id": thread_id}}
            if thread_key in self.checkpointer.storage:
                del self.checkpointer.storage[thread_key]