"""ReAct (Reasoning + Acting) agent implementation."""

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..llm.types import ChatMessage, MessageRole
from .base import Agent, AgentConfig


class ReActStep(str, Enum):
    """ReAct reasoning steps."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"


class ReActAction(BaseModel):
    """A ReAct action with parameters."""
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class ReActTrace(BaseModel):
    """A single step in ReAct reasoning."""
    step_number: int
    step_type: ReActStep
    content: str
    action: Optional[ReActAction] = None
    observation: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ReActSession(BaseModel):
    """A complete ReAct reasoning session."""
    query: str
    traces: List[ReActTrace] = Field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    total_steps: int = 0
    max_iterations: int = 10


class ToolRegistry:
    """Registry for ReAct tools following Elysia's tool decorator pattern."""

    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self.tool_descriptions: Dict[str, str] = {}

    def register_tool(self, name: str, func: callable, description: str):
        """Register a tool for use in ReAct agent."""
        self.tools[name] = func
        self.tool_descriptions[name] = description

    def get_tool_list(self) -> str:
        """Get formatted tool list for prompt."""
        tool_list = []
        for name, desc in self.tool_descriptions.items():
            tool_list.append(f"- {name}: {desc}")
        return "\n".join(tool_list)

    async def execute_tool(self, action: ReActAction) -> str:
        """Execute a tool action."""
        if action.name not in self.tools:
            return f"Error: Tool '{action.name}' not found."

        try:
            tool_func = self.tools[action.name]
            # Handle both sync and async tools
            if hasattr(tool_func, '__call__'):
                if hasattr(tool_func, '__await__'):
                    result = await tool_func(**action.parameters)
                else:
                    result = tool_func(**action.parameters)
                return str(result)
        except Exception as e:
            return f"Error executing {action.name}: {str(e)}"

        return "Tool execution completed."


# Tool decorator inspired by Elysia
def tool(name: str, description: str):
    """Decorator to register tools for ReAct agent."""
    def decorator(func):
        func._tool_name = name
        func._tool_description = description
        return func
    return decorator


class ReActAgent(Agent):
    """ReAct agent with reasoning and acting capabilities."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client,
        rag_engine=None,
        max_iterations: int = 10,
    ):
        super().__init__(config, llm_client, rag_engine)
        self.max_iterations = max_iterations
        self.tool_registry = ToolRegistry()
        self.current_session: Optional[ReActSession] = None
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools."""
        @tool("rag_search", "Search the knowledge base for information")
        async def rag_search(query: str) -> str:
            if self.rag_engine:
                response = await self.rag_engine.query(query)
                return f"Found: {response.answer}"
            return "RAG system not available."

        @tool("calculator", "Perform mathematical calculations")
        def calculator(expression: str) -> str:
            try:
                # Safe evaluation of mathematical expressions
                import ast
                import operator

                # Supported operators
                ops = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Pow: operator.pow,
                    ast.USub: operator.neg,
                }

                def eval_expr(node):
                    if isinstance(node, ast.Constant):
                        return node.value
                    elif isinstance(node, ast.BinOp):
                        return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                    elif isinstance(node, ast.UnaryOp):
                        return ops[type(node.op)](eval_expr(node.operand))
                    else:
                        raise TypeError(node)

                result = eval_expr(ast.parse(expression, mode='eval').body)
                return f"Result: {result}"
            except Exception as e:
                return f"Calculation error: {str(e)}"

        @tool("memory_recall", "Recall information from conversation history")
        def memory_recall(topic: str) -> str:
            # Simple keyword search in conversation history
            relevant_messages = []
            for msg in self.state.memory:
                if topic.lower() in msg.content.lower():
                    relevant_messages.append(msg.content[:100] + "...")

            if relevant_messages:
                return f"Found {len(relevant_messages)} relevant messages: " + "; ".join(relevant_messages[:3])
            return "No relevant information found in conversation history."

        # Register tools
        self.tool_registry.register_tool("rag_search", rag_search, "Search the knowledge base for information")
        self.tool_registry.register_tool("calculator", calculator, "Perform mathematical calculations")
        self.tool_registry.register_tool("memory_recall", memory_recall, "Recall information from conversation history")

    def register_tool(self, func):
        """Register a tool function decorated with @tool."""
        if hasattr(func, '_tool_name') and hasattr(func, '_tool_description'):
            self.tool_registry.register_tool(
                func._tool_name,
                func,
                func._tool_description
            )
        else:
            raise ValueError("Function must be decorated with @tool")

    async def process_message(self, message: str) -> str:
        """Process message using ReAct reasoning pattern."""
        await self._add_to_memory(ChatMessage(role=MessageRole.USER, content=message))

        # Initialize ReAct session
        self.current_session = ReActSession(
            query=message,
            max_iterations=self.max_iterations
        )

        # Execute ReAct loop
        result = await self._react_loop()

        # Add to memory
        await self._add_to_memory(
            ChatMessage(role=MessageRole.ASSISTANT, content=result)
        )

        self.state.turn_count += 1
        return result

    async def _react_loop(self) -> str:
        """Main ReAct reasoning loop."""
        session = self.current_session
        step_number = 1

        while step_number <= self.max_iterations:
            # Thought step
            thought = await self._generate_thought(step_number)
            session.traces.append(ReActTrace(
                step_number=step_number,
                step_type=ReActStep.THOUGHT,
                content=thought
            ))

            # Check if we should take an action or provide answer
            action = await self._decide_action(thought)

            if action and action.name != "answer":
                # Action step
                session.traces.append(ReActTrace(
                    step_number=step_number,
                    step_type=ReActStep.ACTION,
                    content=f"Action: {action.name} with parameters: {action.parameters}",
                    action=action
                ))

                # Observation step
                observation = await self.tool_registry.execute_tool(action)
                session.traces.append(ReActTrace(
                    step_number=step_number,
                    step_type=ReActStep.OBSERVATION,
                    content=observation,
                    observation=observation
                ))

                step_number += 1
            else:
                # Final answer
                final_answer = await self._generate_final_answer()
                session.traces.append(ReActTrace(
                    step_number=step_number,
                    step_type=ReActStep.ANSWER,
                    content=final_answer
                ))
                session.final_answer = final_answer
                session.success = True
                session.total_steps = step_number
                break

        return session.final_answer or "I couldn't provide a complete answer within the iteration limit."

    async def _generate_thought(self, step_number: int) -> str:
        """Generate a reasoning thought."""
        session = self.current_session
        previous_context = self._format_previous_traces()

        system_prompt = f"""
You are a helpful assistant using ReAct (Reasoning + Acting) methodology.

Available tools:
{self.tool_registry.get_tool_list()}

Follow this pattern:
1. Think about what you need to do
2. Decide if you need to use a tool or can provide the final answer
3. If using a tool, specify the tool name and parameters

Current step: {step_number}
Query: {session.query}

Previous reasoning:
{previous_context}

Think step by step about what to do next.
"""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content="What should I think about next?")
        ]

        response = await self.llm_client.acomplete(messages)
        return response.content

    async def _decide_action(self, thought: str) -> Optional[ReActAction]:
        """Decide what action to take based on the thought."""
        action_prompt = f"""
Based on this thought: "{thought}"

Available tools:
{self.tool_registry.get_tool_list()}

Decide if you should:
1. Use a tool (specify tool name and parameters as JSON)
2. Provide the final answer

If using a tool, respond with JSON format:
{{"action": "tool_name", "parameters": {{"param1": "value1"}}, "reasoning": "why this tool"}}

If ready to answer, respond with:
{{"action": "answer", "reasoning": "I have enough information to answer"}}
"""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=action_prompt),
            ChatMessage(role=MessageRole.USER, content="What action should I take?")
        ]

        response = await self.llm_client.acomplete(messages)

        # Parse JSON response
        try:
            action_data = json.loads(response.content)
            return ReActAction(
                name=action_data.get("action", "answer"),
                parameters=action_data.get("parameters", {}),
                reasoning=action_data.get("reasoning", "")
            )
        except json.JSONDecodeError:
            # Fallback parsing
            if "answer" in response.content.lower():
                return ReActAction(name="answer", reasoning="Ready to provide final answer")
            return None

    async def _generate_final_answer(self) -> str:
        """Generate the final answer based on all observations."""
        session = self.current_session
        context = self._format_all_observations()

        final_prompt = f"""
Based on the following reasoning and observations, provide a comprehensive final answer to the user's query.

Query: {session.query}

Reasoning and observations:
{context}

Provide a clear, helpful answer based on the information gathered.
"""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=final_prompt),
            ChatMessage(role=MessageRole.USER, content="Provide the final answer.")
        ]

        response = await self.llm_client.acomplete(messages)
        return response.content

    def _format_previous_traces(self) -> str:
        """Format previous traces for context."""
        if not self.current_session or not self.current_session.traces:
            return "No previous steps."

        formatted_traces = []
        for trace in self.current_session.traces[-3:]:  # Last 3 traces
            if trace.step_type == ReActStep.THOUGHT:
                formatted_traces.append(f"Thought {trace.step_number}: {trace.content}")
            elif trace.step_type == ReActStep.ACTION:
                formatted_traces.append(f"Action {trace.step_number}: {trace.content}")
            elif trace.step_type == ReActStep.OBSERVATION:
                formatted_traces.append(f"Observation {trace.step_number}: {trace.content}")

        return "\n".join(formatted_traces)

    def _format_all_observations(self) -> str:
        """Format all observations for final answer generation."""
        if not self.current_session:
            return ""

        formatted = []
        for trace in self.current_session.traces:
            if trace.step_type == ReActStep.OBSERVATION and trace.observation:
                formatted.append(f"- {trace.observation}")

        return "\n".join(formatted) if formatted else "No observations collected."

    def get_reasoning_trace(self) -> Optional[ReActSession]:
        """Get the complete reasoning trace."""
        return self.current_session

    def export_trace(self) -> Dict[str, Any]:
        """Export reasoning trace for analysis."""
        if not self.current_session:
            return {}

        return {
            "query": self.current_session.query,
            "steps": [
                {
                    "step": trace.step_number,
                    "type": trace.step_type,
                    "content": trace.content,
                    "action": trace.action.model_dump() if trace.action else None,
                    "observation": trace.observation,
                }
                for trace in self.current_session.traces
            ],
            "final_answer": self.current_session.final_answer,
            "success": self.current_session.success,
            "total_steps": self.current_session.total_steps,
        }