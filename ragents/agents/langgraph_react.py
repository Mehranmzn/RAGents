"""LangGraph-based ReAct agent implementation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal, Union
from dataclasses import dataclass

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .langgraph_base import LangGraphAgent, LangGraphAgentState, LangGraphAgentResult
from .base import AgentConfig
from ..llm.client import LLMClient
from ..llm.types import ChatMessage, MessageRole
from ..rag.engine import RAGEngine
from ..query_rewriting.base import QueryRewriter
from ..tools.base import ToolRegistry


class ReActAgentState(LangGraphAgentState):
    """Extended state for ReAct agent with action/observation tracking."""

    current_thought: str
    current_action: Optional[str]
    current_action_input: Optional[str]
    current_observation: Optional[str]
    action_history: List[Dict[str, Any]]
    iteration_count: int
    final_answer: Optional[str]
    max_iterations_reached: bool


@dataclass
class ReActStep:
    """Represents a single ReAct reasoning step."""

    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    is_final: bool = False


class LangGraphReActAgent(LangGraphAgent):
    """LangGraph-based ReAct (Reasoning and Acting) agent."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        rag_engine: Optional[RAGEngine] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        tool_registry: Optional[ToolRegistry] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self.tool_registry = tool_registry or ToolRegistry()
        super().__init__(config, llm_client, rag_engine, query_rewriter, checkpointer)

    def _build_workflow(self) -> StateGraph:
        """Build the ReAct workflow graph."""
        workflow = StateGraph(ReActAgentState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("think", self._think_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("observe", self._observe_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Define the flow
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "think")

        workflow.add_conditional_edges(
            "think",
            self._should_act_or_finish,
            {
                "act": "act",
                "finish": "finalize",
                "error": "error_handler"
            }
        )

        workflow.add_edge("act", "observe")

        workflow.add_conditional_edges(
            "observe",
            self._should_continue_or_finish,
            {
                "continue": "think",
                "finish": "finalize",
                "max_iterations": "finalize",
                "error": "error_handler"
            }
        )

        workflow.add_edge("finalize", END)
        workflow.add_edge("error_handler", END)

        return workflow

    async def _initialize_node(self, state: ReActAgentState) -> ReActAgentState:
        """Initialize the ReAct agent state."""
        try:
            # Get the latest human message
            human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            if human_messages:
                current_query = human_messages[-1].content
                state["current_query"] = current_query
                state["processed_query"] = current_query

            # Initialize ReAct-specific state
            state["current_thought"] = ""
            state["current_action"] = None
            state["current_action_input"] = None
            state["current_observation"] = None
            state["action_history"] = []
            state["iteration_count"] = 0
            state["final_answer"] = None
            state["max_iterations_reached"] = False

            # Initialize metadata
            state["agent_metadata"] = {
                "agent_name": self.config.name,
                "agent_type": "ReAct",
                "available_tools": list(self.tool_registry.get_available_tools()),
                "processing_steps": []
            }

            state["should_continue"] = True
            state["error_message"] = None

        except Exception as e:
            state["error_message"] = str(e)
            state["should_continue"] = False

        return state

    async def _think_node(self, state: ReActAgentState) -> ReActAgentState:
        """ReAct thinking step."""
        try:
            state["iteration_count"] += 1

            # Check max iterations
            if state["iteration_count"] > self.config.max_iterations:
                state["max_iterations_reached"] = True
                return state

            # Build thinking prompt
            thinking_prompt = self._build_react_prompt(state)

            # Get LLM response
            messages = [
                SystemMessage(content=await self._get_react_system_prompt()),
                HumanMessage(content=thinking_prompt)
            ]

            chat_messages = self._convert_to_chat_messages(messages)
            response = await self.llm_client.acomplete(chat_messages)

            # Parse the ReAct response
            react_step = self._parse_react_response(response.content)

            state["current_thought"] = react_step.thought
            state["current_action"] = react_step.action
            state["current_action_input"] = react_step.action_input

            if react_step.is_final:
                state["final_answer"] = react_step.thought

            state["agent_metadata"]["processing_steps"].append(f"think_iteration_{state['iteration_count']}")

        except Exception as e:
            state["error_message"] = f"Thinking step failed: {str(e)}"

        return state

    async def _act_node(self, state: ReActAgentState) -> ReActAgentState:
        """ReAct action execution step."""
        try:
            action = state["current_action"]
            action_input = state["current_action_input"]

            if not action:
                state["error_message"] = "No action specified"
                return state

            # Execute the action
            if action.lower() == "search_knowledge_base" and self.rag_engine:
                observation = await self._execute_rag_search(action_input)
            elif action.lower() == "use_tool":
                observation = await self._execute_tool(action_input)
            elif action in self.tool_registry.get_available_tools():
                observation = await self._execute_registered_tool(action, action_input)
            else:
                observation = f"Unknown action: {action}. Available actions: search_knowledge_base, use_tool, {', '.join(self.tool_registry.get_available_tools())}"

            state["current_observation"] = observation

            # Record action in history
            state["action_history"].append({
                "iteration": state["iteration_count"],
                "thought": state["current_thought"],
                "action": action,
                "action_input": action_input,
                "observation": observation
            })

            state["agent_metadata"]["processing_steps"].append(f"act_iteration_{state['iteration_count']}")

        except Exception as e:
            state["error_message"] = f"Action execution failed: {str(e)}"
            state["current_observation"] = f"Action failed: {str(e)}"

        return state

    async def _observe_node(self, state: ReActAgentState) -> ReActAgentState:
        """ReAct observation processing step."""
        try:
            # The observation is already set in the act node
            # This node can be used for additional observation processing if needed

            state["agent_metadata"]["processing_steps"].append(f"observe_iteration_{state['iteration_count']}")

        except Exception as e:
            state["error_message"] = f"Observation processing failed: {str(e)}"

        return state

    async def _finalize_node(self, state: ReActAgentState) -> ReActAgentState:
        """Finalize the ReAct reasoning and generate response."""
        try:
            if state["final_answer"]:
                final_response = state["final_answer"]
            elif state["max_iterations_reached"]:
                final_response = self._generate_max_iterations_response(state)
            else:
                final_response = self._generate_fallback_response(state)

            # Add the final response to messages
            ai_message = AIMessage(content=final_response)
            state["messages"].append(ai_message)

            state["agent_metadata"]["processing_steps"].append("finalize")
            state["agent_metadata"]["total_iterations"] = state["iteration_count"]
            state["agent_metadata"]["actions_taken"] = len(state["action_history"])

        except Exception as e:
            state["error_message"] = f"Finalization failed: {str(e)}"
            fallback_response = "I apologize, but I encountered an error while processing your request."
            state["messages"].append(AIMessage(content=fallback_response))

        return state

    def _should_act_or_finish(self, state: ReActAgentState) -> Literal["act", "finish", "error"]:
        """Determine if agent should act or finish."""
        if state.get("error_message"):
            return "error"

        if state["final_answer"] or state["max_iterations_reached"]:
            return "finish"

        if state["current_action"]:
            return "act"

        return "finish"

    def _should_continue_or_finish(self, state: ReActAgentState) -> Literal["continue", "finish", "max_iterations", "error"]:
        """Determine if agent should continue reasoning or finish."""
        if state.get("error_message"):
            return "error"

        if state["max_iterations_reached"]:
            return "max_iterations"

        if state["final_answer"]:
            return "finish"

        if state["iteration_count"] >= self.config.max_iterations:
            return "max_iterations"

        return "continue"

    def _build_react_prompt(self, state: ReActAgentState) -> str:
        """Build the ReAct reasoning prompt."""
        query = state["current_query"]
        action_history = state["action_history"]

        prompt = f"""
You are solving this problem: {query}

You have access to the following actions:
- search_knowledge_base: Search the knowledge base for information
- use_tool: Use available tools (specify tool name and input)
{self._format_available_tools()}

You should think step by step using this format:
Thought: [your reasoning about what to do next]
Action: [the action to take]
Action Input: [the input for the action]

OR if you have enough information to answer:
Thought: [your final reasoning]
Final Answer: [your final answer to the original question]

"""

        if action_history:
            prompt += "\nPrevious actions taken:\n"
            for action in action_history[-3:]:  # Show last 3 actions
                prompt += f"Thought: {action['thought']}\n"
                prompt += f"Action: {action['action']}\n"
                prompt += f"Action Input: {action['action_input']}\n"
                prompt += f"Observation: {action['observation']}\n\n"

        if state["current_observation"]:
            prompt += f"Last Observation: {state['current_observation']}\n\n"

        prompt += "What is your next thought and action?"

        return prompt

    def _format_available_tools(self) -> str:
        """Format available tools for the prompt."""
        tools = self.tool_registry.get_available_tools()
        if not tools:
            return "- No additional tools available"

        tool_descriptions = []
        for tool_name in tools:
            tool_info = self.tool_registry.get_tool_info(tool_name)
            description = tool_info.get("description", "No description available")
            tool_descriptions.append(f"- {tool_name}: {description}")

        return "\n".join(tool_descriptions)

    def _parse_react_response(self, response: str) -> ReActStep:
        """Parse the LLM response into ReAct components."""
        lines = response.strip().split('\n')
        thought = ""
        action = None
        action_input = None
        is_final = False

        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()
            if line.startswith("Thought:"):
                if current_section and current_content:
                    self._set_section_content(current_section, current_content, locals())
                current_section = "thought"
                current_content = [line[8:].strip()]
            elif line.startswith("Action:"):
                if current_section and current_content:
                    self._set_section_content(current_section, current_content, locals())
                current_section = "action"
                current_content = [line[7:].strip()]
            elif line.startswith("Action Input:"):
                if current_section and current_content:
                    self._set_section_content(current_section, current_content, locals())
                current_section = "action_input"
                current_content = [line[13:].strip()]
            elif line.startswith("Final Answer:"):
                if current_section and current_content:
                    self._set_section_content(current_section, current_content, locals())
                thought = line[13:].strip()
                is_final = True
                break
            elif line and current_section:
                current_content.append(line)

        # Handle last section
        if current_section and current_content and not is_final:
            self._set_section_content(current_section, current_content, locals())

        return ReActStep(
            thought=thought,
            action=action,
            action_input=action_input,
            is_final=is_final
        )

    def _set_section_content(self, section: str, content: List[str], variables: Dict[str, Any]) -> None:
        """Helper to set section content in variables."""
        content_str = "\n".join(content).strip()
        if section == "thought":
            variables["thought"] = content_str
        elif section == "action":
            variables["action"] = content_str
        elif section == "action_input":
            variables["action_input"] = content_str

    async def _execute_rag_search(self, query: str) -> str:
        """Execute RAG search action."""
        try:
            if not self.rag_engine:
                return "RAG search not available - no knowledge base configured"

            rag_response = await self.rag_engine.query(query)
            return f"Found information: {rag_response.answer[:500]}..."  # Truncate for brevity

        except Exception as e:
            return f"RAG search failed: {str(e)}"

    async def _execute_tool(self, tool_input: str) -> str:
        """Execute a tool action."""
        try:
            # Parse tool input (expecting format like "tool_name: input")
            if ":" in tool_input:
                tool_name, tool_input_data = tool_input.split(":", 1)
                tool_name = tool_name.strip()
                tool_input_data = tool_input_data.strip()
            else:
                return f"Invalid tool input format. Expected 'tool_name: input', got: {tool_input}"

            return await self._execute_registered_tool(tool_name, tool_input_data)

        except Exception as e:
            return f"Tool execution failed: {str(e)}"

    async def _execute_registered_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a registered tool."""
        try:
            if not self.tool_registry.has_tool(tool_name):
                available_tools = ", ".join(self.tool_registry.get_available_tools())
                return f"Tool '{tool_name}' not found. Available tools: {available_tools}"

            result = await self.tool_registry.execute_tool(tool_name, tool_input)
            return str(result)

        except Exception as e:
            return f"Tool '{tool_name}' execution failed: {str(e)}"

    def _generate_max_iterations_response(self, state: ReActAgentState) -> str:
        """Generate response when max iterations reached."""
        query = state["current_query"]
        actions_taken = len(state["action_history"])

        response = f"""I've reached the maximum number of iterations ({self.config.max_iterations}) while trying to answer your question: "{query}"

I took {actions_taken} actions during my reasoning process:
"""

        for i, action in enumerate(state["action_history"][-3:], 1):  # Show last 3 actions
            response += f"\n{i}. {action['action']}: {action['action_input'][:100]}..."

        response += "\n\nBased on the information gathered, I can provide a partial answer, but you may want to refine your question or provide more specific details for a more complete response."

        return response

    def _generate_fallback_response(self, state: ReActAgentState) -> str:
        """Generate fallback response when no final answer was reached."""
        if state["action_history"]:
            last_observation = state["action_history"][-1]["observation"]
            return f"Based on my investigation, here's what I found: {last_observation}"
        else:
            return "I wasn't able to find a specific answer to your question. Could you please provide more details or rephrase your question?"

    async def _get_react_system_prompt(self) -> str:
        """Get the system prompt for ReAct agent."""
        base_prompt = f"""You are {self.config.name}, an intelligent ReAct (Reasoning and Acting) agent.

Your goal is to answer questions by thinking step by step and taking actions to gather information.

You MUST follow this exact format:
Thought: [your reasoning about what to do next]
Action: [the action to take]
Action Input: [the input for the action]

After receiving an observation, you can either:
1. Continue with another Thought/Action/Action Input sequence
2. Provide a final answer using:
   Thought: [your final reasoning]
   Final Answer: [your answer]

Key guidelines:
- Think carefully about what information you need
- Use search_knowledge_base to find relevant information
- Use available tools when appropriate
- Provide clear, helpful final answers
- If you can't find enough information, say so clearly

"""

        if self.rag_engine:
            base_prompt += "- You have access to a knowledge base via search_knowledge_base\n"

        if self.tool_registry.get_available_tools():
            base_prompt += f"- Available tools: {', '.join(self.tool_registry.get_available_tools())}\n"

        return base_prompt

    async def process_message(
        self,
        message: str,
        thread_id: str = "default",
        config: Optional[Dict[str, Any]] = None
    ) -> LangGraphAgentResult:
        """Process a message using the ReAct workflow."""
        # Create initial state
        initial_state = ReActAgentState(
            messages=[HumanMessage(content=message)],
            current_query="",
            processed_query="",
            context_info="",
            rag_results=None,
            reasoning_steps=[],
            agent_metadata={},
            should_continue=True,
            error_message=None,
            # ReAct-specific fields
            current_thought="",
            current_action=None,
            current_action_input=None,
            current_observation=None,
            action_history=[],
            iteration_count=0,
            final_answer=None,
            max_iterations_reached=False
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
                "agent_type": "ReAct",
                "total_iterations": result.get("iteration_count", 0),
                "actions_taken": len(result.get("action_history", [])),
                "max_iterations_reached": result.get("max_iterations_reached", False),
                "processing_steps": result["agent_metadata"].get("processing_steps", []),
                "error_occurred": result.get("error_message") is not None
            }
        )