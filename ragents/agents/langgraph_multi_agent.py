"""LangGraph-based multi-agent system implementation."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal, Union
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .langgraph_base import LangGraphAgent, LangGraphAgentState, LangGraphAgentResult
from .langgraph_react import LangGraphReActAgent
from .base import AgentConfig
from ..llm.client import LLMClient
from ..rag.engine import RAGEngine
from ..query_rewriting.base import QueryRewriter
from ..tools.base import ToolRegistry


class AgentRole(Enum):
    """Predefined agent roles in the multi-agent system."""

    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"
    SPECIALIST = "specialist"


class MultiAgentState(LangGraphAgentState):
    """Extended state for multi-agent orchestration."""

    current_agent: Optional[str]
    agent_assignments: Dict[str, str]
    agent_outputs: Dict[str, Any]
    coordination_history: List[Dict[str, Any]]
    task_decomposition: List[str]
    collaboration_round: int
    final_synthesis: Optional[str]
    routing_decision: Optional[str]


@dataclass
class AgentDefinition:
    """Definition of an agent in the multi-agent system."""

    name: str
    role: AgentRole
    config: AgentConfig
    llm_client: LLMClient
    rag_engine: Optional[RAGEngine] = None
    query_rewriter: Optional[QueryRewriter] = None
    tool_registry: Optional[ToolRegistry] = None
    specialization: Optional[str] = None


class LangGraphMultiAgent:
    """LangGraph-based multi-agent orchestration system."""

    def __init__(
        self,
        coordinator_config: AgentConfig,
        llm_client: LLMClient,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self.coordinator_config = coordinator_config
        self.llm_client = llm_client
        self.checkpointer = checkpointer or MemorySaver()

        # Registry of agents
        self.agents: Dict[str, Union[LangGraphAgent, LangGraphReActAgent]] = {}
        self.agent_definitions: Dict[str, AgentDefinition] = {}

        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def register_agent(self, definition: AgentDefinition) -> None:
        """Register an agent in the multi-agent system."""
        self.agent_definitions[definition.name] = definition

        # Create the actual agent instance
        if definition.role == AgentRole.RESEARCHER:
            agent = LangGraphReActAgent(
                definition.config,
                definition.llm_client,
                definition.rag_engine,
                definition.query_rewriter,
                definition.tool_registry,
                self.checkpointer
            )
        else:
            agent = LangGraphAgent(
                definition.config,
                definition.llm_client,
                definition.rag_engine,
                definition.query_rewriter,
                self.checkpointer
            )

        self.agents[definition.name] = agent

    def _build_workflow(self) -> StateGraph:
        """Build the multi-agent workflow graph."""
        workflow = StateGraph(MultiAgentState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("route_task", self._route_task_node)
        workflow.add_node("coordinate", self._coordinate_node)
        workflow.add_node("execute_agent", self._execute_agent_node)
        workflow.add_node("review_outputs", self._review_outputs_node)
        workflow.add_node("synthesize", self._synthesize_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Define the flow
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "route_task")

        workflow.add_conditional_edges(
            "route_task",
            self._routing_decision,
            {
                "single_agent": "execute_agent",
                "multi_agent": "coordinate",
                "error": "error_handler"
            }
        )

        workflow.add_edge("coordinate", "execute_agent")

        workflow.add_conditional_edges(
            "execute_agent",
            self._execution_decision,
            {
                "continue": "coordinate",
                "review": "review_outputs",
                "error": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "review_outputs",
            self._review_decision,
            {
                "revise": "coordinate",
                "synthesize": "synthesize",
                "error": "error_handler"
            }
        )

        workflow.add_edge("synthesize", END)
        workflow.add_edge("error_handler", END)

        return workflow

    async def _initialize_node(self, state: MultiAgentState) -> MultiAgentState:
        """Initialize the multi-agent system state."""
        try:
            # Get the latest human message
            human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            if human_messages:
                current_query = human_messages[-1].content
                state["current_query"] = current_query
                state["processed_query"] = current_query

            # Initialize multi-agent specific state
            state["current_agent"] = None
            state["agent_assignments"] = {}
            state["agent_outputs"] = {}
            state["coordination_history"] = []
            state["task_decomposition"] = []
            state["collaboration_round"] = 0
            state["final_synthesis"] = None
            state["routing_decision"] = None

            # Initialize metadata
            state["agent_metadata"] = {
                "system_type": "multi_agent",
                "available_agents": list(self.agents.keys()),
                "processing_steps": []
            }

            state["should_continue"] = True
            state["error_message"] = None

        except Exception as e:
            state["error_message"] = str(e)
            state["should_continue"] = False

        return state

    async def _route_task_node(self, state: MultiAgentState) -> MultiAgentState:
        """Route the task to appropriate agent(s)."""
        try:
            query = state["current_query"]

            # Analyze task complexity and determine routing
            routing_prompt = f"""
            Analyze this task and determine the best approach:

            Task: {query}

            Available agents: {list(self.agents.keys())}

            Should this task be handled by:
            1. A single specialized agent
            2. Multiple agents working together

            Consider:
            - Task complexity
            - Required expertise
            - Need for multiple perspectives
            - Research vs analysis vs writing requirements

            Respond with either "single_agent" or "multi_agent" and explain your reasoning.
            """

            messages = [
                SystemMessage(content="You are a task routing coordinator. Analyze tasks and determine the best agent assignment strategy."),
                HumanMessage(content=routing_prompt)
            ]

            chat_messages = self._convert_to_chat_messages(messages)
            response = await self.llm_client.acomplete(chat_messages)

            # Parse routing decision
            response_text = response.content.lower()
            if "single_agent" in response_text:
                state["routing_decision"] = "single_agent"
                # Assign to most appropriate single agent
                state["current_agent"] = await self._select_best_agent(query)
            elif "multi_agent" in response_text:
                state["routing_decision"] = "multi_agent"
                # Decompose task for multiple agents
                state["task_decomposition"] = await self._decompose_task(query)
            else:
                state["routing_decision"] = "single_agent"  # Default fallback
                state["current_agent"] = list(self.agents.keys())[0] if self.agents else None

            state["agent_metadata"]["processing_steps"].append("route_task")

        except Exception as e:
            state["error_message"] = f"Task routing failed: {str(e)}"

        return state

    async def _coordinate_node(self, state: MultiAgentState) -> MultiAgentState:
        """Coordinate multiple agents."""
        try:
            state["collaboration_round"] += 1

            if state["routing_decision"] == "multi_agent":
                # Assign subtasks to agents
                assignments = await self._assign_subtasks(
                    state["task_decomposition"],
                    state["current_query"]
                )
                state["agent_assignments"] = assignments

                # Select next agent to execute
                for agent_name, subtask in assignments.items():
                    if agent_name not in state["agent_outputs"]:
                        state["current_agent"] = agent_name
                        break

            # Record coordination step
            state["coordination_history"].append({
                "round": state["collaboration_round"],
                "assignments": state["agent_assignments"].copy(),
                "current_agent": state["current_agent"]
            })

            state["agent_metadata"]["processing_steps"].append(f"coordinate_round_{state['collaboration_round']}")

        except Exception as e:
            state["error_message"] = f"Coordination failed: {str(e)}"

        return state

    async def _execute_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute the current agent."""
        try:
            current_agent = state["current_agent"]
            if not current_agent or current_agent not in self.agents:
                state["error_message"] = f"Invalid agent: {current_agent}"
                return state

            agent = self.agents[current_agent]

            # Determine what task the agent should work on
            if state["routing_decision"] == "single_agent":
                task = state["current_query"]
            else:
                task = state["agent_assignments"].get(current_agent, state["current_query"])

            # Execute the agent
            result = await agent.process_message(task)

            # Store the agent's output
            state["agent_outputs"][current_agent] = {
                "response": result.response,
                "metadata": result.execution_metadata,
                "subtask": task
            }

            state["agent_metadata"]["processing_steps"].append(f"execute_{current_agent}")

        except Exception as e:
            state["error_message"] = f"Agent execution failed: {str(e)}"

        return state

    async def _review_outputs_node(self, state: MultiAgentState) -> MultiAgentState:
        """Review agent outputs for quality and completeness."""
        try:
            # Check if we have outputs from all assigned agents
            assigned_agents = set(state["agent_assignments"].keys())
            completed_agents = set(state["agent_outputs"].keys())

            if not assigned_agents.issubset(completed_agents):
                # Not all agents have completed - this should trigger continuation
                state["agent_metadata"]["review_status"] = "incomplete"
                return state

            # Review the quality of outputs
            review_prompt = self._build_review_prompt(state)

            messages = [
                SystemMessage(content="You are a quality reviewer. Assess whether agent outputs adequately address the original task."),
                HumanMessage(content=review_prompt)
            ]

            chat_messages = self._convert_to_chat_messages(messages)
            response = await self.llm_client.acomplete(chat_messages)

            # Parse review decision
            response_text = response.content.lower()
            if "revise" in response_text or "insufficient" in response_text:
                state["agent_metadata"]["review_status"] = "needs_revision"
            else:
                state["agent_metadata"]["review_status"] = "approved"

            state["agent_metadata"]["processing_steps"].append("review_outputs")

        except Exception as e:
            state["error_message"] = f"Output review failed: {str(e)}"

        return state

    async def _synthesize_node(self, state: MultiAgentState) -> MultiAgentState:
        """Synthesize outputs from multiple agents into final response."""
        try:
            if state["routing_decision"] == "single_agent":
                # Single agent - just use their output
                agent_name = state["current_agent"]
                if agent_name in state["agent_outputs"]:
                    final_response = state["agent_outputs"][agent_name]["response"]
                else:
                    final_response = "No output generated from agent."
            else:
                # Multi-agent - synthesize outputs
                synthesis_prompt = self._build_synthesis_prompt(state)

                messages = [
                    SystemMessage(content="You are a synthesis expert. Combine multiple agent outputs into a coherent, comprehensive response."),
                    HumanMessage(content=synthesis_prompt)
                ]

                chat_messages = self._convert_to_chat_messages(messages)
                response = await self.llm_client.acomplete(chat_messages)
                final_response = response.content

            state["final_synthesis"] = final_response

            # Add final response to messages
            ai_message = AIMessage(content=final_response)
            state["messages"].append(ai_message)

            state["agent_metadata"]["processing_steps"].append("synthesize")
            state["agent_metadata"]["final_agent_count"] = len(state["agent_outputs"])

        except Exception as e:
            state["error_message"] = f"Synthesis failed: {str(e)}"
            fallback_response = "I apologize, but I encountered an error while synthesizing the response."
            state["messages"].append(AIMessage(content=fallback_response))

        return state

    async def _error_handler_node(self, state: MultiAgentState) -> MultiAgentState:
        """Handle errors in the multi-agent workflow."""
        error_response = f"Multi-agent system error: {state.get('error_message', 'Unknown error')}"
        state["messages"].append(AIMessage(content=error_response))
        state["agent_metadata"]["error_handled"] = True
        return state

    def _routing_decision(self, state: MultiAgentState) -> Literal["single_agent", "multi_agent", "error"]:
        """Determine routing based on analysis."""
        if state.get("error_message"):
            return "error"

        routing = state.get("routing_decision", "single_agent")
        return routing if routing in ["single_agent", "multi_agent"] else "single_agent"

    def _execution_decision(self, state: MultiAgentState) -> Literal["continue", "review", "error"]:
        """Determine if execution should continue."""
        if state.get("error_message"):
            return "error"

        if state["routing_decision"] == "single_agent":
            return "review"

        # Multi-agent case
        assigned_agents = set(state["agent_assignments"].keys())
        completed_agents = set(state["agent_outputs"].keys())

        if assigned_agents.issubset(completed_agents):
            return "review"
        else:
            return "continue"

    def _review_decision(self, state: MultiAgentState) -> Literal["revise", "synthesize", "error"]:
        """Determine if outputs need revision or can be synthesized."""
        if state.get("error_message"):
            return "error"

        review_status = state["agent_metadata"].get("review_status", "approved")

        if review_status == "needs_revision":
            return "revise"
        else:
            return "synthesize"

    async def _select_best_agent(self, query: str) -> str:
        """Select the best single agent for a query."""
        if not self.agents:
            return None

        # Simple selection based on agent roles and query content
        query_lower = query.lower()

        # Check for research-related keywords
        if any(word in query_lower for word in ["research", "find", "search", "investigate"]):
            for name, definition in self.agent_definitions.items():
                if definition.role == AgentRole.RESEARCHER:
                    return name

        # Check for analysis-related keywords
        if any(word in query_lower for word in ["analyze", "compare", "evaluate", "assess"]):
            for name, definition in self.agent_definitions.items():
                if definition.role == AgentRole.ANALYST:
                    return name

        # Check for writing-related keywords
        if any(word in query_lower for word in ["write", "create", "draft", "compose"]):
            for name, definition in self.agent_definitions.items():
                if definition.role == AgentRole.WRITER:
                    return name

        # Default to first available agent
        return list(self.agents.keys())[0]

    async def _decompose_task(self, query: str) -> List[str]:
        """Decompose a complex task into subtasks."""
        decomposition_prompt = f"""
        Break down this complex task into smaller, manageable subtasks:

        Task: {query}

        Available agent roles: {[role.value for role in AgentRole]}

        Provide 2-4 specific subtasks that would collectively address the main task.
        Each subtask should be actionable and suitable for a specialized agent.

        Format your response as a numbered list of subtasks.
        """

        messages = [
            SystemMessage(content="You are a task decomposition expert. Break complex tasks into manageable subtasks."),
            HumanMessage(content=decomposition_prompt)
        ]

        chat_messages = self._convert_to_chat_messages(messages)
        response = await self.llm_client.acomplete(chat_messages)

        # Parse subtasks from response
        subtasks = []
        lines = response.content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and clean up
                cleaned = line.lstrip('0123456789.-• ').strip()
                if cleaned:
                    subtasks.append(cleaned)

        return subtasks if subtasks else [query]  # Fallback to original query

    async def _assign_subtasks(self, subtasks: List[str], original_query: str) -> Dict[str, str]:
        """Assign subtasks to appropriate agents."""
        assignments = {}

        for subtask in subtasks:
            # Find best agent for this subtask
            best_agent = await self._select_best_agent(subtask)
            if best_agent:
                assignments[best_agent] = subtask

        # Ensure we have at least one assignment
        if not assignments and self.agents:
            first_agent = list(self.agents.keys())[0]
            assignments[first_agent] = original_query

        return assignments

    def _build_review_prompt(self, state: MultiAgentState) -> str:
        """Build prompt for reviewing agent outputs."""
        original_query = state["current_query"]
        outputs = state["agent_outputs"]

        prompt = f"""
        Original task: {original_query}

        Agent outputs to review:
        """

        for agent_name, output_data in outputs.items():
            prompt += f"\n{agent_name}:\n{output_data['response'][:300]}...\n"

        prompt += """

        Review criteria:
        1. Do the outputs collectively address the original task?
        2. Is the information accurate and relevant?
        3. Are there any gaps or inconsistencies?
        4. Is the quality sufficient for synthesis?

        Respond with either "approved" if ready for synthesis or "revise" if improvements needed.
        """

        return prompt

    def _build_synthesis_prompt(self, state: MultiAgentState) -> str:
        """Build prompt for synthesizing multiple agent outputs."""
        original_query = state["current_query"]
        outputs = state["agent_outputs"]

        prompt = f"""
        Original question: {original_query}

        Agent contributions:
        """

        for agent_name, output_data in outputs.items():
            subtask = output_data.get('subtask', 'General task')
            response = output_data['response']
            prompt += f"\n{agent_name} (worked on: {subtask}):\n{response}\n"

        prompt += f"""

        Please synthesize these contributions into a comprehensive, coherent response that fully addresses the original question: "{original_query}"

        Guidelines:
        - Integrate all relevant information
        - Resolve any contradictions
        - Ensure logical flow
        - Provide a complete answer
        """

        return prompt

    def _convert_to_chat_messages(self, messages: List[BaseMessage]) -> List:
        """Convert LangChain messages to chat messages."""
        # Simplified conversion for this example
        chat_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                chat_messages.append({"role": "user" if isinstance(msg, HumanMessage) else "system", "content": msg.content})
        return chat_messages

    async def process_message(
        self,
        message: str,
        thread_id: str = "default",
        config: Optional[Dict[str, Any]] = None
    ) -> LangGraphAgentResult:
        """Process a message using the multi-agent workflow."""
        # Create initial state
        initial_state = MultiAgentState(
            messages=[HumanMessage(content=message)],
            current_query="",
            processed_query="",
            context_info="",
            rag_results=None,
            reasoning_steps=[],
            agent_metadata={},
            should_continue=True,
            error_message=None,
            # Multi-agent specific fields
            current_agent=None,
            agent_assignments={},
            agent_outputs={},
            coordination_history=[],
            task_decomposition=[],
            collaboration_round=0,
            final_synthesis=None,
            routing_decision=None
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
                "system_type": "multi_agent",
                "routing_decision": result.get("routing_decision"),
                "agents_used": list(result.get("agent_outputs", {}).keys()),
                "collaboration_rounds": result.get("collaboration_round", 0),
                "processing_steps": result["agent_metadata"].get("processing_steps", []),
                "error_occurred": result.get("error_message") is not None
            }
        )


# Utility functions for creating common agent setups

def create_research_team(
    llm_client: LLMClient,
    rag_engine: Optional[RAGEngine] = None,
    tool_registry: Optional[ToolRegistry] = None
) -> LangGraphMultiAgent:
    """Create a research-focused multi-agent team."""
    coordinator_config = AgentConfig(
        name="ResearchCoordinator",
        description="Coordinates research tasks across specialized agents"
    )

    multi_agent = LangGraphMultiAgent(coordinator_config, llm_client)

    # Research agent
    researcher_config = AgentConfig(
        name="Researcher",
        description="Finds and gathers information from various sources",
        enable_rag=True,
        enable_tools=True
    )
    multi_agent.register_agent(AgentDefinition(
        name="researcher",
        role=AgentRole.RESEARCHER,
        config=researcher_config,
        llm_client=llm_client,
        rag_engine=rag_engine,
        tool_registry=tool_registry
    ))

    # Analyst agent
    analyst_config = AgentConfig(
        name="Analyst",
        description="Analyzes and synthesizes research findings",
        enable_reasoning=True
    )
    multi_agent.register_agent(AgentDefinition(
        name="analyst",
        role=AgentRole.ANALYST,
        config=analyst_config,
        llm_client=llm_client
    ))

    # Writer agent
    writer_config = AgentConfig(
        name="Writer",
        description="Creates well-structured written outputs"
    )
    multi_agent.register_agent(AgentDefinition(
        name="writer",
        role=AgentRole.WRITER,
        config=writer_config,
        llm_client=llm_client
    ))

    return multi_agent


def create_analysis_team(
    llm_client: LLMClient,
    rag_engine: Optional[RAGEngine] = None
) -> LangGraphMultiAgent:
    """Create an analysis-focused multi-agent team."""
    coordinator_config = AgentConfig(
        name="AnalysisCoordinator",
        description="Coordinates analytical tasks"
    )

    multi_agent = LangGraphMultiAgent(coordinator_config, llm_client)

    # Data analyst
    analyst_config = AgentConfig(
        name="DataAnalyst",
        description="Performs quantitative and qualitative analysis",
        enable_reasoning=True,
        reasoning_depth=5
    )
    multi_agent.register_agent(AgentDefinition(
        name="data_analyst",
        role=AgentRole.ANALYST,
        config=analyst_config,
        llm_client=llm_client,
        rag_engine=rag_engine
    ))

    # Critic agent
    critic_config = AgentConfig(
        name="Critic",
        description="Reviews and critiques analysis for quality and accuracy"
    )
    multi_agent.register_agent(AgentDefinition(
        name="critic",
        role=AgentRole.CRITIC,
        config=critic_config,
        llm_client=llm_client
    ))

    return multi_agent