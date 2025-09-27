"""Graph-based planning agent with DFS traversal."""

import asyncio
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from ..llm.types import ChatMessage, MessageRole, StructuredThought
from .base import Agent, AgentConfig


class NodeType(str, Enum):
    """Types of nodes in the planning graph."""
    START = "start"
    GOAL = "goal"
    ACTION = "action"
    CONDITION = "condition"
    TOOL_USE = "tool_use"
    RAG_QUERY = "rag_query"
    DECISION = "decision"


class PlanningNode(BaseModel):
    """A node in the planning graph."""
    node_id: str
    node_type: NodeType
    name: str
    description: str
    conditions: Dict[str, Any] = Field(default_factory=dict)
    effects: Dict[str, Any] = Field(default_factory=dict)
    cost: float = 1.0
    prerequisites: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlanningEdge(BaseModel):
    """An edge connecting planning nodes."""
    from_node: str
    to_node: str
    condition: Optional[str] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Result of executing a planning node."""
    node_id: str
    success: bool
    output: Any = None
    state_changes: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float
    error: Optional[str] = None


class PlanningGraph:
    """Graph structure for planning with DFS capabilities."""

    def __init__(self):
        self.nodes: Dict[str, PlanningNode] = {}
        self.edges: List[PlanningEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, node: PlanningNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: PlanningEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        self.adjacency[edge.from_node].append(edge.to_node)
        self.reverse_adjacency[edge.to_node].append(edge.from_node)

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node."""
        return self.adjacency.get(node_id, [])

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get predecessors of a node."""
        return self.reverse_adjacency.get(node_id, [])

    def dfs_paths(self, start: str, goal: str) -> List[List[str]]:
        """Find all paths from start to goal using DFS."""
        paths = []
        visited = set()

        def dfs(current: str, path: List[str]) -> None:
            if current == goal:
                paths.append(path.copy())
                return

            if current in visited:
                return

            visited.add(current)
            for neighbor in self.get_neighbors(current):
                path.append(neighbor)
                dfs(neighbor, path)
                path.pop()
            visited.remove(current)

        dfs(start, [start])
        return paths

    def find_optimal_path(self, start: str, goal: str) -> Optional[List[str]]:
        """Find optimal path using A* with cost heuristic."""
        if start not in self.nodes or goal not in self.nodes:
            return None

        # Simple implementation - could be enhanced with proper A*
        paths = self.dfs_paths(start, goal)
        if not paths:
            return None

        # Find path with minimum cost
        min_cost = float('inf')
        best_path = None

        for path in paths:
            total_cost = sum(self.nodes[node_id].cost for node_id in path)
            if total_cost < min_cost:
                min_cost = total_cost
                best_path = path

        return best_path

    def get_subgraph(self, nodes: Set[str]) -> "PlanningGraph":
        """Extract a subgraph containing only specified nodes."""
        subgraph = PlanningGraph()

        # Add nodes
        for node_id in nodes:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])

        # Add edges between included nodes
        for edge in self.edges:
            if edge.from_node in nodes and edge.to_node in nodes:
                subgraph.add_edge(edge)

        return subgraph


class GraphPlannerAgent(Agent):
    """Agent that uses graph-based planning with DFS traversal."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client,
        rag_engine=None,
        planning_graph: Optional[PlanningGraph] = None,
    ):
        super().__init__(config, llm_client, rag_engine)
        self.planning_graph = planning_graph or self._build_default_graph()
        self.execution_state: Dict[str, Any] = {}
        self.execution_history: List[ExecutionResult] = []

    def _build_default_graph(self) -> PlanningGraph:
        """Build a default planning graph for general use."""
        graph = PlanningGraph()

        # Define nodes
        nodes = [
            PlanningNode(
                node_id="start",
                node_type=NodeType.START,
                name="Start",
                description="Initial state",
            ),
            PlanningNode(
                node_id="analyze_query",
                node_type=NodeType.DECISION,
                name="Analyze Query",
                description="Understand user intent and requirements",
                cost=1.0,
            ),
            PlanningNode(
                node_id="check_knowledge",
                node_type=NodeType.CONDITION,
                name="Check Knowledge",
                description="Determine if external knowledge is needed",
                cost=0.5,
            ),
            PlanningNode(
                node_id="query_rag",
                node_type=NodeType.RAG_QUERY,
                name="Query RAG",
                description="Retrieve relevant information",
                cost=2.0,
            ),
            PlanningNode(
                node_id="use_tool",
                node_type=NodeType.TOOL_USE,
                name="Use Tool",
                description="Execute external tool",
                cost=3.0,
            ),
            PlanningNode(
                node_id="synthesize_response",
                node_type=NodeType.ACTION,
                name="Synthesize Response",
                description="Generate final response",
                cost=1.5,
            ),
            PlanningNode(
                node_id="goal",
                node_type=NodeType.GOAL,
                name="Goal",
                description="Task completed",
            ),
        ]

        for node in nodes:
            graph.add_node(node)

        # Define edges
        edges = [
            PlanningEdge("start", "analyze_query"),
            PlanningEdge("analyze_query", "check_knowledge"),
            PlanningEdge("check_knowledge", "query_rag", condition="needs_external_info"),
            PlanningEdge("check_knowledge", "use_tool", condition="needs_tool"),
            PlanningEdge("check_knowledge", "synthesize_response", condition="has_sufficient_info"),
            PlanningEdge("query_rag", "synthesize_response"),
            PlanningEdge("use_tool", "synthesize_response"),
            PlanningEdge("synthesize_response", "goal"),
        ]

        for edge in edges:
            graph.add_edge(edge)

        return graph

    async def process_message(self, message: str) -> str:
        """Process message using graph-based planning."""
        await self._add_to_memory(ChatMessage(role=MessageRole.USER, content=message))

        # Reset execution state
        self.execution_state = {"query": message, "context": {}}
        self.execution_history = []

        # Plan execution path
        execution_plan = await self._plan_execution(message)
        if not execution_plan:
            return "I couldn't create a plan to handle your request."

        # Execute the plan
        result = await self._execute_plan(execution_plan, message)

        # Add to memory
        await self._add_to_memory(
            ChatMessage(role=MessageRole.ASSISTANT, content=result)
        )

        self.state.turn_count += 1
        return result

    async def _plan_execution(self, message: str) -> Optional[List[str]]:
        """Create execution plan using graph search."""
        if self.config.enable_reasoning:
            # Use LLM to analyze requirements and select relevant subgraph
            planning_prompt = f"""
            Analyze this user request: "{message}"

            Available planning nodes: {list(self.planning_graph.nodes.keys())}

            Determine:
            1. What type of response is needed?
            2. What information or tools are required?
            3. What's the optimal execution path?

            Consider these factors:
            - Does it need external information (RAG)?
            - Does it require tool usage?
            - Can it be answered directly?
            """

            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=planning_prompt),
                ChatMessage(role=MessageRole.USER, content="Plan the optimal execution path."),
            ]

            thought = await self.llm_client.acomplete(
                messages, response_model=StructuredThought
            )

            # Update execution state with reasoning
            self.execution_state["planning_analysis"] = thought.query_analysis
            self.execution_state["reasoning_steps"] = thought.reasoning_steps

        # Find optimal path from start to goal
        optimal_path = self.planning_graph.find_optimal_path("start", "goal")
        return optimal_path

    async def _execute_plan(self, plan: List[str], message: str) -> str:
        """Execute the planned sequence of nodes."""
        final_result = ""

        for node_id in plan:
            if node_id not in self.planning_graph.nodes:
                continue

            node = self.planning_graph.nodes[node_id]
            result = await self._execute_node(node, message)
            self.execution_history.append(result)

            if not result.success and node.node_type != NodeType.START:
                # Handle failure - could implement retry logic or alternative paths
                break

            # Update execution state
            self.execution_state.update(result.state_changes)

            # Store final output
            if node.node_type == NodeType.ACTION and result.output:
                final_result = result.output

        return final_result or "Task completed but no output generated."

    async def _execute_node(self, node: PlanningNode, message: str) -> ExecutionResult:
        """Execute a single planning node."""
        import time
        start_time = time.time()

        try:
            if node.node_type == NodeType.START:
                return ExecutionResult(
                    node_id=node.node_id,
                    success=True,
                    execution_time=time.time() - start_time,
                    state_changes={"initialized": True},
                )

            elif node.node_type == NodeType.DECISION:
                # Analyze and make decisions
                analysis = await self._analyze_query(message)
                return ExecutionResult(
                    node_id=node.node_id,
                    success=True,
                    output=analysis,
                    execution_time=time.time() - start_time,
                    state_changes={"query_analysis": analysis},
                )

            elif node.node_type == NodeType.CONDITION:
                # Evaluate conditions
                needs_rag = await self._evaluate_rag_need(message)
                needs_tool = await self._evaluate_tool_need(message)
                return ExecutionResult(
                    node_id=node.node_id,
                    success=True,
                    execution_time=time.time() - start_time,
                    state_changes={
                        "needs_rag": needs_rag,
                        "needs_tool": needs_tool,
                        "has_sufficient_info": not (needs_rag or needs_tool),
                    },
                )

            elif node.node_type == NodeType.RAG_QUERY:
                # Query RAG system
                if self.rag_engine:
                    rag_response = await self.rag_engine.query(message)
                    return ExecutionResult(
                        node_id=node.node_id,
                        success=True,
                        output=rag_response.answer,
                        execution_time=time.time() - start_time,
                        state_changes={"rag_context": rag_response},
                    )

            elif node.node_type == NodeType.ACTION:
                # Generate final response
                response = await self._synthesize_response(message)
                return ExecutionResult(
                    node_id=node.node_id,
                    success=True,
                    output=response,
                    execution_time=time.time() - start_time,
                )

            else:
                return ExecutionResult(
                    node_id=node.node_id,
                    success=True,
                    execution_time=time.time() - start_time,
                )

        except Exception as e:
            return ExecutionResult(
                node_id=node.node_id,
                success=False,
                execution_time=time.time() - start_time,
                error=str(e),
            )

    async def _analyze_query(self, message: str) -> str:
        """Analyze the user query."""
        system_prompt = "Analyze the user's query and determine the type of response needed."
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=message),
        ]
        response = await self.llm_client.acomplete(messages)
        return response.content

    async def _evaluate_rag_need(self, message: str) -> bool:
        """Evaluate if RAG is needed."""
        question_indicators = ["what", "how", "why", "when", "where", "explain", "describe"]
        return any(indicator in message.lower() for indicator in question_indicators)

    async def _evaluate_tool_need(self, message: str) -> bool:
        """Evaluate if tools are needed."""
        tool_indicators = ["calculate", "compute", "search", "find", "lookup"]
        return any(indicator in message.lower() for indicator in tool_indicators)

    async def _synthesize_response(self, message: str) -> str:
        """Synthesize final response."""
        context_parts = []
        if "rag_context" in self.execution_state:
            rag_response = self.execution_state["rag_context"]
            context_parts.append(f"Retrieved information: {rag_response.answer}")

        context_str = "\n".join(context_parts) if context_parts else ""

        system_prompt = await self._get_system_prompt()
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=message),
        ]

        if context_str:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=context_str))

        response = await self.llm_client.acomplete(messages)
        return response.content

    def add_planning_node(self, node: PlanningNode) -> None:
        """Add a new node to the planning graph."""
        self.planning_graph.add_node(node)

    def add_planning_edge(self, edge: PlanningEdge) -> None:
        """Add a new edge to the planning graph."""
        self.planning_graph.add_edge(edge)

    def get_execution_trace(self) -> List[ExecutionResult]:
        """Get the execution history."""
        return self.execution_history.copy()

    def visualize_plan(self, plan: List[str]) -> Dict[str, Any]:
        """Create a visualization-friendly representation of the plan."""
        return {
            "nodes": [
                {
                    "id": node_id,
                    "name": self.planning_graph.nodes[node_id].name,
                    "type": self.planning_graph.nodes[node_id].node_type,
                    "cost": self.planning_graph.nodes[node_id].cost,
                }
                for node_id in plan
                if node_id in self.planning_graph.nodes
            ],
            "edges": [
                {"from": plan[i], "to": plan[i + 1]}
                for i in range(len(plan) - 1)
            ],
            "total_cost": sum(
                self.planning_graph.nodes[node_id].cost
                for node_id in plan
                if node_id in self.planning_graph.nodes
            ),
        }