"""Decision tree agent implementation."""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ..llm.types import ChatMessage, MessageRole, StructuredThought
from .base import Agent, AgentConfig
from ..query_rewriting.base import QueryRewriter


class ActionType(str, Enum):
    """Types of actions an agent can take."""
    RESPOND = "respond"
    QUERY_RAG = "query_rag"
    USE_TOOL = "use_tool"
    THINK = "think"
    BRANCH = "branch"
    END = "end"


class DecisionCriteria(BaseModel):
    """Criteria for making decisions in the tree."""
    condition: str
    threshold: Optional[float] = None
    keywords: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)


class DecisionResult(BaseModel):
    """Result of a decision evaluation."""
    action: ActionType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    next_node: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class DecisionNode(BaseModel):
    """A node in the decision tree."""
    node_id: str
    name: str
    description: str
    action_type: ActionType
    criteria: DecisionCriteria
    success_node: Optional[str] = None
    failure_node: Optional[str] = None
    alternative_nodes: Dict[str, str] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    is_terminal: bool = False


class DecisionTreeAgent(Agent):
    """Agent that uses decision trees for action selection."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client,
        rag_engine=None,
        decision_tree: Optional[Dict[str, DecisionNode]] = None,
        root_node: str = "root",
    ):
        super().__init__(config, llm_client, rag_engine)
        self.decision_tree = decision_tree or self._build_default_tree()
        self.root_node = root_node
        self.current_node = root_node

    def _build_default_tree(self) -> Dict[str, DecisionNode]:
        """Build a default decision tree for general purpose use."""
        return {
            "root": DecisionNode(
                node_id="root",
                name="Root Decision",
                description="Initial decision point for message processing",
                action_type=ActionType.THINK,
                criteria=DecisionCriteria(
                    condition="analyze_message_type",
                    keywords=["question", "request", "task"],
                ),
                success_node="analyze_query",
                failure_node="respond_direct",
            ),
            "analyze_query": DecisionNode(
                node_id="analyze_query",
                name="Query Analysis",
                description="Analyze if the query needs information retrieval",
                action_type=ActionType.THINK,
                criteria=DecisionCriteria(
                    condition="needs_external_info",
                    keywords=["what", "how", "why", "when", "where", "explain"],
                    threshold=0.7,
                ),
                success_node="query_rag",
                failure_node="respond_direct",
            ),
            "query_rag": DecisionNode(
                node_id="query_rag",
                name="Query Knowledge Base",
                description="Query the RAG system for relevant information",
                action_type=ActionType.QUERY_RAG,
                criteria=DecisionCriteria(condition="rag_available"),
                success_node="respond_with_context",
                failure_node="respond_direct",
            ),
            "respond_with_context": DecisionNode(
                node_id="respond_with_context",
                name="Respond with Context",
                description="Generate response using retrieved context",
                action_type=ActionType.RESPOND,
                criteria=DecisionCriteria(condition="always_true"),
                is_terminal=True,
            ),
            "respond_direct": DecisionNode(
                node_id="respond_direct",
                name="Direct Response",
                description="Generate response without external context",
                action_type=ActionType.RESPOND,
                criteria=DecisionCriteria(condition="always_true"),
                is_terminal=True,
            ),
        }

    async def process_message(self, message: str) -> str:
        """Process message through the decision tree."""
        await self._add_to_memory(ChatMessage(role=MessageRole.USER, content=message))

        # Reset to root node for new message
        self.current_node = self.root_node
        context = {"original_message": message, "rag_context": None}

        # Traverse the decision tree
        max_iterations = self.config.max_iterations
        for iteration in range(max_iterations):
            if self.current_node not in self.decision_tree:
                break

            node = self.decision_tree[self.current_node]
            decision = await self._evaluate_node(node, message, context)

            # Execute the action
            result = await self._execute_action(decision, message, context)

            if node.is_terminal or decision.action == ActionType.END:
                break

            # Move to next node
            if decision.next_node:
                self.current_node = decision.next_node
            elif decision.confidence > 0.7 and node.success_node:
                self.current_node = node.success_node
            elif node.failure_node:
                self.current_node = node.failure_node
            else:
                break

        # Return the final response
        final_response = context.get("response", "I couldn't process your request.")
        await self._add_to_memory(
            ChatMessage(role=MessageRole.ASSISTANT, content=final_response)
        )

        self.state.turn_count += 1
        return final_response

    async def _evaluate_node(
        self, node: DecisionNode, message: str, context: Dict[str, Any]
    ) -> DecisionResult:
        """Evaluate a decision node to determine the next action."""
        if self.config.enable_reasoning:
            # Use LLM for sophisticated decision making
            system_prompt = f"""
You are evaluating a decision node in an agent's decision tree.

Node: {node.name}
Description: {node.description}
Action Type: {node.action_type}
Criteria: {node.criteria.condition}

Analyze the message and context to determine:
1. Should this action be taken?
2. What's the confidence level (0.0-1.0)?
3. What's the reasoning?
4. Any parameters needed for the action?

Message: {message}
Context: {context}
"""

            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(
                    role=MessageRole.USER,
                    content="Evaluate this decision node and provide your analysis.",
                ),
            ]

            structured_thought = await self.llm_client.acomplete(
                messages, response_model=StructuredThought
            )

            return DecisionResult(
                action=node.action_type,
                confidence=structured_thought.confidence_score,
                reasoning=structured_thought.query_analysis,
                parameters={"thought_process": structured_thought.reasoning_steps},
            )
        else:
            # Simple rule-based evaluation
            confidence = self._evaluate_simple_criteria(node.criteria, message)
            return DecisionResult(
                action=node.action_type,
                confidence=confidence,
                reasoning=f"Rule-based evaluation for {node.criteria.condition}",
            )

    def _evaluate_simple_criteria(
        self, criteria: DecisionCriteria, message: str
    ) -> float:
        """Simple rule-based criteria evaluation."""
        message_lower = message.lower()

        if criteria.condition == "always_true":
            return 1.0
        elif criteria.condition == "needs_external_info":
            question_indicators = ["what", "how", "why", "when", "where", "explain"]
            return (
                0.8
                if any(word in message_lower for word in question_indicators)
                else 0.3
            )
        elif criteria.condition == "rag_available":
            return 1.0 if self.rag_engine else 0.0
        elif criteria.condition == "analyze_message_type":
            task_indicators = ["help", "can you", "please", "?"]
            return (
                0.9 if any(indicator in message_lower for indicator in task_indicators) else 0.5
            )
        else:
            # Check keywords
            keyword_matches = sum(
                1 for keyword in criteria.keywords if keyword in message_lower
            )
            return min(keyword_matches / max(len(criteria.keywords), 1), 1.0)

    async def _execute_action(
        self, decision: DecisionResult, message: str, context: Dict[str, Any]
    ) -> Any:
        """Execute the decided action."""
        if decision.action == ActionType.QUERY_RAG:
            if self.rag_engine:
                rag_response = await self.rag_engine.query(message)
                context["rag_context"] = rag_response
                context["rag_answer"] = rag_response.answer
                return rag_response
        elif decision.action == ActionType.RESPOND:
            response = await self._generate_response(message, context)
            context["response"] = response
            return response
        elif decision.action == ActionType.THINK:
            # Log thinking process
            self.state.context["last_thought"] = decision.reasoning
            return decision.reasoning
        elif decision.action == ActionType.USE_TOOL:
            # Tool usage would be implemented here
            pass

        return None

    async def _generate_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate the final response."""
        system_prompt = await self._get_system_prompt()

        # Build context string
        context_parts = []
        if context.get("rag_answer"):
            context_parts.append(f"Relevant Information: {context['rag_answer']}")

        context_str = "\n".join(context_parts) if context_parts else ""

        messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)]

        # Add conversation history
        messages.extend(self.state.memory[-self.config.memory_window :])

        if context_str:
            messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=context_str)
            )

        response = await self.llm_client.acomplete(messages)
        return response.content

    def add_node(self, node: DecisionNode) -> None:
        """Add a new node to the decision tree."""
        self.decision_tree[node.node_id] = node

    def update_node(self, node_id: str, **kwargs) -> None:
        """Update an existing node."""
        if node_id in self.decision_tree:
            node = self.decision_tree[node_id]
            for key, value in kwargs.items():
                if hasattr(node, key):
                    setattr(node, key, value)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the decision tree."""
        if node_id in self.decision_tree:
            del self.decision_tree[node_id]

    def get_tree_structure(self) -> Dict[str, Any]:
        """Get a representation of the tree structure."""
        return {
            node_id: {
                "name": node.name,
                "action_type": node.action_type,
                "success_node": node.success_node,
                "failure_node": node.failure_node,
                "is_terminal": node.is_terminal,
            }
            for node_id, node in self.decision_tree.items()
        }