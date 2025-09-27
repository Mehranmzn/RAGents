"""Server-Sent Events streaming for real-time agent transparency."""

import asyncio
import json
import time
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of events that can be streamed."""
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    DECISION_NODE = "decision_node"
    TOOL_EXECUTION = "tool_execution"
    RAG_QUERY = "rag_query"
    LLM_CALL = "llm_call"
    REASONING_STEP = "reasoning_step"
    ERROR = "error"
    STATUS_UPDATE = "status_update"
    TREE_TRAVERSAL = "tree_traversal"
    DATA_DISPLAY = "data_display"


class StreamEvent(BaseModel):
    """Event structure for SSE streaming."""
    event_type: EventType
    timestamp: float = Field(default_factory=time.time)
    session_id: str
    agent_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DecisionTreeEvent(StreamEvent):
    """Specialized event for decision tree traversal."""
    event_type: EventType = EventType.TREE_TRAVERSAL
    current_node: str
    next_node: Optional[str] = None
    decision_confidence: float
    reasoning: str
    tree_structure: Dict[str, Any] = Field(default_factory=dict)


class ToolExecutionEvent(StreamEvent):
    """Specialized event for tool execution."""
    event_type: EventType = EventType.TOOL_EXECUTION
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Any = None
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class DataDisplayEvent(StreamEvent):
    """Event for adaptive data display recommendations."""
    event_type: EventType = EventType.DATA_DISPLAY
    data_type: str
    schema: Dict[str, Any]
    recommended_renderer: str
    display_config: Dict[str, Any] = Field(default_factory=dict)


class EventStreamer:
    """Manages Server-Sent Events streaming for real-time transparency."""

    def __init__(self):
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        self.session_events: Dict[str, List[StreamEvent]] = {}

    async def subscribe(self, session_id: str) -> AsyncGenerator[str, None]:
        """Subscribe to events for a session."""
        if session_id not in self.subscribers:
            self.subscribers[session_id] = []

        queue = asyncio.Queue()
        self.subscribers[session_id].append(queue)

        try:
            # Send any existing events for this session
            if session_id in self.session_events:
                for event in self.session_events[session_id]:
                    yield self._format_sse_event(event)

            # Stream new events
            while True:
                event = await queue.get()
                if event is None:  # Shutdown signal
                    break
                yield self._format_sse_event(event)

        finally:
            # Clean up subscription
            if session_id in self.subscribers:
                self.subscribers[session_id].remove(queue)
                if not self.subscribers[session_id]:
                    del self.subscribers[session_id]

    async def emit_event(self, event: StreamEvent) -> None:
        """Emit an event to all subscribers of the session."""
        session_id = event.session_id

        # Store event for replay
        if session_id not in self.session_events:
            self.session_events[session_id] = []
        self.session_events[session_id].append(event)

        # Send to subscribers
        if session_id in self.subscribers:
            for queue in self.subscribers[session_id]:
                try:
                    await queue.put(event)
                except Exception:
                    # Remove broken subscriptions
                    continue

    def _format_sse_event(self, event: StreamEvent) -> str:
        """Format event for SSE transmission."""
        event_data = {
            "type": event.event_type,
            "timestamp": event.timestamp,
            "session_id": event.session_id,
            "agent_id": event.agent_id,
            "data": event.data,
            "metadata": event.metadata,
        }

        # Add specialized fields
        if isinstance(event, DecisionTreeEvent):
            event_data.update({
                "current_node": event.current_node,
                "next_node": event.next_node,
                "decision_confidence": event.decision_confidence,
                "reasoning": event.reasoning,
                "tree_structure": event.tree_structure,
            })
        elif isinstance(event, ToolExecutionEvent):
            event_data.update({
                "tool_name": event.tool_name,
                "inputs": event.inputs,
                "outputs": event.outputs,
                "execution_time": event.execution_time,
                "success": event.success,
                "error_message": event.error_message,
            })
        elif isinstance(event, DataDisplayEvent):
            event_data.update({
                "data_type": event.data_type,
                "schema": event.schema,
                "recommended_renderer": event.recommended_renderer,
                "display_config": event.display_config,
            })

        return f"data: {json.dumps(event_data)}\n\n"

    async def clear_session(self, session_id: str) -> None:
        """Clear events for a session."""
        if session_id in self.session_events:
            del self.session_events[session_id]

        # Notify subscribers of session end
        if session_id in self.subscribers:
            for queue in self.subscribers[session_id]:
                await queue.put(None)

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of events for a session."""
        if session_id not in self.session_events:
            return {"session_id": session_id, "events": [], "summary": {}}

        events = self.session_events[session_id]
        event_counts = {}
        for event in events:
            event_type = event.event_type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        total_time = 0
        if events:
            total_time = events[-1].timestamp - events[0].timestamp

        return {
            "session_id": session_id,
            "total_events": len(events),
            "event_counts": event_counts,
            "total_time": total_time,
            "events": [event.model_dump() for event in events],
        }


class StreamingAgentWrapper:
    """Wrapper to add streaming capabilities to agents."""

    def __init__(self, agent, streamer: EventStreamer, session_id: str):
        self.agent = agent
        self.streamer = streamer
        self.session_id = session_id
        self.agent_id = f"{agent.__class__.__name__}_{id(agent)}"

    async def process_message_with_streaming(self, message: str) -> str:
        """Process message with real-time event streaming."""
        # Emit start event
        await self.streamer.emit_event(
            StreamEvent(
                event_type=EventType.AGENT_STARTED,
                session_id=self.session_id,
                agent_id=self.agent_id,
                data={"message": message, "agent_type": self.agent.__class__.__name__},
            )
        )

        try:
            # Wrap agent methods to emit events
            self._wrap_agent_methods()

            # Process the message
            result = await self.agent.process_message(message)

            # Emit completion event
            await self.streamer.emit_event(
                StreamEvent(
                    event_type=EventType.AGENT_COMPLETED,
                    session_id=self.session_id,
                    agent_id=self.agent_id,
                    data={"result": result, "success": True},
                )
            )

            return result

        except Exception as e:
            # Emit error event
            await self.streamer.emit_event(
                StreamEvent(
                    event_type=EventType.ERROR,
                    session_id=self.session_id,
                    agent_id=self.agent_id,
                    data={"error": str(e), "error_type": type(e).__name__},
                )
            )
            raise

    def _wrap_agent_methods(self):
        """Wrap agent methods to emit events."""
        # Wrap decision tree node execution
        if hasattr(self.agent, '_execute_node'):
            original_execute_node = self.agent._execute_node

            async def wrapped_execute_node(node, *args, **kwargs):
                await self.streamer.emit_event(
                    DecisionTreeEvent(
                        session_id=self.session_id,
                        agent_id=self.agent_id,
                        current_node=node.node_id,
                        decision_confidence=0.8,  # Could be calculated
                        reasoning=f"Executing node: {node.name}",
                        tree_structure=self._get_tree_structure(),
                        data={"node_name": node.name, "node_type": node.node_type.value},
                    )
                )
                return await original_execute_node(node, *args, **kwargs)

            self.agent._execute_node = wrapped_execute_node

        # Wrap tool execution
        if hasattr(self.agent, 'tool_registry'):
            original_execute = self.agent.tool_registry.execute

            async def wrapped_tool_execute(tool_name, **kwargs):
                start_time = time.time()

                await self.streamer.emit_event(
                    ToolExecutionEvent(
                        session_id=self.session_id,
                        agent_id=self.agent_id,
                        tool_name=tool_name,
                        inputs=kwargs,
                        data={"status": "started"},
                    )
                )

                try:
                    result = await original_execute(tool_name, **kwargs)
                    execution_time = time.time() - start_time

                    await self.streamer.emit_event(
                        ToolExecutionEvent(
                            session_id=self.session_id,
                            agent_id=self.agent_id,
                            tool_name=tool_name,
                            inputs=kwargs,
                            outputs=result.result if hasattr(result, 'result') else result,
                            execution_time=execution_time,
                            success=True,
                        )
                    )

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time

                    await self.streamer.emit_event(
                        ToolExecutionEvent(
                            session_id=self.session_id,
                            agent_id=self.agent_id,
                            tool_name=tool_name,
                            inputs=kwargs,
                            execution_time=execution_time,
                            success=False,
                            error_message=str(e),
                        )
                    )
                    raise

            self.agent.tool_registry.execute = wrapped_tool_execute

    def _get_tree_structure(self) -> Dict[str, Any]:
        """Get current tree structure for visualization."""
        if hasattr(self.agent, 'planning_graph'):
            # Graph planner
            return {
                "type": "graph",
                "nodes": list(self.agent.planning_graph.nodes.keys()),
                "current_node": getattr(self.agent, 'current_node', None),
            }
        elif hasattr(self.agent, 'decision_tree'):
            # Decision tree agent
            return {
                "type": "decision_tree",
                "nodes": list(self.agent.decision_tree.keys()),
                "current_node": getattr(self.agent, 'current_node', None),
            }
        else:
            return {"type": "unknown"}


# Global streamer instance
_global_streamer = EventStreamer()


def get_event_streamer() -> EventStreamer:
    """Get the global event streamer."""
    return _global_streamer