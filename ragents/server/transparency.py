"""Transparency engine for adaptive data display inspired by Elysia's approach."""

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .events import DataDisplayEvent, EventStreamer, get_event_streamer


class SchemaAnalyzer:
    """Analyzes data schemas to recommend appropriate display renderers."""

    @staticmethod
    def analyze_data(data: Any) -> Dict[str, Any]:
        """Analyze data structure and recommend display format."""
        if isinstance(data, list) and data:
            return SchemaAnalyzer._analyze_list(data)
        elif isinstance(data, dict):
            return SchemaAnalyzer._analyze_dict(data)
        else:
            return {
                "type": "scalar",
                "data_type": type(data).__name__,
                "recommended_renderer": "text",
                "confidence": 1.0,
            }

    @staticmethod
    def _analyze_list(data: List[Any]) -> Dict[str, Any]:
        """Analyze list data structure."""
        if not data:
            return {"type": "empty_list", "recommended_renderer": "text"}

        first_item = data[0]
        all_same_type = all(type(item) == type(first_item) for item in data)

        if isinstance(first_item, dict):
            # List of objects - good for table/cards
            keys = set()
            for item in data:
                if isinstance(item, dict):
                    keys.update(item.keys())

            schema = {
                "type": "object_list",
                "item_count": len(data),
                "all_same_structure": all_same_type,
                "common_keys": list(keys),
                "sample_item": first_item,
            }

            # Recommend renderer based on data characteristics
            if len(keys) <= 3:
                renderer = "cards"
            elif len(keys) <= 8:
                renderer = "table"
            else:
                renderer = "detailed_list"

            # Check for special patterns
            if "name" in keys and "value" in keys:
                renderer = "key_value_pairs"
            elif any(key in keys for key in ["timestamp", "date", "time"]):
                renderer = "timeline"
            elif any(key in keys for key in ["count", "amount", "value", "score"]):
                renderer = "chart"

            schema.update({
                "recommended_renderer": renderer,
                "confidence": 0.9 if all_same_type else 0.7,
            })

            return schema

        elif isinstance(first_item, (int, float)):
            # Numeric list - good for charts
            return {
                "type": "numeric_list",
                "item_count": len(data),
                "recommended_renderer": "line_chart",
                "confidence": 0.8,
                "data_range": [min(data), max(data)] if all(isinstance(x, (int, float)) for x in data) else None,
            }

        else:
            # Generic list
            return {
                "type": "generic_list",
                "item_count": len(data),
                "recommended_renderer": "list",
                "confidence": 0.6,
            }

    @staticmethod
    def _analyze_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dictionary data structure."""
        keys = list(data.keys())
        values = list(data.values())

        # Check for specific patterns
        if all(isinstance(v, (int, float)) for v in values):
            # Numeric values - good for charts
            return {
                "type": "numeric_dict",
                "key_count": len(keys),
                "recommended_renderer": "bar_chart",
                "confidence": 0.8,
                "keys": keys,
            }

        elif all(isinstance(v, str) for v in values):
            # String values - good for key-value display
            return {
                "type": "string_dict",
                "key_count": len(keys),
                "recommended_renderer": "key_value_pairs",
                "confidence": 0.7,
                "keys": keys,
            }

        elif any(isinstance(v, list) for v in values):
            # Contains lists - complex structure
            return {
                "type": "complex_dict",
                "key_count": len(keys),
                "recommended_renderer": "expandable_tree",
                "confidence": 0.6,
                "keys": keys,
            }

        else:
            # Mixed types
            return {
                "type": "mixed_dict",
                "key_count": len(keys),
                "recommended_renderer": "property_grid",
                "confidence": 0.5,
                "keys": keys,
            }


class DisplayRenderer:
    """Defines display configurations for different renderer types."""

    RENDERER_CONFIGS = {
        "text": {
            "component": "TextDisplay",
            "props": {"fontSize": "14px", "fontFamily": "monospace"},
        },
        "table": {
            "component": "DataTable",
            "props": {
                "sortable": True,
                "filterable": True,
                "pagination": True,
                "pageSize": 25,
            },
        },
        "cards": {
            "component": "CardGrid",
            "props": {
                "columns": "auto",
                "cardSize": "medium",
                "showBorder": True,
            },
        },
        "chart": {
            "component": "Chart",
            "props": {
                "type": "auto",
                "responsive": True,
                "showLegend": True,
            },
        },
        "line_chart": {
            "component": "LineChart",
            "props": {
                "responsive": True,
                "showPoints": True,
                "smooth": True,
            },
        },
        "bar_chart": {
            "component": "BarChart",
            "props": {
                "responsive": True,
                "showValues": True,
                "orientation": "vertical",
            },
        },
        "key_value_pairs": {
            "component": "KeyValueDisplay",
            "props": {
                "layout": "rows",
                "showBorder": True,
                "highlightKeys": True,
            },
        },
        "timeline": {
            "component": "Timeline",
            "props": {
                "sortBy": "timestamp",
                "showTime": True,
                "interactive": True,
            },
        },
        "list": {
            "component": "SimpleList",
            "props": {
                "bulletStyle": "disc",
                "spacing": "normal",
            },
        },
        "detailed_list": {
            "component": "DetailedList",
            "props": {
                "expandable": True,
                "showIndex": True,
                "maxHeight": "400px",
            },
        },
        "expandable_tree": {
            "component": "TreeView",
            "props": {
                "expandable": True,
                "showLines": True,
                "defaultExpanded": False,
            },
        },
        "property_grid": {
            "component": "PropertyGrid",
            "props": {
                "groupable": True,
                "editable": False,
                "showTypes": True,
            },
        },
    }

    @classmethod
    def get_config(cls, renderer_type: str) -> Dict[str, Any]:
        """Get configuration for a renderer type."""
        return cls.RENDERER_CONFIGS.get(renderer_type, cls.RENDERER_CONFIGS["text"])


class TransparencyEngine:
    """
    Engine for providing real-time transparency into agent operations.

    Inspired by Elysia's transparency approach but implemented independently.
    Provides adaptive data display and real-time reasoning visualization.
    """

    def __init__(self, streamer: Optional[EventStreamer] = None):
        self.streamer = streamer or get_event_streamer()
        self.schema_analyzer = SchemaAnalyzer()

    async def display_data(
        self,
        data: Any,
        session_id: str,
        title: Optional[str] = None,
        context: Optional[str] = None,
    ) -> None:
        """
        Display data with adaptive rendering based on schema analysis.

        This provides similar functionality to Elysia's dynamic data display
        but implemented independently with our own analysis algorithms.
        """
        # Analyze data structure
        analysis = self.schema_analyzer.analyze_data(data)

        # Get recommended renderer
        renderer_type = analysis["recommended_renderer"]
        display_config = DisplayRenderer.get_config(renderer_type)

        # Create display event
        display_event = DataDisplayEvent(
            session_id=session_id,
            data_type=analysis["type"],
            schema=analysis,
            recommended_renderer=renderer_type,
            display_config=display_config,
            data={
                "content": data,
                "title": title,
                "context": context,
                "analysis": analysis,
            },
        )

        # Stream the display event
        await self.streamer.emit_event(display_event)

    async def show_reasoning_step(
        self,
        session_id: str,
        step_number: int,
        description: str,
        reasoning: str,
        confidence: float,
        evidence: Optional[Any] = None,
    ) -> None:
        """Show a reasoning step with supporting evidence."""
        reasoning_data = {
            "step_number": step_number,
            "description": description,
            "reasoning": reasoning,
            "confidence": confidence,
            "evidence": evidence,
        }

        # Display evidence if provided
        if evidence is not None:
            await self.display_data(
                evidence,
                session_id,
                title=f"Evidence for Step {step_number}",
                context=description,
            )

        # Stream reasoning event
        from .events import StreamEvent, EventType

        await self.streamer.emit_event(
            StreamEvent(
                event_type=EventType.REASONING_STEP,
                session_id=session_id,
                data=reasoning_data,
            )
        )

    async def visualize_decision_tree(
        self,
        session_id: str,
        tree_structure: Dict[str, Any],
        current_node: str,
        traversal_path: List[str],
    ) -> None:
        """Visualize decision tree with current position and traversal path."""
        tree_data = {
            "structure": tree_structure,
            "current_node": current_node,
            "traversal_path": traversal_path,
            "visualization_type": "decision_tree",
        }

        await self.display_data(
            tree_data,
            session_id,
            title="Decision Tree Traversal",
            context="Real-time agent decision process",
        )

    async def show_tool_results(
        self,
        session_id: str,
        tool_name: str,
        inputs: Dict[str, Any],
        outputs: Any,
        execution_time: float,
    ) -> None:
        """Show tool execution results with adaptive display."""
        # Display inputs
        await self.display_data(
            inputs,
            session_id,
            title=f"{tool_name} - Inputs",
            context=f"Tool execution inputs for {tool_name}",
        )

        # Display outputs with adaptive rendering
        await self.display_data(
            outputs,
            session_id,
            title=f"{tool_name} - Results",
            context=f"Tool execution results (took {execution_time:.2f}s)",
        )

    async def show_rag_context(
        self,
        session_id: str,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        confidence_scores: List[float],
    ) -> None:
        """Show RAG retrieval context with relevance visualization."""
        # Combine chunks with scores for display
        context_data = []
        for i, (chunk, score) in enumerate(zip(retrieved_chunks, confidence_scores)):
            context_data.append(
                {
                    "rank": i + 1,
                    "content": chunk.get("content", ""),
                    "source": chunk.get("source", "Unknown"),
                    "relevance_score": score,
                    "metadata": chunk.get("metadata", {}),
                }
            )

        await self.display_data(
            context_data,
            session_id,
            title="Retrieved Context",
            context=f"RAG retrieval results for: {query}",
        )

    def create_transparency_report(self, session_id: str) -> Dict[str, Any]:
        """Create a comprehensive transparency report for a session."""
        session_summary = self.streamer.get_session_summary(session_id)

        # Analyze session patterns
        events = session_summary.get("events", [])
        transparency_metrics = {
            "total_decisions": sum(
                1 for event in events if event.get("event_type") == "tree_traversal"
            ),
            "tools_used": len(
                set(
                    event.get("tool_name")
                    for event in events
                    if event.get("event_type") == "tool_execution"
                    and event.get("tool_name")
                )
            ),
            "reasoning_steps": sum(
                1 for event in events if event.get("event_type") == "reasoning_step"
            ),
            "data_displays": sum(
                1 for event in events if event.get("event_type") == "data_display"
            ),
            "session_duration": session_summary.get("total_time", 0),
        }

        return {
            "session_summary": session_summary,
            "transparency_metrics": transparency_metrics,
            "report_generated_at": json.dumps({"timestamp": "now"}),
        }