"""
Elysia compatibility layer for tool migration.

This module provides compatibility patterns inspired by Elysia's design
while respecting their BSD-3-Clause license. We implement similar
functionality independently rather than copying code directly.

Reference: https://github.com/weaviate/Elysia (BSD-3-Clause License)
"""

from typing import Any, Callable, Dict, List, Optional

from .base import ToolRegistry, get_tool_registry
from .decorators import tool


class ElysiaCompatibilityLayer:
    """
    Compatibility layer for Elysia-style tool patterns.

    This class provides an interface similar to Elysia's Tree functionality
    but implemented independently. It's designed to help users migrate
    from Elysia or use similar patterns without license conflicts.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or get_tool_registry()
        self.decision_tree = {}

    def add_tool(self, func: Callable, **metadata) -> Callable:
        """
        Add a tool to the compatibility layer.

        This provides similar functionality to Elysia's tool registration
        but uses our independent implementation.
        """
        return tool(**metadata)(func)

    def create_tree_structure(self, tools: List[str], decision_logic: Dict[str, Any]) -> None:
        """
        Create a decision tree structure for tool selection.

        This provides tree-like functionality inspired by Elysia's approach
        but implemented independently.
        """
        self.decision_tree = {
            "tools": tools,
            "logic": decision_logic,
            "type": "compatibility_tree"
        }

    def get_tool_for_context(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Select appropriate tool based on context.

        This mimics Elysia's context-based tool selection pattern.
        """
        if not self.decision_tree:
            return None

        # Simple rule-based selection (can be enhanced)
        available_tools = self.decision_tree.get("tools", [])

        # Example decision logic
        if "query" in context:
            query = context["query"].lower()
            if "search" in query or "find" in query:
                return next((tool for tool in available_tools if "search" in tool), None)
            elif "calculate" in query or "math" in query:
                return next((tool for tool in available_tools if "calc" in tool), None)

        # Default to first available tool
        return available_tools[0] if available_tools else None

    def preprocess_input(self, input_data: Any) -> Dict[str, Any]:
        """
        Preprocess input data similar to Elysia's preprocessing pattern.

        This provides preprocessing functionality inspired by Elysia's approach.
        """
        if isinstance(input_data, str):
            return {
                "text": input_data,
                "length": len(input_data),
                "type": "string",
                "preprocessing_applied": True
            }
        elif isinstance(input_data, dict):
            return {
                **input_data,
                "preprocessing_applied": True,
                "processed_keys": list(input_data.keys())
            }
        else:
            return {
                "data": input_data,
                "type": type(input_data).__name__,
                "preprocessing_applied": True
            }

    def create_weaviate_connection_config(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create Weaviate connection configuration.

        This provides similar functionality to Elysia's Weaviate client setup
        but uses our independent vector store interface.
        """
        config = {
            "store_type": "weaviate",
            "url": url,
            "api_key": api_key,
            "collection_name": kwargs.get("collection_name", "default"),
            "embedding_dimension": kwargs.get("embedding_dimension", 384),
        }

        # Add extra configuration
        for key, value in kwargs.items():
            if key not in config:
                config[key] = value

        return config

    def migrate_elysia_tools(self, elysia_tools: List[Dict[str, Any]]) -> List[str]:
        """
        Migrate tools from Elysia format to RAGents format.

        This helps users transition from Elysia to RAGents by providing
        a migration path for existing tool definitions.
        """
        migrated_tools = []

        for elysia_tool in elysia_tools:
            # Extract Elysia tool information
            name = elysia_tool.get("name")
            description = elysia_tool.get("description", "")
            func = elysia_tool.get("function")

            if not name or not func:
                continue

            # Register in our system
            self.registry.register_function(
                func=func,
                name=name,
                description=description,
                category="migrated_from_elysia"
            )

            migrated_tools.append(name)

        return migrated_tools

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for compatibility tracking."""
        return {
            "total_tools": len(self.registry.tools),
            "decision_tree_configured": bool(self.decision_tree),
            "compatibility_layer": "elysia_inspired",
            "license_compliance": "independent_implementation",
        }


def create_elysia_style_tool(
    name: str,
    description: str,
    func: Callable,
    tree_position: Optional[str] = None
) -> Callable:
    """
    Create a tool using Elysia-style patterns.

    This function provides a familiar interface for users coming from Elysia
    while using our independent implementation.
    """
    # Register the tool
    decorated_func = tool(name=name, description=description)(func)

    # Store tree position metadata if provided
    if tree_position:
        decorated_func._tree_position = tree_position

    return decorated_func


# License compliance note
def get_license_info() -> Dict[str, str]:
    """
    Get license information for compliance.

    This ensures transparency about our relationship with Elysia.
    """
    return {
        "ragents_license": "MIT",
        "elysia_reference": "BSD-3-Clause",
        "implementation": "independent",
        "inspiration": "Elysia design patterns respected and reimplemented",
        "compliance_note": "No Elysia code copied; patterns inspired and independently implemented"
    }