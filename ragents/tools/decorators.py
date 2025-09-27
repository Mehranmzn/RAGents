"""Tool decorators inspired by Elysia's design pattern."""

from typing import Callable, Optional

from .base import get_tool_registry


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general",
    registry=None,
    **metadata_kwargs
):
    """
    Decorator to register a function as a tool.

    Inspired by Elysia's @tool decorator pattern but implemented independently
    to respect their BSD-3 license and avoid direct copying.

    Usage:
        @tool(name="my_tool", description="Does something useful")
        def my_function(param1: str, param2: int) -> str:
            return f"Result: {param1} {param2}"

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        category: Tool category for organization
        registry: Tool registry to use (defaults to global registry)
        **metadata_kwargs: Additional metadata for the tool
    """
    def decorator(func: Callable) -> Callable:
        # Use provided registry or global one
        tool_registry = registry or get_tool_registry()

        # Register the function as a tool
        tool_registry.register_function(
            func=func,
            name=name,
            description=description,
            category=category,
            **metadata_kwargs
        )

        # Return the original function unchanged
        return func

    return decorator


def rag_tool(name: Optional[str] = None, description: Optional[str] = None, **kwargs):
    """Decorator specifically for RAG-related tools."""
    return tool(name=name, description=description, category="rag", **kwargs)


def agent_tool(name: Optional[str] = None, description: Optional[str] = None, **kwargs):
    """Decorator specifically for agent-related tools."""
    return tool(name=name, description=description, category="agent", **kwargs)


def utility_tool(name: Optional[str] = None, description: Optional[str] = None, **kwargs):
    """Decorator for utility tools."""
    return tool(name=name, description=description, category="utility", **kwargs)


def data_tool(name: Optional[str] = None, description: Optional[str] = None, **kwargs):
    """Decorator for data processing tools."""
    return tool(name=name, description=description, category="data", **kwargs)