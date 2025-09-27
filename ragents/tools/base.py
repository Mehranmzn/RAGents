"""Base tool classes and registry."""

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolMetadata(BaseModel):
    """Metadata for a tool."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    returns: Optional[str] = None
    examples: List[str] = Field(default_factory=list)
    category: str = "general"
    requires_auth: bool = False


class ToolResult(BaseModel):
    """Result of tool execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """Base class for tools."""

    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    @property
    def name(self) -> str:
        """Get tool name."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return self.metadata.description

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "parameters": self.metadata.parameters,
            "returns": self.metadata.returns,
        }


class FunctionTool(Tool):
    """Tool that wraps a function."""

    def __init__(self, func: Callable, metadata: ToolMetadata):
        super().__init__(metadata)
        self.func = func
        self.is_async = asyncio.iscoroutinefunction(func)

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the wrapped function."""
        import time
        start_time = time.time()

        try:
            if self.is_async:
                result = await self.func(**kwargs)
            else:
                result = self.func(**kwargs)

            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.categories: Dict[str, List[str]] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

        # Add to category
        category = tool.metadata.category
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(tool.name)

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: str = "general",
        **metadata_kwargs
    ) -> Tool:
        """Register a function as a tool."""
        # Auto-generate metadata from function
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        # Extract parameters from function signature
        sig = inspect.signature(func)
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # Default type

            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_info["type"] = "integer"
                elif param.annotation == float:
                    param_info["type"] = "number"
                elif param.annotation == bool:
                    param_info["type"] = "boolean"
                elif param.annotation == list:
                    param_info["type"] = "array"
                elif param.annotation == dict:
                    param_info["type"] = "object"

            parameters["properties"][param_name] = param_info

            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        metadata = ToolMetadata(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            category=category,
            **metadata_kwargs
        )

        tool = FunctionTool(func, metadata)
        self.register(tool)
        return tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List available tools, optionally filtered by category."""
        if category:
            return self.categories.get(category, [])
        return list(self.tools.keys())

    def get_tools_for_llm(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tool schemas formatted for LLM function calling."""
        tool_names = self.list_tools(category)
        return [self.tools[name].get_schema() for name in tool_names]

    async def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found"
            )

        return await tool.execute(**kwargs)

    def remove(self, name: str) -> bool:
        """Remove a tool from registry."""
        if name in self.tools:
            tool = self.tools[name]
            del self.tools[name]

            # Remove from category
            category = tool.metadata.category
            if category in self.categories and name in self.categories[category]:
                self.categories[category].remove(name)

            return True
        return False

    def clear(self) -> None:
        """Clear all tools."""
        self.tools.clear()
        self.categories.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_tools": len(self.tools),
            "categories": dict(self.categories),
            "category_counts": {cat: len(tools) for cat, tools in self.categories.items()}
        }


# Global registry instance
_global_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry