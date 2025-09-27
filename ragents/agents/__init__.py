"""Agent system module."""

from .base import Agent, AgentConfig, AgentState, SimpleAgent

# LangGraph-based agents
from .langgraph_base import (
    LangGraphAgent,
    LangGraphAgentState,
    LangGraphAgentResult,
)
from .langgraph_react import LangGraphReActAgent
from .langgraph_multi_agent import (
    LangGraphMultiAgent,
    AgentRole,
    AgentDefinition,
    MultiAgentState,
    create_research_team,
    create_analysis_team,
)

__all__ = [
    # Base agents
    "Agent",
    "AgentConfig",
    "AgentState",
    "SimpleAgent",

    # LangGraph agents
    "LangGraphAgent",
    "LangGraphAgentState",
    "LangGraphAgentResult",
    "LangGraphReActAgent",
    "LangGraphMultiAgent",
    "AgentRole",
    "AgentDefinition",
    "MultiAgentState",
    "create_research_team",
    "create_analysis_team",
]