"""BentoML service implementation for scalable RAGents deployment."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

import bentoml
from pydantic import BaseModel, Field
import opik

from ..agents.base import Agent, AgentConfig, SimpleAgent
from ..agents.langgraph_base import LangGraphAgent
from ..agents.langgraph_react import LangGraphReActAgent
from ..llm.client import LLMClient
from ..llm.types import ModelConfig
from ..rag.engine import RAGEngine
from ..rag.cache import RAGCache, CacheConfig
from ..config.rag_config import RAGConfig
from ..logical_llm.integration import LogicalAgent, LogicalLLMIntegration
from ..agents.checkpointing import create_checkpoint_saver, CheckpointConfig

logger = logging.getLogger(__name__)


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for agent queries."""

    query: str = Field(..., description="The user query to process")
    agent_type: str = Field(default="simple", description="Type of agent to use")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID")
    enable_logical_llm: bool = Field(default=True, description="Enable logical LLM processing")
    config_overrides: Optional[Dict[str, Any]] = Field(default=None, description="Agent config overrides")


class QueryResponse(BaseModel):
    """Response model for agent queries."""

    response: str = Field(..., description="Agent response to the query")
    thread_id: str = Field(..., description="Conversation thread ID")
    agent_type: str = Field(..., description="Agent type used")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    logical_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Logical LLM analysis")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    agents_loaded: int = Field(..., description="Number of agents loaded")
    cache_status: str = Field(..., description="Cache health status")
    checkpoint_status: str = Field(..., description="Checkpoint system status")
    version: str = Field(default="0.1.3", description="RAGents version")


class ServiceConfig(BaseModel):
    """Configuration for BentoML service."""

    model_config: ModelConfig
    rag_config: Optional[RAGConfig] = None
    cache_config: Optional[CacheConfig] = None
    checkpoint_config: Optional[CheckpointConfig] = None
    agent_configs: Optional[Dict[str, AgentConfig]] = None
    enable_opik: bool = Field(default=True, description="Enable Opik observability")


# BentoML Service
@bentoml.service(
    name="ragents",
    resources={
        "cpu": "4",
        "memory": "8Gi",
    },
    traffic={
        "timeout": 300,  # 5 minute timeout for complex queries
        "max_concurrency": 50,  # Handle up to 50 concurrent requests
    },
)
class RAGentsService:
    """
    Production-grade BentoML service for RAGents.

    Features:
    - Multiple agent types with LangGraph workflows
    - Distributed caching with Redis
    - Distributed checkpointing (Redis/PostgreSQL)
    - Opik observability integration
    - Automatic batching and load balancing
    - Health checks and metrics
    """

    def __init__(self):
        """Initialize the RAGents service."""
        self.logger = logging.getLogger(__name__)
        self.llm_client: Optional[LLMClient] = None
        self.rag_engine: Optional[RAGEngine] = None
        self.cache: Optional[RAGCache] = None
        self.checkpoint_saver = None
        self.agents: Dict[str, Agent] = {}
        self.logical_integration: Optional[LogicalLLMIntegration] = None
        self.opik_client: Optional[opik.Opik] = None
        self.config: Optional[ServiceConfig] = None

    async def initialize(self, config: ServiceConfig) -> None:
        """
        Initialize service components.

        This is called once when the service starts up.
        """
        self.config = config

        # Initialize Opik observability
        if config.enable_opik:
            try:
                self.opik_client = opik.Opik()
                self.logger.info("Opik observability initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Opik: {e}")

        # Initialize LLM client
        self.llm_client = LLMClient(config.model_config)
        self.logger.info(f"LLM client initialized with model: {config.model_config.model_name}")

        # Initialize caching layer
        if config.cache_config:
            try:
                from ..rag.cache import get_cache
                self.cache = await get_cache(config.cache_config)
                cache_health = await self.cache.health_check()
                self.logger.info(f"Cache initialized: {cache_health['status']}")
            except Exception as e:
                self.logger.warning(f"Cache initialization failed: {e}")

        # Initialize distributed checkpointing
        if config.checkpoint_config:
            try:
                self.checkpoint_saver = await create_checkpoint_saver(config.checkpoint_config)
                self.logger.info(f"Checkpoint saver initialized: {config.checkpoint_config.backend}")
            except Exception as e:
                self.logger.warning(f"Checkpoint initialization failed: {e}")

        # Initialize RAG engine
        if config.rag_config:
            self.rag_engine = RAGEngine(
                config.rag_config,
                self.llm_client,
                cache=self.cache,
                cache_config=config.cache_config
            )
            self.logger.info("RAG engine initialized")

        # Initialize logical LLM integration
        self.logical_integration = LogicalLLMIntegration(self.llm_client)

        # Initialize agents
        await self._initialize_agents(config.agent_configs)

        self.logger.info("RAGents BentoML service initialized successfully")

    async def _initialize_agents(self, agent_configs: Optional[Dict[str, AgentConfig]]) -> None:
        """Initialize different types of agents."""
        default_configs = {
            "simple": AgentConfig(
                name="SimpleAgent",
                description="Basic RAG agent with logical LLM enhancement",
                enable_rag=True
            ),
            "logical": AgentConfig(
                name="LogicalAgent",
                description="Agent with advanced logical reasoning",
                enable_reasoning=True,
                enable_query_rewriting=True,
                enable_rag=True
            ),
            "langgraph": AgentConfig(
                name="LangGraphAgent",
                description="Agent with LangGraph workflow orchestration",
                enable_reasoning=True,
                enable_rag=True,
                max_iterations=8
            ),
            "react": AgentConfig(
                name="ReActAgent",
                description="Reasoning and Acting agent with tool usage",
                enable_tools=True,
                enable_rag=True,
                max_iterations=10
            ),
        }

        # Override with user configurations
        if agent_configs:
            default_configs.update(agent_configs)

        # Create agent instances
        for agent_type, agent_config in default_configs.items():
            try:
                if agent_type == "simple":
                    self.agents[agent_type] = SimpleAgent(
                        agent_config,
                        self.llm_client,
                        self.rag_engine
                    )
                elif agent_type == "logical":
                    self.agents[agent_type] = LogicalAgent(
                        agent_config,
                        self.llm_client,
                        self.rag_engine
                    )
                elif agent_type == "langgraph":
                    self.agents[agent_type] = LangGraphAgent(
                        agent_config,
                        self.llm_client,
                        self.rag_engine,
                        checkpointer=self.checkpoint_saver
                    )
                elif agent_type == "react":
                    self.agents[agent_type] = LangGraphReActAgent(
                        agent_config,
                        self.llm_client,
                        self.rag_engine,
                        checkpointer=self.checkpoint_saver
                    )
                else:
                    # Default to simple agent
                    self.agents[agent_type] = SimpleAgent(
                        agent_config,
                        self.llm_client,
                        self.rag_engine
                    )

                self.logger.info(f"Initialized {agent_type} agent")

            except Exception as e:
                self.logger.error(f"Failed to initialize {agent_type} agent: {e}")
                # Use simple agent as fallback
                self.agents[agent_type] = SimpleAgent(
                    default_configs["simple"],
                    self.llm_client,
                    self.rag_engine
                )

    @bentoml.api(
        route="/query",
        batchable=True,
        max_batch_size=8,
        max_latency_ms=100,
    )
    @opik.track()
    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process agent query with intelligent batching.

        This endpoint supports automatic request batching for improved throughput.
        """
        import time

        start_time = time.time()

        try:
            # Track with Opik
            if self.opik_client:
                opik.track_metadata({
                    "agent_type": request.agent_type,
                    "query_length": len(request.query),
                    "thread_id": request.thread_id
                })

            # Generate thread ID if not provided
            thread_id = request.thread_id or f"thread_{int(time.time() * 1000)}"

            # Select agent
            agent = self.agents.get(request.agent_type, self.agents.get("simple"))
            if not agent:
                raise ValueError(f"Agent type {request.agent_type} not found")

            # Process with logical LLM if enabled and using logical agent
            logical_analysis = None
            if request.enable_logical_llm and request.agent_type == "logical":
                logical_result = await self.logical_integration.process_query(
                    request.query,
                    interactive_mode=False
                )
                logical_analysis = {
                    "domain": logical_result.logical_query.domain,
                    "confidence": logical_result.processing_confidence,
                    "token_reduction": logical_result.estimated_token_reduction,
                    "optimized_query": logical_result.optimized_search_query,
                }

            # Process query with agent
            if hasattr(agent, 'process_message'):
                # For LangGraph agents with thread support
                if request.agent_type in ["langgraph", "react"]:
                    result = await agent.process_message(request.query, thread_id)
                    response_text = result.response if hasattr(result, 'response') else str(result)
                else:
                    response_text = await agent.process_message(request.query)
            else:
                # Fallback for basic agents
                response_text = await agent.run(request.query)

            processing_time = time.time() - start_time

            # Track metrics with Opik
            if self.opik_client:
                opik.track_metadata({
                    "processing_time": processing_time,
                    "success": True,
                    "response_length": len(response_text)
                })

            return QueryResponse(
                response=response_text,
                thread_id=thread_id,
                agent_type=request.agent_type,
                processing_time=processing_time,
                metadata={
                    "model": self.config.model_config.model_name,
                    "timestamp": time.time(),
                },
                logical_analysis=logical_analysis
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing query: {e}")

            # Track error with Opik
            if self.opik_client:
                opik.track_metadata({
                    "error": str(e),
                    "processing_time": processing_time,
                    "success": False
                })

            return QueryResponse(
                response=f"Error processing request: {str(e)}",
                thread_id=request.thread_id or "error",
                agent_type=request.agent_type,
                processing_time=processing_time,
                metadata={"error": True, "error_message": str(e)}
            )

    @bentoml.api(route="/health")
    async def health(self) -> HealthResponse:
        """Health check endpoint."""
        try:
            # Check cache status
            cache_status = "disabled"
            if self.cache:
                cache_health = await self.cache.health_check()
                cache_status = cache_health["status"]

            # Check checkpoint status
            checkpoint_status = "not_configured"
            if self.checkpoint_saver:
                checkpoint_status = "healthy"

            return HealthResponse(
                status="healthy",
                agents_loaded=len(self.agents),
                cache_status=cache_status,
                checkpoint_status=checkpoint_status
            )

        except Exception as e:
            return HealthResponse(
                status="unhealthy",
                agents_loaded=len(self.agents),
                cache_status="error",
                checkpoint_status="error"
            )

    @bentoml.api(route="/agents")
    async def list_agents(self) -> Dict[str, List[str]]:
        """List available agent types."""
        return {
            "agents": list(self.agents.keys()),
            "default": "simple"
        }

    @bentoml.api(route="/cache/stats")
    async def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {"error": "Cache not enabled"}

        try:
            stats = await self.cache.get_stats()
            return stats.model_dump()
        except Exception as e:
            return {"error": str(e)}

    @bentoml.api(route="/cache/clear", method="POST")
    async def clear_cache(self) -> Dict[str, str]:
        """Clear all cache entries."""
        if not self.cache:
            return {"error": "Cache not enabled"}

        try:
            await self.cache.clear_all()
            return {"status": "success", "message": "Cache cleared"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Helper function to create and configure the service
def create_ragents_service(
    model_config: ModelConfig,
    rag_config: Optional[RAGConfig] = None,
    cache_config: Optional[CacheConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
) -> RAGentsService:
    """
    Create a configured RAGents BentoML service.

    Args:
        model_config: LLM model configuration
        rag_config: RAG engine configuration
        cache_config: Redis cache configuration
        checkpoint_config: Distributed checkpoint configuration

    Returns:
        Configured RAGentsService instance
    """
    service = RAGentsService()

    service_config = ServiceConfig(
        model_config=model_config,
        rag_config=rag_config,
        cache_config=cache_config,
        checkpoint_config=checkpoint_config,
        enable_opik=True
    )

    # Initialize service asynchronously
    asyncio.run(service.initialize(service_config))

    return service
