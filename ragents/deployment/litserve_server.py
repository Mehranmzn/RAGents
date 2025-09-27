"""LitServe-based server implementation for RAGents."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import litserve as ls
from pydantic import BaseModel, Field

from ..agents.base import Agent, AgentConfig, SimpleAgent
from ..agents.langgraph_base import LangGraphAgent, LangGraphAgentResult
from ..agents.langgraph_react import LangGraphReActAgent
from ..agents.langgraph_multi_agent import LangGraphMultiAgent
from ..llm.client import LLMClient
from ..llm.types import ModelConfig
from ..rag.engine import RAGEngine
from ..config.rag_config import RAGConfig
from ..logical_llm.integration import LogicalAgent, LogicalLLMIntegration
from .monitoring import MetricsCollector
from .health_checks import HealthChecker


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for agent queries."""

    query: str = Field(..., description="The user query to process")
    agent_type: str = Field(default="simple", description="Type of agent to use")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID")
    enable_logical_llm: bool = Field(default=True, description="Enable logical LLM processing")
    interactive_mode: bool = Field(default=False, description="Enable interactive clarification")
    config_overrides: Optional[Dict[str, Any]] = Field(default=None, description="Agent config overrides")


class QueryResponse(BaseModel):
    """Response model for agent queries."""

    response: str = Field(..., description="Agent response to the query")
    thread_id: str = Field(..., description="Conversation thread ID")
    agent_type: str = Field(..., description="Agent type used")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    logical_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Logical LLM analysis")
    clarification_requests: Optional[List[Dict[str, Any]]] = Field(default=None, description="Clarification requests")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Health status")
    timestamp: float = Field(..., description="Timestamp of health check")
    version: str = Field(..., description="RAGents version")
    components: Dict[str, str] = Field(..., description="Component health status")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="System metrics")


class MetricsResponse(BaseModel):
    """Metrics response model."""

    requests_total: int = Field(..., description="Total requests processed")
    requests_per_second: float = Field(..., description="Current RPS")
    average_response_time: float = Field(..., description="Average response time")
    error_rate: float = Field(..., description="Error rate percentage")
    active_threads: int = Field(..., description="Active conversation threads")
    memory_usage: float = Field(..., description="Memory usage in MB")
    cpu_usage: float = Field(..., description="CPU usage percentage")


@dataclass
class ServerConfig:
    """Configuration for RAGents LitServe server."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    timeout: int = 30
    max_batch_size: int = 1
    batch_timeout: float = 0.1
    enable_metrics: bool = True
    enable_health_checks: bool = True
    log_level: str = "INFO"
    model_config: Optional[ModelConfig] = None
    rag_config: Optional[RAGConfig] = None
    agent_configs: Optional[Dict[str, AgentConfig]] = None


class RAGentsAPILit(ls.LitAPI):
    """LitServe API implementation for RAGents."""

    def setup(self, device: str) -> None:
        """Setup the RAGents API with all components."""
        # Initialize logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)

        # Initialize metrics collector
        if self.config.enable_metrics:
            self.metrics = MetricsCollector()

        # Initialize health checker
        if self.config.enable_health_checks:
            self.health_checker = HealthChecker()

        # Initialize LLM client
        if not self.config.model_config:
            raise ValueError("Model configuration is required")

        self.llm_client = LLMClient(self.config.model_config)

        # Initialize RAG engine if configured
        self.rag_engine = None
        if self.config.rag_config:
            self.rag_engine = RAGEngine(self.config.rag_config, self.llm_client)

        # Initialize logical LLM integration
        self.logical_integration = LogicalLLMIntegration(self.llm_client)

        # Initialize agents
        self.agents = self._initialize_agents()

        # Track active conversations
        self.active_threads = {}

        self.logger.info("RAGents LitServe API initialized successfully")

    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize different types of agents."""
        agents = {}

        # Default agent configurations
        default_configs = {
            "simple": AgentConfig(
                name="SimpleAgent",
                description="Basic RAG agent with logical LLM enhancement"
            ),
            "logical": AgentConfig(
                name="LogicalAgent",
                description="Agent with advanced logical reasoning",
                enable_reasoning=True,
                enable_query_rewriting=True
            ),
            "langgraph": AgentConfig(
                name="LangGraphAgent",
                description="Agent with LangGraph workflow orchestration",
                enable_reasoning=True,
                max_iterations=8
            ),
            "react": AgentConfig(
                name="ReActAgent",
                description="Reasoning and Acting agent with tool usage",
                enable_tools=True,
                max_iterations=10
            ),
        }

        # Override with user configurations if provided
        if self.config.agent_configs:
            default_configs.update(self.config.agent_configs)

        # Create agent instances
        for agent_type, agent_config in default_configs.items():
            try:
                if agent_type == "simple":
                    agents[agent_type] = SimpleAgent(agent_config, self.llm_client, self.rag_engine)
                elif agent_type == "logical":
                    agents[agent_type] = LogicalAgent(agent_config, self.llm_client, self.rag_engine)
                elif agent_type == "langgraph":
                    agents[agent_type] = LangGraphAgent(agent_config, self.llm_client, self.rag_engine)
                elif agent_type == "react":
                    agents[agent_type] = LangGraphReActAgent(agent_config, self.llm_client, self.rag_engine)
                else:
                    # Default to simple agent for unknown types
                    agents[agent_type] = SimpleAgent(agent_config, self.llm_client, self.rag_engine)

                self.logger.info(f"Initialized {agent_type} agent")

            except Exception as e:
                self.logger.error(f"Failed to initialize {agent_type} agent: {e}")
                # Use simple agent as fallback
                agents[agent_type] = SimpleAgent(default_configs["simple"], self.llm_client, self.rag_engine)

        return agents

    def decode_request(self, request: Dict[str, Any]) -> QueryRequest:
        """Decode incoming request."""
        return QueryRequest(**request)

    def predict(self, request: QueryRequest) -> QueryResponse:
        """Process agent query request."""
        start_time = time.time()

        try:
            # Update metrics
            if hasattr(self, 'metrics'):
                self.metrics.increment_requests()

            # Generate thread ID if not provided
            thread_id = request.thread_id or f"thread_{int(time.time() * 1000)}"

            # Select agent
            agent = self.agents.get(request.agent_type, self.agents["simple"])

            # Process with logical LLM if enabled
            logical_analysis = None
            clarification_requests = None

            if request.enable_logical_llm and request.agent_type == "logical":
                # Use LogicalAgent which has built-in logical processing
                response_text = asyncio.run(agent.process_message(request.query))

                # Get logical analysis for metadata
                logical_result = asyncio.run(
                    self.logical_integration.process_query(request.query, request.interactive_mode)
                )
                logical_analysis = {
                    "domain": logical_result.logical_query.domain,
                    "confidence": logical_result.processing_confidence,
                    "token_reduction": logical_result.estimated_token_reduction,
                    "optimized_query": logical_result.optimized_search_query,
                    "retrieval_mode": logical_result.search_directive.mode.value,
                    "graph_entry_points": logical_result.search_directive.graph_entry_points,
                    "graph_query": logical_result.search_directive.graph_query,
                }

                if logical_result.clarification_requests:
                    clarification_requests = [
                        {
                            "field": req.field_name,
                            "question": req.question,
                            "options": req.options,
                            "priority": req.priority
                        }
                        for req in logical_result.clarification_requests
                    ]

            else:
                # Standard agent processing
                response_text = asyncio.run(agent.process_message(request.query))

            # Track active thread
            self.active_threads[thread_id] = {
                "last_activity": time.time(),
                "agent_type": request.agent_type,
                "message_count": self.active_threads.get(thread_id, {}).get("message_count", 0) + 1
            }

            processing_time = time.time() - start_time

            # Update metrics
            if hasattr(self, 'metrics'):
                self.metrics.record_response_time(processing_time)

            response = QueryResponse(
                response=response_text,
                thread_id=thread_id,
                agent_type=request.agent_type,
                processing_time=processing_time,
                metadata={
                    "model": self.config.model_config.model_name,
                    "timestamp": time.time(),
                    "message_count": self.active_threads[thread_id]["message_count"]
                },
                logical_analysis=logical_analysis,
                clarification_requests=clarification_requests
            )

            return response

        except Exception as e:
            # Update error metrics
            if hasattr(self, 'metrics'):
                self.metrics.increment_errors()

            self.logger.error(f"Error processing request: {e}")

            processing_time = time.time() - start_time

            return QueryResponse(
                response=f"Error processing request: {str(e)}",
                thread_id=request.thread_id or "error",
                agent_type=request.agent_type,
                processing_time=processing_time,
                metadata={"error": True, "error_message": str(e)}
            )

    def encode_response(self, response: QueryResponse) -> Dict[str, Any]:
        """Encode response for transmission."""
        return response.model_dump()


class RAGentsLitServer:
    """Main LitServe server for RAGents deployment."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.api = None
        self.server = None

    def create_api(self) -> RAGentsAPILit:
        """Create the LitServe API instance."""
        api = RAGentsAPILit()
        api.config = self.config
        return api

    def add_health_endpoint(self, server: ls.LitServer) -> None:
        """Add health check endpoint to server."""

        @server.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            try:
                health_status = "healthy"
                components = {"llm_client": "healthy", "rag_engine": "healthy"}

                # Check component health if health checker is available
                if hasattr(self.api, 'health_checker'):
                    health_result = self.api.health_checker.check_health()
                    health_status = health_result.status.value
                    components = health_result.components

                return HealthResponse(
                    status=health_status,
                    timestamp=time.time(),
                    version="0.1.0",
                    components=components,
                    metrics=self._get_current_metrics() if hasattr(self.api, 'metrics') else None
                ).model_dump()

            except Exception as e:
                return {
                    "status": "unhealthy",
                    "timestamp": time.time(),
                    "version": "0.1.0",
                    "error": str(e)
                }

    def add_metrics_endpoint(self, server: ls.LitServer) -> None:
        """Add metrics endpoint to server."""

        @server.app.get("/metrics")
        async def get_metrics():
            """Metrics endpoint."""
            try:
                if hasattr(self.api, 'metrics'):
                    metrics_data = self.api.metrics.get_metrics()

                    return MetricsResponse(
                        requests_total=metrics_data.get("requests_total", 0),
                        requests_per_second=metrics_data.get("requests_per_second", 0.0),
                        average_response_time=metrics_data.get("average_response_time", 0.0),
                        error_rate=metrics_data.get("error_rate", 0.0),
                        active_threads=len(self.api.active_threads),
                        memory_usage=metrics_data.get("memory_usage_mb", 0.0),
                        cpu_usage=metrics_data.get("cpu_usage_percent", 0.0)
                    ).model_dump()
                else:
                    return {"error": "Metrics not enabled"}

            except Exception as e:
                return {"error": f"Failed to get metrics: {str(e)}"}

    def add_custom_endpoints(self, server: ls.LitServer) -> None:
        """Add custom endpoints for RAGents functionality."""

        @server.app.get("/agents")
        async def list_agents():
            """List available agent types."""
            return {
                "agents": list(self.api.agents.keys()) if self.api else [],
                "default": "simple"
            }

        @server.app.get("/threads/{thread_id}")
        async def get_thread_info(thread_id: str):
            """Get information about a conversation thread."""
            if hasattr(self.api, 'active_threads'):
                thread_info = self.api.active_threads.get(thread_id)
                if thread_info:
                    return {
                        "thread_id": thread_id,
                        "last_activity": thread_info["last_activity"],
                        "agent_type": thread_info["agent_type"],
                        "message_count": thread_info["message_count"],
                        "active": time.time() - thread_info["last_activity"] < 3600  # 1 hour
                    }
            return {"error": "Thread not found"}

        @server.app.post("/threads/{thread_id}/reset")
        async def reset_thread(thread_id: str):
            """Reset a conversation thread."""
            if hasattr(self.api, 'active_threads'):
                if thread_id in self.api.active_threads:
                    del self.api.active_threads[thread_id]
                    return {"message": f"Thread {thread_id} reset successfully"}
            return {"error": "Thread not found"}

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics for health check."""
        if hasattr(self.api, 'metrics'):
            return self.api.metrics.get_metrics()
        return {}

    def start(self) -> None:
        """Start the LitServe server."""
        # Create API instance
        self.api = self.create_api()

        # Create LitServe server
        self.server = ls.LitServer(
            self.api,
            accelerator="auto",
            max_batch_size=self.config.max_batch_size,
            batch_timeout=self.config.batch_timeout,
            timeout=self.config.timeout,
            workers_per_device=self.config.workers
        )

        # Add custom endpoints
        self.add_health_endpoint(self.server)
        self.add_metrics_endpoint(self.server)
        self.add_custom_endpoints(self.server)

        # Start server
        print(f"Starting RAGents LitServe server on {self.config.host}:{self.config.port}")
        self.server.run(
            host=self.config.host,
            port=self.config.port,
            num_api_servers=self.config.workers
        )

    def stop(self) -> None:
        """Stop the server gracefully."""
        if self.server:
            print("Stopping RAGents LitServe server...")
            # LitServe handles graceful shutdown automatically


# Utility functions for easy server creation

def create_simple_server(
    model_config: ModelConfig,
    rag_config: Optional[RAGConfig] = None,
    host: str = "0.0.0.0",
    port: int = 8000
) -> RAGentsLitServer:
    """Create a simple RAGents server with basic configuration."""

    server_config = ServerConfig(
        host=host,
        port=port,
        model_config=model_config,
        rag_config=rag_config,
        workers=1,
        enable_metrics=True,
        enable_health_checks=True
    )

    return RAGentsLitServer(server_config)


def create_production_server(
    model_config: ModelConfig,
    rag_config: RAGConfig,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 4,
    max_batch_size: int = 8
) -> RAGentsLitServer:
    """Create a production-ready RAGents server with optimized configuration."""

    # Production agent configurations
    agent_configs = {
        "simple": AgentConfig(
            name="ProductionSimpleAgent",
            description="Production simple agent",
            enable_memory=True,
            memory_window=10
        ),
        "logical": AgentConfig(
            name="ProductionLogicalAgent",
            description="Production logical agent with optimization",
            enable_reasoning=True,
            enable_query_rewriting=True,
            enable_memory=True,
            memory_window=15
        ),
        "langgraph": AgentConfig(
            name="ProductionLangGraphAgent",
            description="Production LangGraph agent",
            enable_reasoning=True,
            max_iterations=6,
            enable_memory=True
        )
    }

    server_config = ServerConfig(
        host=host,
        port=port,
        workers=workers,
        max_batch_size=max_batch_size,
        batch_timeout=0.05,  # Faster batching for production
        timeout=60,  # Longer timeout for complex queries
        model_config=model_config,
        rag_config=rag_config,
        agent_configs=agent_configs,
        enable_metrics=True,
        enable_health_checks=True,
        log_level="INFO"
    )

    return RAGentsLitServer(server_config)
