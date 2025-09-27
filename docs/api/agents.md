# Agents API Reference

Complete API reference for RAGents agent classes and configurations.

## Agent Base Classes

### Agent

The base class for all RAGents agents.

```python
class Agent:
    """Base class for all RAGents agents."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        rag_engine: Optional[RAGEngine] = None,
        **kwargs
    ):
        """Initialize agent with configuration and dependencies."""
```

**Parameters:**
- `config` (AgentConfig): Agent configuration
- `llm_client` (LLMClient): LLM client for text generation
- `rag_engine` (RAGEngine, optional): RAG engine for knowledge retrieval
- `**kwargs`: Additional configuration options

**Methods:**

#### process_message()

```python
async def process_message(
    self,
    message: str,
    context: Optional[dict] = None,
    **kwargs
) -> Union[str, AgentResponse]:
    """Process a user message and return a response."""
```

**Parameters:**
- `message` (str): User message to process
- `context` (dict, optional): Additional context for processing
- `**kwargs`: Additional processing options

**Returns:**
- `Union[str, AgentResponse]`: Agent response

#### get_conversation_history()

```python
async def get_conversation_history(
    self,
    limit: Optional[int] = None
) -> List[dict]:
    """Get conversation history."""
```

#### clear_memory()

```python
async def clear_memory() -> None:
    """Clear agent memory."""
```

## Agent Implementations

### DecisionTreeAgent

Agent that uses configurable decision trees for action selection.

```python
class DecisionTreeAgent(Agent):
    """Agent with decision tree-based reasoning."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        rag_engine: Optional[RAGEngine] = None,
        decision_tree: Optional[List[DecisionNode]] = None
    ):
```

**Additional Parameters:**
- `decision_tree` (List[DecisionNode], optional): Custom decision tree

**Methods:**

#### add_decision_path()

```python
async def add_decision_path(
    self,
    condition: str,
    action: str,
    node_id: Optional[str] = None
) -> str:
    """Add a new decision path to the tree."""
```

#### update_node()

```python
async def update_node(
    self,
    node_id: str,
    **updates
) -> bool:
    """Update an existing decision node."""
```

#### get_execution_trace()

```python
def get_last_execution_trace() -> ExecutionTrace:
    """Get the last execution trace for debugging."""
```

### ReActAgent

Agent implementing the Reasoning + Acting pattern.

```python
class ReActAgent(Agent):
    """Agent using ReAct (Reasoning + Acting) pattern."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        rag_engine: Optional[RAGEngine] = None,
        max_iterations: int = 5
    ):
```

**Additional Parameters:**
- `max_iterations` (int): Maximum reasoning iterations

**Methods:**

#### get_reasoning_trace()

```python
def get_reasoning_trace() -> ReasoningTrace:
    """Get detailed reasoning trace from last execution."""
```

### GraphPlannerAgent

Agent using graph-based planning for complex tasks.

```python
class GraphPlannerAgent(Agent):
    """Agent with graph-based planning capabilities."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        rag_engine: Optional[RAGEngine] = None,
        planning_algorithm: str = "dfs"
    ):
```

**Additional Parameters:**
- `planning_algorithm` (str): Planning algorithm to use

**Methods:**

#### create_plan()

```python
async def create_plan(
    self,
    goal: str,
    constraints: Optional[List[str]] = None
) -> TaskGraph:
    """Create an execution plan for achieving a goal."""
```

#### execute_plan()

```python
async def execute_plan(
    self,
    plan: TaskGraph,
    monitor_progress: bool = True
) -> ExecutionResult:
    """Execute a task plan."""
```

## Configuration Classes

### AgentConfig

Main configuration class for agents.

```python
@dataclass
class AgentConfig:
    """Configuration for RAGents agents."""

    # Basic settings
    name: str
    description: Optional[str] = None

    # Core capabilities
    enable_rag: bool = False
    enable_memory: bool = True
    enable_tools: bool = True
    enable_reasoning: bool = False

    # Memory settings
    memory_window_size: int = 10
    memory_decay_factor: float = 0.95
    enable_memory_search: bool = True

    # LLM settings
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Tool settings
    allowed_tools: Optional[List[str]] = None
    tool_timeout: int = 30
    max_tool_calls: int = 10

    # Reasoning settings
    reasoning_depth: int = 3
    confidence_threshold: float = 0.7
    enable_logical_reasoning: bool = False

    # Query processing
    enable_query_rewriting: bool = False
    enable_query_expansion: bool = False

    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600
    max_retries: int = 3
    retry_delay: float = 1.0

    # Decision tree settings (for DecisionTreeAgent)
    decision_tree: Optional[List[DecisionNode]] = None
    enable_tree_caching: bool = True
    track_execution_path: bool = False

    # ReAct settings (for ReActAgent)
    max_iterations: int = 5
    early_stop_threshold: float = 0.9
    reflection_frequency: int = 2

    # Graph planning settings (for GraphPlannerAgent)
    planning_algorithm: str = "dfs"
    max_planning_depth: int = 10
    enable_resource_optimization: bool = False
```

**Class Methods:**

#### from_dict()

```python
@classmethod
def from_dict(cls, config_dict: dict) -> "AgentConfig":
    """Create configuration from dictionary."""
```

#### from_env()

```python
@classmethod
def from_env(cls, prefix: str = "RAGENTS_AGENT") -> "AgentConfig":
    """Create configuration from environment variables."""
```

#### to_dict()

```python
def to_dict(self) -> dict:
    """Convert configuration to dictionary."""
```

### DecisionNode

Configuration for decision tree nodes.

```python
@dataclass
class DecisionNode:
    """Decision node in agent decision tree."""

    id: str
    condition: str
    branches: Dict[str, str]
    condition_type: ConditionType = ConditionType.EXPRESSION
    context_variables: Optional[List[str]] = None
    timeout: Optional[int] = None
    cache_result: bool = True
    error_node: Optional[str] = None
```

**Enums:**

```python
class ConditionType(Enum):
    EXPRESSION = "expression"  # Python expression
    FUNCTION = "function"      # Function call
    LLM = "llm"               # LLM-based classification
    REGEX = "regex"           # Regular expression
```

### ActionNode

Configuration for action nodes in decision trees.

```python
@dataclass
class ActionNode:
    """Action node in agent decision tree."""

    id: str
    action: str
    tools: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    response_template: Optional[str] = None
    system_prompt: Optional[str] = None
    use_context: bool = True
    use_llm: bool = False
    next_node: Optional[str] = None

    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_action: Optional[str] = None

    # Performance
    timeout: Optional[int] = None
    cache_result: bool = False
```

## Response Classes

### AgentResponse

Standard response from agent processing.

```python
@dataclass
class AgentResponse:
    """Response from agent message processing."""

    content: str
    confidence: float = 0.0
    sources: Optional[List[Source]] = None
    reasoning_steps: Optional[List[ReasoningStep]] = None
    tool_calls: Optional[List[ToolCall]] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None

    # Decision tree specific
    execution_path: Optional[List[str]] = None
    decision_rationale: Optional[str] = None

    # ReAct specific
    reasoning_trace: Optional[ReasoningTrace] = None
    iterations_used: Optional[int] = None

    # Error information
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
```

### ReasoningStep

Individual step in reasoning process.

```python
@dataclass
class ReasoningStep:
    """Single step in agent reasoning process."""

    step_number: int
    thought: str
    action: Optional[str] = None
    tool_used: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 0.0
    duration: Optional[float] = None
    error: Optional[str] = None
```

### ExecutionTrace

Trace of decision tree execution.

```python
@dataclass
class ExecutionTrace:
    """Trace of decision tree execution."""

    steps: List[ExecutionStep]
    total_time: float
    success: bool
    final_node: str
    errors: List[str]
```

### ExecutionStep

Single step in execution trace.

```python
@dataclass
class ExecutionStep:
    """Single step in execution trace."""

    node_id: str
    node_type: str  # "decision" or "action"
    condition: Optional[str] = None
    result: Optional[Any] = None
    next_node: Optional[str] = None
    duration: float = 0.0
    error: Optional[str] = None
```

## Exception Classes

### AgentError

Base exception for agent-related errors.

```python
class AgentError(Exception):
    """Base exception for agent errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(message)
```

### DecisionTreeError

Exception for decision tree specific errors.

```python
class DecisionTreeError(AgentError):
    """Exception for decision tree execution errors."""

    def __init__(
        self,
        message: str,
        node_id: Optional[str] = None,
        execution_path: Optional[List[str]] = None,
        **kwargs
    ):
        self.node_id = node_id
        self.execution_path = execution_path
        super().__init__(message, **kwargs)
```

### ReActError

Exception for ReAct agent errors.

```python
class ReActError(AgentError):
    """Exception for ReAct agent errors."""

    def __init__(
        self,
        message: str,
        iteration: Optional[int] = None,
        reasoning_trace: Optional[ReasoningTrace] = None,
        **kwargs
    ):
        self.iteration = iteration
        self.reasoning_trace = reasoning_trace
        super().__init__(message, **kwargs)
```

## Utility Functions

### create_agent()

Factory function for creating agents.

```python
def create_agent(
    agent_type: str,
    config: AgentConfig,
    llm_client: LLMClient,
    rag_engine: Optional[RAGEngine] = None,
    **kwargs
) -> Agent:
    """Factory function to create agents of different types."""
```

**Parameters:**
- `agent_type` (str): Type of agent ("decision_tree", "react", "graph_planner")
- `config` (AgentConfig): Agent configuration
- `llm_client` (LLMClient): LLM client
- `rag_engine` (RAGEngine, optional): RAG engine
- `**kwargs`: Additional type-specific parameters

### validate_config()

Validate agent configuration.

```python
def validate_config(config: AgentConfig) -> Tuple[bool, List[str]]:
    """Validate agent configuration."""
```

**Returns:**
- `Tuple[bool, List[str]]`: (is_valid, error_messages)

## Usage Examples

### Basic Agent Creation

```python
from ragents import DecisionTreeAgent, AgentConfig
from ragents.llm.client import LLMClient

# Create configuration
config = AgentConfig(
    name="My Agent",
    enable_rag=True,
    enable_tools=True,
    temperature=0.1
)

# Create agent
agent = DecisionTreeAgent(
    config=config,
    llm_client=llm_client,
    rag_engine=rag_engine
)

# Process message
response = await agent.process_message("Hello!")
print(response.content)
```

### Custom Decision Tree

```python
from ragents.agents.decision_tree import DecisionNode, ActionNode

# Define custom decision tree
decision_tree = [
    DecisionNode(
        id="intent_check",
        condition="analyze_intent(message)",
        branches={
            "question": "handle_question",
            "greeting": "handle_greeting",
            "default": "general_response"
        }
    ),
    ActionNode(
        id="handle_question",
        action="search_knowledge",
        tools=["rag_search"],
        next_node="formulate_response"
    )
]

# Create agent with custom tree
config = AgentConfig(
    name="Custom Agent",
    decision_tree=decision_tree
)

agent = DecisionTreeAgent(config=config, llm_client=llm_client)
```

### Error Handling

```python
try:
    response = await agent.process_message("Complex query")
    print(response.content)
except DecisionTreeError as e:
    print(f"Decision tree error at node {e.node_id}: {e.message}")
    print(f"Execution path: {e.execution_path}")
except ReActError as e:
    print(f"ReAct error at iteration {e.iteration}: {e.message}")
except AgentError as e:
    print(f"General agent error: {e.message}")
    print(f"Error code: {e.error_code}")
    print(f"Context: {e.context}")
```

## Performance Monitoring

### Agent Metrics

```python
from ragents.observability import AgentMetrics

metrics = AgentMetrics(agent)

# Get performance statistics
stats = metrics.get_performance_stats()
print(f"Average response time: {stats.avg_response_time}")
print(f"Success rate: {stats.success_rate}")
print(f"Tool usage: {stats.tool_usage_distribution}")

# Get memory usage
memory_stats = metrics.get_memory_stats()
print(f"Memory usage: {memory_stats.current_usage}")
print(f"Conversation count: {memory_stats.conversation_count}")
```

### Custom Metrics

```python
# Add custom metric tracking
@metrics.track_metric("custom_processing_quality")
async def track_quality(message: str, response: AgentResponse):
    # Your custom quality assessment
    quality_score = assess_response_quality(message, response)
    return {"quality": quality_score, "message_length": len(message)}

# Use in agent processing
response = await agent.process_message(message)
await track_quality(message, response)
```

## Next Steps

- **[RAG API Reference](rag.md)** - RAG engine API documentation
- **[LLM API Reference](llm.md)** - LLM client API documentation
- **[Tools API Reference](tools.md)** - Tools system API documentation
- **[Custom Agents Guide](../agents/custom.md)** - Building custom agents