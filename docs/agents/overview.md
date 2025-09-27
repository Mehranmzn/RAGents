# Agents Overview

RAGents provides multiple intelligent agent types, each designed for different use cases and reasoning patterns. All agents share a common foundation while offering specialized capabilities.

## Agent Architecture

### Base Agent Class

All agents inherit from the `Agent` base class, which provides:

- **Memory Management** - Conversation history and context tracking
- **Tool Integration** - Access to calculators, RAG search, and custom tools
- **Async Processing** - Non-blocking message processing
- **Configuration** - Flexible agent behavior customization
- **Observability** - Built-in tracing and monitoring

### Common Features

**Memory System:**
- Configurable conversation window
- Automatic context management
- Memory recall capabilities

**Tool Access:**
- Calculator for mathematical operations
- RAG search for knowledge retrieval
- Memory recall for conversation history
- Custom tool registration

**Structured Thinking:**
- Step-by-step reasoning
- Confidence scoring
- Decision tracking

## Agent Types

### 1. Decision Tree Agent

The **Decision Tree Agent** uses configurable decision trees for sophisticated reasoning and action selection.

**Best For:**
- Complex decision-making scenarios
- Rule-based reasoning
- Structured problem-solving
- Deterministic workflows

**Key Features:**
- Configurable decision nodes
- Action selection based on context
- Tool integration at decision points
- Memory-aware decision making

**Example Use Cases:**
- Customer support workflows
- Content moderation
- Data analysis pipelines
- Troubleshooting guides

### 2. ReAct Agent

The **ReAct Agent** implements the Reasoning + Acting pattern, combining step-by-step reasoning with tool usage.

**Best For:**
- Multi-step problem solving
- Research and analysis tasks
- Interactive tool usage
- Exploratory workflows

**Key Features:**
- Iterative reasoning cycles
- Dynamic tool selection
- Self-correcting behavior
- Transparent thought process

**Example Use Cases:**
- Research assistance
- Data exploration
- Technical troubleshooting
- Creative problem solving

### 3. Graph Planner Agent

The **Graph Planner Agent** uses graph-based planning with depth-first search for complex task orchestration.

**Best For:**
- Multi-step planning
- Resource optimization
- Dependency management
- Complex workflows

**Key Features:**
- Graph-based task representation
- DFS planning algorithm
- Dependency resolution
- Resource allocation

**Example Use Cases:**
- Project planning
- Resource scheduling
- Workflow orchestration
- Task automation

### 4. LangGraph Integration

RAGents integrates with **LangGraph** for advanced workflow capabilities:

- `LangGraphBaseAgent` - Basic workflow integration
- `LangGraphReActAgent` - ReAct pattern with LangGraph
- `LangGraphMultiAgent` - Multi-agent coordination

## Agent Configuration

### Basic Configuration

```python
from ragents import AgentConfig

config = AgentConfig(
    name="My Agent",
    description="A helpful assistant",

    # Core Features
    enable_rag=True,           # Enable RAG capabilities
    enable_memory=True,        # Enable conversation memory
    enable_tools=True,         # Enable tool access
    enable_reasoning=True,     # Enable structured thinking

    # Memory Settings
    memory_window_size=10,     # Number of messages to remember

    # LLM Settings
    temperature=0.1,           # Model creativity
    max_tokens=2048,          # Response length limit

    # Query Processing
    enable_query_rewriting=True,  # Optimize queries
    enable_logical_reasoning=True # Use Logic-LLM
)
```

### Advanced Configuration

```python
from ragents import AgentConfig, DecisionTreeAgent
from ragents.agents.decision_tree import DecisionNode, ActionNode

# Configure decision tree structure
decision_nodes = [
    DecisionNode(
        condition="user_intent == 'question'",
        true_action="search_knowledge",
        false_action="general_response"
    ),
    ActionNode(
        action="search_knowledge",
        tools=["rag_search", "calculator"]
    )
]

config = AgentConfig(
    name="Advanced Agent",
    decision_tree_config=decision_nodes,
    reasoning_depth=3,
    confidence_threshold=0.8
)
```

## Memory Management

### Conversation Memory

Agents maintain conversation context automatically:

```python
# Memory is managed automatically
response1 = await agent.process_message("My name is Alice")
response2 = await agent.process_message("What's my name?")
# Agent remembers Alice from previous message
```

### Memory Configuration

```python
config = AgentConfig(
    enable_memory=True,
    memory_window_size=20,      # Keep last 20 messages
    memory_decay_factor=0.95,   # Gradually reduce old message importance
    enable_memory_search=True   # Allow searching conversation history
)
```

### Manual Memory Control

```python
# Access conversation history
history = await agent.get_conversation_history()

# Clear memory
await agent.clear_memory()

# Add specific memory
await agent.add_memory("Important fact to remember")
```

## Tool Integration

### Built-in Tools

All agents have access to standard tools:

- **Calculator** - Mathematical operations
- **RAG Search** - Knowledge base queries
- **Memory Recall** - Conversation history search
- **Time** - Current date and time

### Custom Tools

```python
from ragents.tools import tool

@tool(name="weather", description="Get current weather")
async def get_weather(location: str) -> str:
    # Your weather API integration
    return f"Weather in {location}: Sunny, 75°F"

# Tools are automatically discovered and registered
```

### Tool Usage Control

```python
config = AgentConfig(
    enable_tools=True,
    allowed_tools=["calculator", "rag_search"],  # Restrict tool access
    tool_timeout=30,                             # Tool execution timeout
    max_tool_calls=5                            # Limit tool usage per message
)
```

## Reasoning Capabilities

### Structured Thinking

Agents can break down complex problems:

```python
from ragents.llm.types import StructuredThought

# Agents automatically use structured thinking for complex queries
response = await agent.process_message("How do I optimize my database performance?")

# Access the reasoning process
if hasattr(response, 'thinking'):
    for step in response.thinking.steps:
        print(f"Step: {step.description}")
        print(f"Confidence: {step.confidence}")
```

### Logic-LLM Integration

Enhanced reasoning with logical constraints:

```python
config = AgentConfig(
    enable_logical_reasoning=True,
    logical_domains=["mathematics", "programming"],
    constraint_timeout=30
)
```

## Error Handling

### Graceful Degradation

Agents handle errors gracefully:

```python
try:
    response = await agent.process_message("Complex query")
except AgentError as e:
    print(f"Agent error: {e}")
    # Agent continues functioning with reduced capabilities
```

### Retry Logic

Built-in retry mechanisms:

```python
config = AgentConfig(
    max_retries=3,
    retry_delay=1.0,
    exponential_backoff=True
)
```

## Performance Optimization

### Async Processing

All agents are async-first:

```python
import asyncio

async def process_multiple():
    tasks = [
        agent.process_message(f"Query {i}")
        for i in range(10)
    ]
    responses = await asyncio.gather(*tasks)
    return responses
```

### Caching

Enable response caching:

```python
config = AgentConfig(
    enable_caching=True,
    cache_ttl=3600,  # 1 hour cache
    cache_key_strategy="content_hash"
)
```

## Monitoring and Observability

### Built-in Tracing

Agents automatically trace operations:

```python
from ragents.observability import get_tracer

tracer = get_tracer()

# Tracing is automatic, but you can add custom spans
with tracer.trace("custom_operation") as span:
    response = await agent.process_message("Query")
    span.add_tag("response_length", len(response))
```

### Metrics Collection

Monitor agent performance:

```python
from ragents.observability import MetricsCollector

metrics = MetricsCollector()

# Metrics are collected automatically
avg_response_time = metrics.get_average_response_time()
success_rate = metrics.get_success_rate()
```

## Best Practices

### Choosing the Right Agent

- **Decision Tree Agent** - For structured, rule-based scenarios
- **ReAct Agent** - For exploratory, research-heavy tasks
- **Graph Planner Agent** - For complex, multi-step planning
- **LangGraph Agents** - For workflow orchestration

### Configuration Tips

1. **Start simple** - Begin with basic configuration
2. **Enable memory** - For conversational scenarios
3. **Use tools selectively** - Enable only needed tools
4. **Monitor performance** - Track response times and success rates
5. **Tune temperature** - Lower for deterministic, higher for creative

### Error Handling

1. **Implement try/catch** - Handle agent errors gracefully
2. **Set timeouts** - Prevent hanging operations
3. **Use retries** - For transient failures
4. **Log errors** - For debugging and monitoring

## Next Steps

- **[Decision Tree Agent](decision-tree.md)** - Deep dive into decision trees
- **[ReAct Agent](react.md)** - Learn the ReAct pattern
- **[Graph Planner](graph-planner.md)** - Understand graph-based planning
- **[Custom Agents](custom.md)** - Build your own agent types