# ReAct Agent

The ReAct Agent implements the Reasoning + Acting pattern, combining step-by-step reasoning with dynamic tool usage. It excels at multi-step problem solving and exploratory tasks.

## Overview

ReAct (Reasoning + Acting) is a paradigm where the agent:
1. **Thinks** - Reasons about the current situation
2. **Acts** - Executes tools or actions based on reasoning
3. **Observes** - Processes the results of actions
4. **Reflects** - Updates understanding and plans next steps

This cycle continues until the task is completed or a stopping condition is met.

## Key Features

- **Iterative Reasoning** - Step-by-step problem breakdown
- **Dynamic Tool Selection** - Chooses tools based on current context
- **Self-Correction** - Adjusts approach based on results
- **Transparent Process** - Shows reasoning steps to users
- **Flexible Execution** - Adapts to unexpected situations
- **Memory Integration** - Learns from previous interactions

## Basic Usage

### Simple ReAct Agent

```python
import asyncio
from ragents import ReActAgent, AgentConfig
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env

async def basic_react():
    # Initialize LLM client
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Create ReAct agent
    config = AgentConfig(
        name="ReAct Assistant",
        enable_tools=True,
        enable_reasoning=True,
        max_iterations=5
    )

    agent = ReActAgent(
        config=config,
        llm_client=llm_client
    )

    # Multi-step reasoning example
    response = await agent.process_message(
        "What's the population of Tokyo and how does it compare to New York?"
    )

    print(response.content)
    print("\nReasoning Steps:")
    for i, step in enumerate(response.reasoning_steps):
        print(f"{i+1}. {step.thought}")
        if step.action:
            print(f"   Action: {step.action}")
        if step.observation:
            print(f"   Result: {step.observation}")

asyncio.run(basic_react())
```

### ReAct with RAG

```python
from ragents import RAGEngine, RAGConfig

async def react_with_rag():
    # Set up RAG engine
    rag_config = RAGConfig.from_env()
    rag_engine = RAGEngine(rag_config, llm_client)

    # Add documents
    await rag_engine.add_document("research_papers.pdf")

    # Create ReAct agent with RAG
    config = AgentConfig(
        name="Research Assistant",
        enable_rag=True,
        enable_reasoning=True,
        max_iterations=10
    )

    agent = ReActAgent(
        config=config,
        llm_client=llm_client,
        rag_engine=rag_engine
    )

    # Complex research query
    response = await agent.process_message(
        "Analyze the methodology used in the climate change papers and "
        "suggest improvements based on recent statistical techniques."
    )

    print(response.content)
```

## ReAct Cycle

### 1. Thinking Phase

The agent analyzes the situation and plans actions:

```python
# Example thinking process
{
    "thought": "The user wants population data for Tokyo and NYC. I need to search for current population figures for both cities.",
    "reasoning": "I should look up each city separately to get accurate, recent data.",
    "plan": ["search Tokyo population", "search NYC population", "compare the numbers"],
    "confidence": 0.9
}
```

### 2. Acting Phase

The agent selects and executes appropriate tools:

```python
# Tool selection and execution
{
    "action": "rag_search",
    "parameters": {
        "query": "Tokyo population 2024 current",
        "top_k": 3
    },
    "reasoning": "Search knowledge base for Tokyo population data"
}
```

### 3. Observation Phase

The agent processes tool results:

```python
# Processing results
{
    "observation": "Tokyo metropolitan area has approximately 37.4 million people as of 2024",
    "confidence": 0.85,
    "source_quality": "high",
    "next_action_needed": True
}
```

### 4. Reflection Phase

The agent updates its understanding:

```python
# Reflection and planning
{
    "reflection": "Got Tokyo data successfully. Now need NYC data for comparison.",
    "updated_context": "Tokyo: 37.4M people",
    "next_steps": ["search NYC population", "calculate comparison"],
    "progress": 0.5
}
```

## Advanced Configuration

### Iteration Control

```python
config = AgentConfig(
    name="Controlled ReAct",

    # Iteration limits
    max_iterations=10,           # Maximum reasoning cycles
    min_iterations=2,            # Minimum cycles before stopping
    early_stop_threshold=0.95,   # Stop if confidence is high enough

    # Reasoning depth
    reasoning_depth=3,           # How detailed the reasoning should be
    reflection_frequency=2,      # Reflect every N iterations

    # Error handling
    max_consecutive_errors=3,    # Stop after too many tool errors
    error_recovery_strategy="backtrack"  # How to handle errors
)
```

### Custom Stopping Conditions

```python
from ragents.agents.react import StoppingCondition

def custom_stopping_condition(iteration: int, context: dict) -> bool:
    """Stop if we have sufficient information."""
    required_info = context.get("required_information", [])
    gathered_info = context.get("gathered_information", [])

    return len(gathered_info) >= len(required_info) * 0.8

config = AgentConfig(
    name="Custom Stop ReAct",
    stopping_conditions=[
        StoppingCondition.MAX_ITERATIONS,
        StoppingCondition.HIGH_CONFIDENCE,
        custom_stopping_condition
    ]
)
```

## Tool Integration

### Dynamic Tool Selection

The ReAct agent automatically selects appropriate tools:

```python
# Agent reasoning for tool selection
{
    "available_tools": ["calculator", "rag_search", "web_search", "code_executor"],
    "context": "User asked about statistical analysis",
    "selected_tool": "calculator",
    "selection_reasoning": "Need to perform mathematical calculations for statistics",
    "confidence": 0.8
}
```

### Tool Chaining

ReAct agents can chain multiple tools:

```python
# Example tool chain execution
response = await agent.process_message(
    "Calculate the compound interest on $10,000 at 5% for 10 years, "
    "then find investments with similar returns."
)

# Agent might execute:
# 1. calculator: compound interest formula
# 2. rag_search: investment options with 5% returns
# 3. calculator: compare different investment scenarios
```

### Custom Tool Integration

```python
from ragents.tools import tool

@tool(name="data_analyzer", description="Analyze datasets")
async def analyze_data(dataset_path: str, analysis_type: str) -> str:
    """Custom data analysis tool."""
    # Your analysis logic
    return f"Analysis complete: {analysis_type} on {dataset_path}"

# Tool is automatically available to ReAct agents
```

## Error Handling and Recovery

### Error Recovery Strategies

```python
config = AgentConfig(
    name="Resilient ReAct",
    error_recovery_strategy="adaptive",  # Options: retry, backtrack, alternative, adaptive
    max_retries_per_tool=3,
    retry_delay=1.0,
    fallback_tools={
        "rag_search": "web_search",      # If RAG fails, try web search
        "calculator": "code_executor"     # If calculator fails, try code execution
    }
)
```

### Graceful Degradation

```python
# Handle partial failures gracefully
async def robust_react_processing():
    try:
        response = await agent.process_message("Complex multi-step query")
    except PartialProcessingError as e:
        # Agent completed some steps but not all
        print(f"Partial result: {e.partial_result}")
        print(f"Failed at step: {e.failed_step}")
        print(f"Reason: {e.error_reason}")

        # Continue with what we have
        return e.partial_result
```

## Reasoning Patterns

### Chain of Thought

ReAct supports explicit chain-of-thought reasoning:

```python
config = AgentConfig(
    name="CoT ReAct",
    reasoning_style="chain_of_thought",
    show_intermediate_steps=True,
    reasoning_detail_level="verbose"
)
```

### Problem Decomposition

Break complex problems into smaller parts:

```python
# Example decomposition
{
    "original_problem": "Plan a vacation to Japan",
    "sub_problems": [
        "Research visa requirements",
        "Find flight options",
        "Book accommodations",
        "Plan itinerary",
        "Calculate budget"
    ],
    "execution_order": "sequential",  # or "parallel"
    "dependencies": {
        "book_accommodations": ["research_visa"],
        "plan_itinerary": ["book_accommodations"]
    }
}
```

## Memory and Learning

### Experience Learning

ReAct agents learn from past interactions:

```python
config = AgentConfig(
    name="Learning ReAct",
    enable_experience_learning=True,
    memory_consolidation_interval=10,  # Consolidate every 10 interactions
    learning_rate=0.1,
    experience_buffer_size=1000
)
```

### Pattern Recognition

Recognize and reuse successful patterns:

```python
# Agent recognizes similar problems
{
    "current_problem": "Calculate investment returns",
    "similar_past_problems": [
        "Calculate loan interest",
        "Analyze stock performance"
    ],
    "successful_pattern": "use_calculator_then_search_context",
    "confidence_in_pattern": 0.85
}
```

## Monitoring and Debugging

### Reasoning Trace

Get detailed reasoning traces:

```python
response = await agent.process_message("Complex query")

# Access detailed trace
trace = response.reasoning_trace
for step in trace.steps:
    print(f"Iteration {step.iteration}:")
    print(f"  Thought: {step.thought}")
    print(f"  Action: {step.action}")
    print(f"  Tool: {step.tool_used}")
    print(f"  Result: {step.observation}")
    print(f"  Confidence: {step.confidence}")
    print(f"  Time: {step.duration}ms")
```

### Performance Metrics

Monitor ReAct performance:

```python
from ragents.observability import ReActMetrics

metrics = ReActMetrics(agent)

# Get execution statistics
stats = metrics.get_performance_stats()
print(f"Average iterations: {stats.avg_iterations}")
print(f"Success rate: {stats.success_rate}")
print(f"Tool usage distribution: {stats.tool_usage}")
print(f"Error rate by iteration: {stats.error_rates}")
```

## Advanced Features

### Multi-Modal Reasoning

ReAct with vision and document processing:

```python
config = AgentConfig(
    name="Multi-Modal ReAct",
    enable_vision=True,
    enable_document_processing=True,
    supported_formats=["pdf", "image", "text"]
)

# Process images and documents
response = await agent.process_message(
    "Analyze this chart and explain the trends",
    attachments=["sales_chart.png"]
)
```

### Collaborative ReAct

Multiple ReAct agents working together:

```python
from ragents.agents.react import CollaborativeReAct

# Create specialist agents
data_agent = ReActAgent(config=data_specialist_config)
analysis_agent = ReActAgent(config=analysis_specialist_config)

# Collaborative processing
collaborative_agent = CollaborativeReAct(
    agents=[data_agent, analysis_agent],
    coordination_strategy="divide_and_conquer"
)

response = await collaborative_agent.process_message(
    "Analyze sales data and provide recommendations"
)
```

## Best Practices

### Query Design

1. **Be specific** - Clear, detailed queries get better results
2. **Provide context** - Include relevant background information
3. **Break down complex tasks** - Let the agent decompose naturally
4. **Set expectations** - Specify desired output format or depth

### Configuration

1. **Set appropriate iteration limits** - Balance thoroughness with efficiency
2. **Choose right stopping conditions** - Based on your use case
3. **Configure error handling** - Plan for tool failures
4. **Enable appropriate features** - Only what you need for performance

### Tool Management

1. **Provide quality tools** - Well-designed tools improve reasoning
2. **Document tool capabilities** - Clear descriptions help selection
3. **Handle tool errors gracefully** - Provide fallback options
4. **Monitor tool usage** - Identify popular and problematic tools

### Performance

1. **Monitor iteration counts** - High counts may indicate issues
2. **Track reasoning quality** - Ensure coherent thought processes
3. **Optimize tool response times** - Slow tools hurt the experience
4. **Cache expensive operations** - Reuse results when possible

## Example Use Cases

### Research Assistant

```python
research_config = AgentConfig(
    name="Research Assistant",
    max_iterations=15,
    enable_rag=True,
    tools=["rag_search", "web_search", "citation_formatter"],
    reasoning_style="scholarly",
    fact_checking=True
)
```

### Data Analysis Agent

```python
analysis_config = AgentConfig(
    name="Data Analyst",
    max_iterations=8,
    tools=["calculator", "code_executor", "data_visualizer"],
    reasoning_style="analytical",
    validation_required=True
)
```

### Creative Problem Solver

```python
creative_config = AgentConfig(
    name="Creative Solver",
    max_iterations=12,
    temperature=0.7,  # Higher for creativity
    tools=["brainstormer", "idea_evaluator", "research_tool"],
    reasoning_style="divergent",
    exploration_factor=0.8
)
```

## Troubleshooting

### Common Issues

**Infinite Loops:**
```python
# Prevent with proper stopping conditions
config = AgentConfig(
    max_iterations=10,
    loop_detection=True,
    early_stop_threshold=0.9
)
```

**Poor Tool Selection:**
```python
# Improve with better tool descriptions and examples
@tool(
    name="calculator",
    description="Performs mathematical calculations. Use for: arithmetic, algebra, statistics",
    examples=["Calculate 25 * 34", "Find average of [1,2,3,4,5]"]
)
```

**Inconsistent Reasoning:**
```python
# Stabilize with lower temperature and better prompts
config = AgentConfig(
    temperature=0.1,
    reasoning_style="structured",
    consistency_check=True
)
```

## Next Steps

- **[Graph Planner Agent](graph-planner.md)** - Learn about graph-based planning
- **[Custom Agents](custom.md)** - Build specialized agent types
- **[Tools Development](../api/tools.md)** - Create custom tools for ReAct
- **[Advanced Features](../advanced/logical-reasoning.md)** - Enhance reasoning capabilities