# Decision Tree Agent

The Decision Tree Agent uses configurable decision trees for sophisticated reasoning and action selection. It's ideal for scenarios requiring structured, rule-based decision making.

## Overview

The Decision Tree Agent processes user messages by traversing a configurable decision tree, where each node can:
- Evaluate conditions based on context
- Execute actions like tool calls or responses
- Branch to different paths based on results
- Maintain state across the decision process

## Key Features

- **Configurable Decision Nodes** - Define custom decision logic
- **Action Nodes** - Execute specific actions or tool calls
- **Context-Aware Decisions** - Access to conversation history and user intent
- **Tool Integration** - Seamless tool usage within decision paths
- **Memory Management** - Remember decisions and outcomes
- **Fallback Handling** - Graceful degradation for unexpected scenarios

## Basic Usage

### Simple Decision Tree

```python
import asyncio
from ragents import DecisionTreeAgent, AgentConfig
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env

async def basic_decision_tree():
    # Initialize LLM client
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Create agent with default decision tree
    config = AgentConfig(
        name="Decision Assistant",
        enable_tools=True,
        enable_memory=True
    )

    agent = DecisionTreeAgent(
        config=config,
        llm_client=llm_client
    )

    # The agent uses a default decision tree
    response = await agent.process_message("What's 25 * 34?")
    print(response)  # Will use calculator tool

asyncio.run(basic_decision_tree())
```

### Custom Decision Tree

```python
from ragents.agents.decision_tree import DecisionNode, ActionNode, ConditionType

# Define custom decision tree structure
decision_tree = [
    DecisionNode(
        id="intent_classifier",
        condition="classify_user_intent(message)",
        condition_type=ConditionType.FUNCTION,
        branches={
            "question": "handle_question",
            "calculation": "handle_calculation",
            "greeting": "handle_greeting",
            "default": "general_response"
        }
    ),

    ActionNode(
        id="handle_question",
        action="search_knowledge",
        tools=["rag_search"],
        next_node="formulate_response"
    ),

    ActionNode(
        id="handle_calculation",
        action="calculate",
        tools=["calculator"],
        next_node="formulate_response"
    ),

    ActionNode(
        id="handle_greeting",
        action="friendly_greeting",
        response_template="Hello! How can I help you today?",
        next_node=None  # End conversation
    ),

    ActionNode(
        id="formulate_response",
        action="generate_response",
        use_context=True,
        confidence_threshold=0.8
    )
]

# Create agent with custom tree
config = AgentConfig(
    name="Custom Decision Agent",
    decision_tree=decision_tree,
    enable_reasoning=True
)

agent = DecisionTreeAgent(config=config, llm_client=llm_client)
```

## Decision Node Types

### Condition Nodes

Evaluate conditions and branch accordingly:

```python
# Simple condition based on message content
DecisionNode(
    id="check_math",
    condition="contains_math_expression(message)",
    branches={"true": "calculator_node", "false": "general_node"}
)

# Complex condition with multiple factors
DecisionNode(
    id="priority_check",
    condition="user.priority == 'high' and message.urgency > 7",
    condition_type=ConditionType.EXPRESSION,
    branches={"true": "urgent_response", "false": "normal_response"}
)

# Function-based condition
DecisionNode(
    id="intent_check",
    condition="analyze_intent",  # Calls custom function
    condition_type=ConditionType.FUNCTION,
    branches={
        "question": "qa_handler",
        "request": "task_handler",
        "complaint": "support_handler"
    }
)
```

### Action Nodes

Execute specific actions:

```python
# Tool execution node
ActionNode(
    id="calculator_action",
    action="calculate",
    tools=["calculator"],
    parameters={"expression": "extracted_expression"},
    next_node="format_result"
)

# RAG search node
ActionNode(
    id="search_knowledge",
    action="rag_search",
    tools=["rag_search"],
    parameters={"query": "processed_query"},
    rerank_results=True,
    next_node="generate_answer"
)

# Response generation node
ActionNode(
    id="generate_response",
    action="llm_response",
    use_context=True,
    system_prompt="You are a helpful assistant.",
    temperature=0.1,
    max_tokens=500
)
```

## Advanced Features

### Dynamic Tree Modification

Modify the decision tree at runtime:

```python
# Add new decision path
agent.add_decision_path(
    condition="user.expertise == 'expert'",
    action="detailed_technical_response"
)

# Update existing node
agent.update_node(
    node_id="intent_classifier",
    new_condition="enhanced_intent_analysis(message, context)"
)

# Remove node
agent.remove_node("outdated_node")
```

### Context-Aware Decisions

Access conversation context in decisions:

```python
DecisionNode(
    id="context_aware",
    condition="previous_topic == 'python' and user.skill_level > 5",
    context_variables=["previous_topic", "user.skill_level"],
    branches={"true": "advanced_python", "false": "basic_python"}
)
```

### Parallel Decision Paths

Execute multiple decision paths:

```python
ActionNode(
    id="parallel_analysis",
    action="parallel_processing",
    parallel_actions=[
        "sentiment_analysis",
        "intent_classification",
        "entity_extraction"
    ],
    merge_strategy="weighted_average",
    next_node="combined_response"
)
```

## Custom Condition Functions

Define custom logic for decision nodes:

```python
from ragents.agents.decision_tree import register_condition_function

@register_condition_function
async def analyze_user_intent(message: str, context: dict) -> str:
    """Analyze user intent with custom logic."""
    # Your custom intent analysis
    if "calculate" in message.lower():
        return "calculation"
    elif "?" in message:
        return "question"
    elif any(greeting in message.lower() for greeting in ["hello", "hi", "hey"]):
        return "greeting"
    else:
        return "general"

@register_condition_function
async def check_user_expertise(message: str, context: dict) -> bool:
    """Check if user is an expert based on conversation history."""
    history = context.get("conversation_history", [])
    technical_terms = ["API", "algorithm", "optimization", "scalability"]

    technical_count = sum(
        1 for msg in history[-5:]  # Check last 5 messages
        for term in technical_terms
        if term.lower() in msg.lower()
    )

    return technical_count >= 3
```

## Tool Integration

### Tool Selection in Decision Trees

```python
ActionNode(
    id="smart_tool_selection",
    action="select_and_execute_tool",
    tool_selection_strategy="best_match",
    available_tools=["calculator", "rag_search", "web_search"],
    tool_confidence_threshold=0.7,
    fallback_tool="rag_search"
)
```

### Tool Chaining

Chain multiple tools in sequence:

```python
ActionNode(
    id="tool_chain",
    action="execute_tool_chain",
    tool_chain=[
        {"tool": "rag_search", "output": "search_results"},
        {"tool": "summarizer", "input": "search_results", "output": "summary"},
        {"tool": "fact_checker", "input": "summary", "output": "verified_facts"}
    ],
    final_output="verified_facts"
)
```

## Error Handling

### Fallback Nodes

Define fallback behavior for errors:

```python
DecisionNode(
    id="main_classifier",
    condition="classify_intent(message)",
    branches={
        "question": "qa_handler",
        "calculation": "calc_handler",
        "error": "error_handler",      # Handle classification errors
        "default": "general_handler"   # Default fallback
    },
    error_node="error_handler"  # Global error fallback
)

ActionNode(
    id="error_handler",
    action="handle_error",
    response_template="I apologize, but I encountered an issue. Let me try a different approach.",
    retry_with_node="general_handler",
    log_error=True
)
```

### Retry Logic

Implement retry mechanisms:

```python
ActionNode(
    id="resilient_action",
    action="api_call",
    max_retries=3,
    retry_delay=1.0,
    exponential_backoff=True,
    retry_conditions=["timeout", "rate_limit"],
    fallback_action="alternative_approach"
)
```

## Performance Optimization

### Tree Caching

Cache decision tree evaluations:

```python
config = AgentConfig(
    name="Optimized Agent",
    enable_tree_caching=True,
    cache_decision_results=True,
    cache_ttl=300,  # 5 minutes
    cache_key_strategy="content_hash"
)
```

### Lazy Evaluation

Evaluate nodes only when needed:

```python
DecisionNode(
    id="expensive_analysis",
    condition="deep_analysis(message)",
    lazy_evaluation=True,  # Only evaluate if path is taken
    cache_result=True,     # Cache expensive operations
    timeout=30             # Timeout for expensive operations
)
```

## Monitoring and Debugging

### Decision Tree Visualization

Visualize the decision tree execution:

```python
# Enable decision tracking
config = AgentConfig(
    enable_decision_tracking=True,
    track_execution_path=True,
    log_decision_rationale=True
)

# Get execution trace
response = await agent.process_message("What's the weather?")
execution_trace = agent.get_last_execution_trace()

for step in execution_trace.steps:
    print(f"Node: {step.node_id}")
    print(f"Condition: {step.condition}")
    print(f"Result: {step.result}")
    print(f"Next: {step.next_node}")
    print("---")
```

### Performance Metrics

Monitor decision tree performance:

```python
from ragents.observability import DecisionTreeMetrics

metrics = DecisionTreeMetrics(agent)

# Get execution statistics
stats = metrics.get_execution_stats()
print(f"Average decision time: {stats.avg_decision_time}")
print(f"Most used path: {stats.most_common_path}")
print(f"Success rate: {stats.success_rate}")

# Get node-specific metrics
node_stats = metrics.get_node_performance("intent_classifier")
print(f"Node accuracy: {node_stats.accuracy}")
print(f"Execution count: {node_stats.execution_count}")
```

## Best Practices

### Tree Design

1. **Keep trees shallow** - Avoid deep nesting for better performance
2. **Use meaningful node IDs** - For easier debugging and maintenance
3. **Define clear conditions** - Avoid ambiguous decision logic
4. **Include fallback paths** - Handle unexpected scenarios gracefully
5. **Test decision paths** - Verify all branches work correctly

### Condition Design

1. **Make conditions deterministic** - Avoid random or time-dependent logic
2. **Use caching for expensive operations** - Cache complex condition evaluations
3. **Handle edge cases** - Consider null, empty, or malformed inputs
4. **Document condition logic** - Make decision rationale clear

### Performance

1. **Profile decision trees** - Identify bottlenecks in execution
2. **Use lazy evaluation** - For expensive or rarely-used conditions
3. **Cache frequently used results** - Reduce redundant computations
4. **Monitor execution times** - Set reasonable timeouts

### Maintenance

1. **Version your trees** - Track changes to decision logic
2. **Test thoroughly** - Verify behavior with edge cases
3. **Monitor in production** - Track success rates and errors
4. **Document decision rationale** - Explain why decisions are made

## Example Use Cases

### Customer Support Agent

```python
support_tree = [
    DecisionNode(
        id="categorize_issue",
        condition="categorize_support_issue(message)",
        branches={
            "technical": "technical_support",
            "billing": "billing_support",
            "general": "general_support"
        }
    ),
    ActionNode(
        id="technical_support",
        action="search_technical_docs",
        tools=["rag_search"],
        knowledge_base="technical_docs",
        escalation_threshold=0.3
    )
]
```

### Content Moderation Agent

```python
moderation_tree = [
    DecisionNode(
        id="safety_check",
        condition="content_safety_score(message) < 0.7",
        branches={"true": "flag_content", "false": "approve_content"}
    ),
    ActionNode(
        id="flag_content",
        action="moderate_content",
        severity_levels=["low", "medium", "high"],
        auto_action_threshold=0.9
    )
]
```

## Next Steps

- **[ReAct Agent](react.md)** - Learn about the ReAct pattern
- **[Graph Planner](graph-planner.md)** - Explore graph-based planning
- **[Custom Agents](custom.md)** - Build your own agent types
- **[Tools Integration](../api/tools.md)** - Learn about tool development