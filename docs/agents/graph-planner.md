# Graph Planner Agent

The Graph Planner Agent uses graph-based planning with depth-first search for complex task orchestration. It excels at multi-step planning, dependency management, and resource optimization.

## Overview

The Graph Planner Agent models tasks as directed graphs where:
- **Nodes** represent individual tasks or sub-goals
- **Edges** represent dependencies between tasks
- **Attributes** store task metadata (priority, resources, duration)
- **Execution** follows optimal paths through the graph

This approach enables sophisticated planning for complex, multi-step workflows.

## Key Features

- **Graph-Based Modeling** - Tasks represented as connected graphs
- **Dependency Resolution** - Automatic handling of task dependencies
- **DFS Planning** - Depth-first search for optimal execution paths
- **Resource Management** - Track and allocate resources across tasks
- **Parallel Execution** - Execute independent tasks concurrently
- **Dynamic Replanning** - Adapt plans based on execution results

## Basic Usage

```python
import asyncio
from ragents import GraphPlannerAgent, AgentConfig
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env

async def basic_graph_planner():
    # Initialize LLM client
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Create graph planner agent
    config = AgentConfig(
        name="Project Planner",
        enable_tools=True,
        enable_reasoning=True,
        max_planning_depth=10
    )

    agent = GraphPlannerAgent(
        config=config,
        llm_client=llm_client
    )

    # Complex planning task
    response = await agent.process_message(
        "Plan a website redesign project including research, design, "
        "development, testing, and deployment phases."
    )

    print("Execution Plan:")
    print(response.content)

    # Access the generated plan graph
    plan_graph = response.plan_graph
    for task in plan_graph.execution_order:
        print(f"Task: {task.name}")
        print(f"Dependencies: {task.dependencies}")
        print(f"Estimated Duration: {task.duration}")
        print("---")

asyncio.run(basic_graph_planner())
```

## Graph Planning Concepts

### Task Nodes

Individual tasks in the plan:

```python
from ragents.agents.graph_planner import TaskNode

task = TaskNode(
    id="user_research",
    name="Conduct User Research",
    description="Survey users and analyze requirements",

    # Dependencies
    dependencies=[],  # No dependencies for this initial task

    # Resources
    required_resources=["researcher", "survey_tool"],
    estimated_duration=5,  # days
    priority=1,  # 1 = highest

    # Execution details
    tools=["survey_tool", "analysis_tool"],
    success_criteria=["100+ responses", "analysis_complete"],

    # Metadata
    task_type="research",
    assignee="research_team",
    deadline="2024-02-01"
)
```

### Task Dependencies

Define relationships between tasks:

```python
# Sequential dependencies
design_task = TaskNode(
    id="ui_design",
    name="UI Design",
    dependencies=["user_research"],  # Must complete research first
    dependency_type="sequential"
)

# Parallel dependencies (can start when any dependency is done)
testing_tasks = [
    TaskNode(
        id="unit_testing",
        name="Unit Testing",
        dependencies=["development"],
        dependency_type="parallel"
    ),
    TaskNode(
        id="integration_testing",
        name="Integration Testing",
        dependencies=["development"],
        dependency_type="parallel"
    )
]

# Conditional dependencies
deployment_task = TaskNode(
    id="deployment",
    name="Deploy to Production",
    dependencies=["unit_testing", "integration_testing"],
    dependency_type="all_complete",  # Wait for all testing to finish
    conditions=["all_tests_pass", "security_approved"]
)
```

### Graph Construction

Build task graphs programmatically:

```python
from ragents.agents.graph_planner import TaskGraph

# Create task graph
graph = TaskGraph()

# Add tasks
graph.add_task(user_research_task)
graph.add_task(design_task)
graph.add_task(development_task)
graph.add_task(testing_task)
graph.add_task(deployment_task)

# Add dependencies
graph.add_dependency("user_research", "ui_design")
graph.add_dependency("ui_design", "development")
graph.add_dependency("development", "unit_testing")
graph.add_dependency("development", "integration_testing")
graph.add_dependency(["unit_testing", "integration_testing"], "deployment")

# Validate graph
if graph.validate():
    print("Graph is valid - no circular dependencies")
else:
    print("Graph has issues:", graph.get_validation_errors())
```

## Advanced Planning

### Resource-Aware Planning

Plan with resource constraints:

```python
from ragents.agents.graph_planner import ResourceManager

# Define available resources
resources = ResourceManager()
resources.add_resource("developer", capacity=2, skills=["python", "javascript"])
resources.add_resource("designer", capacity=1, skills=["ui", "ux"])
resources.add_resource("tester", capacity=1, skills=["testing", "qa"])

# Create resource-aware agent
config = AgentConfig(
    name="Resource-Aware Planner",
    resource_manager=resources,
    enable_resource_optimization=True,
    resource_allocation_strategy="balanced"
)

agent = GraphPlannerAgent(config=config, llm_client=llm_client)
```

### Multi-Objective Optimization

Optimize for multiple goals:

```python
config = AgentConfig(
    name="Multi-Objective Planner",
    optimization_objectives=[
        {"name": "minimize_duration", "weight": 0.4},
        {"name": "minimize_cost", "weight": 0.3},
        {"name": "maximize_quality", "weight": 0.3}
    ],
    optimization_algorithm="pareto_frontier"
)
```

### Dynamic Replanning

Adapt plans during execution:

```python
# Enable dynamic replanning
config = AgentConfig(
    name="Adaptive Planner",
    enable_dynamic_replanning=True,
    replan_triggers=["task_failure", "resource_unavailable", "deadline_change"],
    replan_strategy="minimal_change"  # or "full_replan"
)

# Handle execution updates
async def handle_task_completion(task_id: str, result: dict):
    if result["status"] == "failed":
        # Trigger replanning
        new_plan = await agent.replan(
            failed_task=task_id,
            failure_reason=result["error"],
            remaining_resources=result["available_resources"]
        )
        print(f"Replanned execution path: {new_plan.execution_order}")
```

## Execution Strategies

### Sequential Execution

Execute tasks one after another:

```python
config = AgentConfig(
    name="Sequential Planner",
    execution_strategy="sequential",
    wait_for_completion=True,
    progress_tracking=True
)
```

### Parallel Execution

Execute independent tasks concurrently:

```python
config = AgentConfig(
    name="Parallel Planner",
    execution_strategy="parallel",
    max_concurrent_tasks=3,
    load_balancing="round_robin"
)

# Execute plan with parallel tasks
execution_result = await agent.execute_plan(
    plan_graph,
    monitor_progress=True,
    handle_failures="isolate"  # Don't let one failure stop everything
)
```

### Hybrid Execution

Combine sequential and parallel strategies:

```python
config = AgentConfig(
    name="Hybrid Planner",
    execution_strategy="hybrid",
    parallel_task_groups=True,  # Group independent tasks for parallel execution
    sequential_milestones=True,  # Major milestones must complete sequentially
    dependency_optimization=True
)
```

## Planning Algorithms

### Depth-First Search (DFS)

Default planning algorithm:

```python
config = AgentConfig(
    name="DFS Planner",
    planning_algorithm="dfs",
    search_depth_limit=15,
    backtrack_on_failure=True,
    path_optimization=True
)
```

### Critical Path Method (CPM)

Optimize for project duration:

```python
config = AgentConfig(
    name="CPM Planner",
    planning_algorithm="critical_path",
    identify_critical_tasks=True,
    slack_time_analysis=True,
    schedule_optimization=True
)
```

### A* Search

Goal-directed planning:

```python
config = AgentConfig(
    name="A* Planner",
    planning_algorithm="a_star",
    heuristic_function="estimated_completion_time",
    goal_state_definition="all_tasks_complete",
    admissible_heuristic=True
)
```

## Integration with Tools

### Planning Tools

Specialized tools for planning tasks:

```python
from ragents.tools import tool

@tool(name="project_estimator", description="Estimate project duration and resources")
async def estimate_project(tasks: list, complexity: str) -> dict:
    """Estimate project parameters based on tasks and complexity."""
    # Your estimation logic
    return {
        "estimated_duration": 30,  # days
        "required_resources": ["developers:2", "designers:1"],
        "confidence": 0.8
    }

@tool(name="dependency_analyzer", description="Analyze task dependencies")
async def analyze_dependencies(task_list: list) -> dict:
    """Identify potential dependencies between tasks."""
    # Dependency analysis logic
    return {
        "dependencies": [("task_a", "task_b")],
        "circular_dependencies": [],
        "critical_path": ["task_a", "task_c", "task_d"]
    }
```

### External System Integration

Integrate with project management tools:

```python
@tool(name="jira_integration", description="Sync with Jira")
async def sync_with_jira(plan_graph: dict) -> dict:
    """Sync planning graph with Jira project."""
    # Jira API integration
    return {"jira_project_id": "PROJ-123", "tickets_created": 15}

@tool(name="calendar_integration", description="Schedule tasks in calendar")
async def schedule_tasks(tasks: list, calendar_id: str) -> dict:
    """Schedule tasks in external calendar system."""
    # Calendar API integration
    return {"scheduled_events": 10, "calendar_id": calendar_id}
```

## Monitoring and Progress Tracking

### Execution Monitoring

Track plan execution in real-time:

```python
from ragents.agents.graph_planner import ExecutionMonitor

monitor = ExecutionMonitor(agent)

# Start monitoring
await monitor.start_monitoring(plan_graph)

# Get progress updates
progress = await monitor.get_progress()
print(f"Completion: {progress.completion_percentage}%")
print(f"Current tasks: {progress.active_tasks}")
print(f"Completed: {progress.completed_tasks}")
print(f"Remaining: {progress.remaining_tasks}")

# Handle events
@monitor.on_task_start
async def handle_task_start(task_id: str):
    print(f"Started task: {task_id}")

@monitor.on_task_complete
async def handle_task_complete(task_id: str, result: dict):
    print(f"Completed task: {task_id}")
    if result["status"] == "success":
        print(f"Result: {result['output']}")
```

### Performance Analytics

Analyze planning and execution performance:

```python
from ragents.observability import PlanningMetrics

metrics = PlanningMetrics(agent)

# Get planning statistics
stats = metrics.get_planning_stats()
print(f"Average planning time: {stats.avg_planning_time}")
print(f"Plan accuracy: {stats.plan_accuracy}")
print(f"Resource utilization: {stats.resource_utilization}")

# Get execution statistics
exec_stats = metrics.get_execution_stats()
print(f"On-time completion rate: {exec_stats.on_time_rate}")
print(f"Average execution efficiency: {exec_stats.efficiency}")
print(f"Replanning frequency: {exec_stats.replan_frequency}")
```

## Best Practices

### Graph Design

1. **Keep graphs manageable** - Break large projects into sub-graphs
2. **Define clear dependencies** - Avoid ambiguous relationships
3. **Use meaningful task names** - For better understanding and debugging
4. **Include buffer time** - Account for uncertainties
5. **Plan for failures** - Include alternative paths

### Resource Management

1. **Model resources accurately** - Reflect real-world constraints
2. **Plan for resource conflicts** - Identify potential bottlenecks
3. **Include skill requirements** - Match tasks to capable resources
4. **Account for availability** - Consider schedules and time zones
5. **Monitor resource utilization** - Optimize allocation over time

### Execution

1. **Monitor progress actively** - Track execution against plan
2. **Handle failures gracefully** - Have contingency plans
3. **Communicate updates** - Keep stakeholders informed
4. **Document decisions** - Record why plans changed
5. **Learn from execution** - Improve future planning

## Example Use Cases

### Software Development Project

```python
dev_project_config = AgentConfig(
    name="Software Project Planner",
    planning_algorithm="critical_path",
    execution_strategy="hybrid",
    tools=["jira_integration", "git_integration", "ci_cd_tools"],
    resource_types=["developer", "tester", "devops"],
    quality_gates=["code_review", "testing", "security_scan"]
)
```

### Event Planning

```python
event_config = AgentConfig(
    name="Event Planner",
    planning_algorithm="dfs",
    execution_strategy="parallel",
    tools=["venue_booking", "catering", "vendor_management"],
    constraint_types=["budget", "timeline", "capacity"],
    contingency_planning=True
)
```

### Manufacturing Process

```python
manufacturing_config = AgentConfig(
    name="Manufacturing Planner",
    planning_algorithm="critical_path",
    execution_strategy="sequential",
    tools=["inventory_check", "equipment_scheduler", "quality_control"],
    optimization_objectives=["minimize_cost", "maximize_throughput"],
    resource_constraints=["machines", "materials", "workers"]
)
```

## Troubleshooting

### Common Issues

**Circular Dependencies:**
```python
# Detect and resolve circular dependencies
validation_result = graph.validate()
if not validation_result.valid:
    for cycle in validation_result.cycles:
        print(f"Circular dependency: {' -> '.join(cycle)}")
        # Suggest resolution
        suggested_fix = graph.suggest_cycle_resolution(cycle)
        print(f"Suggested fix: {suggested_fix}")
```

**Resource Conflicts:**
```python
# Identify resource conflicts
conflicts = resource_manager.find_conflicts(plan_graph)
for conflict in conflicts:
    print(f"Resource conflict: {conflict.resource} at {conflict.time}")
    print(f"Conflicting tasks: {conflict.tasks}")
    # Suggest resolution
    resolution = resource_manager.suggest_resolution(conflict)
    print(f"Suggested resolution: {resolution}")
```

**Plan Optimization:**
```python
# Optimize plan for specific objectives
optimizer = PlanOptimizer(objectives=["minimize_duration", "minimize_cost"])
optimized_plan = optimizer.optimize(original_plan)

print(f"Duration improvement: {optimized_plan.duration_reduction}")
print(f"Cost savings: {optimized_plan.cost_reduction}")
print(f"Modified tasks: {optimized_plan.changes}")
```

## Next Steps

- **[Custom Agents](custom.md)** - Learn to build specialized agent types
- **[RAG Integration](../rag/overview.md)** - Enhance planning with knowledge
- **[Advanced Features](../advanced/observability.md)** - Monitor planning performance
- **[Deployment](../deployment/kubernetes.md)** - Deploy planning agents at scale