# Logical Reasoning

RAGents includes a powerful **Logical LLM module** based on [Logic-LLM](https://github.com/teacherpeterpan/Logic-LLM) that enhances reasoning capabilities while reducing token usage through symbolic reasoning and constraint satisfaction.

## Overview

The Logical Reasoning system combines neural language models with symbolic reasoning to:
- **Reduce Token Usage** - Use logical patterns to minimize LLM calls
- **Improve Accuracy** - Apply formal reasoning rules
- **Handle Constraints** - Solve problems with logical constraints
- **Enable Verification** - Validate reasoning steps symbolically

## Key Components

- **LogicalReasoner** - Core reasoning engine
- **ConstraintEngine** - Rule-based constraint satisfaction
- **SymbolicSolver** - Mathematical and logical problem solving
- **QueryClarifier** - Interactive query refinement
- **LogicPattern** - Pattern matching for logical structures

## Basic Usage

### Simple Logical Reasoning

```python
import asyncio
from ragents.logical_llm import LogicalReasoner
from ragents.llm.client import LLMClient
from ragents.config.environment import get_llm_config_from_env

async def basic_logical_reasoning():
    # Initialize components
    llm_config = get_llm_config_from_env()
    llm_client = LLMClient(llm_config)

    # Create logical reasoner
    reasoner = LogicalReasoner(llm_client)

    # Mathematical reasoning
    result = await reasoner.solve_logical_problem(
        "If all birds can fly, and penguins are birds, can penguins fly?"
    )

    print(f"Answer: {result.answer}")
    print(f"Reasoning: {result.reasoning_steps}")
    print(f"Confidence: {result.confidence}")
    print(f"Tokens used: {result.token_usage}")

asyncio.run(basic_logical_reasoning())
```

### Constraint-Based Problem Solving

```python
from ragents.logical_llm import ConstraintEngine

async def constraint_solving():
    constraint_engine = ConstraintEngine()

    # Define constraints
    constraints = [
        "x + y = 10",
        "x > y",
        "x, y are positive integers"
    ]

    # Solve constraint satisfaction problem
    solution = await constraint_engine.solve_constraints(
        constraints,
        variables=["x", "y"]
    )

    print(f"Solution: {solution.values}")
    print(f"Verification: {solution.verification}")
```

## Domain-Specific Reasoning

### Mathematical Reasoning

```python
from ragents.logical_llm import SymbolicSolver

async def mathematical_reasoning():
    solver = SymbolicSolver()

    # Algebraic problem solving
    math_problem = """
    A train travels 120 miles in 2 hours.
    If it increases its speed by 10 mph, how long will it take to travel 180 miles?
    """

    result = await solver.solve_math_problem(math_problem)

    print(f"Steps: {result.solution_steps}")
    print(f"Answer: {result.final_answer}")
    print(f"Verification: {result.symbolic_verification}")
```

### Logical Deduction

```python
async def logical_deduction():
    reasoner = LogicalReasoner(llm_client)

    # Define premises and query
    premises = [
        "All humans are mortal",
        "Socrates is human",
        "Mortality implies finite lifespan"
    ]

    query = "Does Socrates have a finite lifespan?"

    result = await reasoner.deduce(premises, query)

    print(f"Conclusion: {result.conclusion}")
    print(f"Logical chain: {result.deduction_chain}")
    print(f"Validity: {result.is_valid}")
```

## Query Clarification

### Interactive Clarification

```python
from ragents.logical_llm import QueryClarifier

async def interactive_clarification():
    clarifier = QueryClarifier(llm_client)

    # Ambiguous query
    query = "What's the best way to optimize performance?"

    # Analyze and clarify
    clarification = await clarifier.analyze_query(query)

    if clarification.needs_clarification:
        print("Query needs clarification:")
        for question in clarification.clarifying_questions:
            print(f"- {question}")

        # Simulate user responses
        user_responses = {
            "What type of performance?": "database query performance",
            "What system?": "PostgreSQL database",
            "What constraints?": "limited memory, high concurrency"
        }

        # Generate clarified query
        clarified_query = await clarifier.clarify_query(
            original_query=query,
            responses=user_responses
        )

        print(f"Clarified query: {clarified_query.refined_query}")
        print(f"Domain: {clarified_query.identified_domain}")
```

### Domain-Aware Analysis

```python
async def domain_aware_analysis():
    clarifier = QueryClarifier(llm_client)

    # Configure domain knowledge
    await clarifier.add_domain_knowledge(
        domain="database_optimization",
        keywords=["index", "query", "performance", "optimization"],
        concepts=["indexing", "query_planning", "caching"],
        constraints=["memory_limit", "response_time", "concurrency"]
    )

    query = "How to make database faster?"

    analysis = await clarifier.analyze_domain_query(query)

    print(f"Domain: {analysis.primary_domain}")
    print(f"Concepts: {analysis.relevant_concepts}")
    print(f"Constraints: {analysis.implicit_constraints}")
```

## Pattern Recognition

### Logic Pattern Matching

```python
from ragents.logical_llm import LogicPattern

async def pattern_matching():
    pattern_matcher = LogicPattern()

    # Define logical patterns
    patterns = [
        {
            "name": "modus_ponens",
            "pattern": "If P then Q. P. Therefore Q.",
            "confidence": 0.95
        },
        {
            "name": "syllogism",
            "pattern": "All A are B. All B are C. Therefore all A are C.",
            "confidence": 0.90
        }
    ]

    await pattern_matcher.load_patterns(patterns)

    # Analyze text for logical patterns
    text = "If it rains, the ground gets wet. It is raining. Therefore, the ground is wet."

    matches = await pattern_matcher.find_patterns(text)

    for match in matches:
        print(f"Pattern: {match.pattern_name}")
        print(f"Confidence: {match.confidence}")
        print(f"Components: {match.extracted_components}")
```

### Custom Pattern Definition

```python
async def custom_patterns():
    pattern_matcher = LogicPattern()

    # Define custom reasoning patterns
    custom_pattern = {
        "name": "causal_chain",
        "pattern": "{cause} leads to {effect1}, which causes {effect2}",
        "variables": ["cause", "effect1", "effect2"],
        "validation_rules": [
            "cause != effect1",
            "effect1 != effect2",
            "temporal_order(cause, effect1, effect2)"
        ]
    }

    await pattern_matcher.add_custom_pattern(custom_pattern)

    text = "Poor diet leads to obesity, which causes heart disease."
    matches = await pattern_matcher.find_patterns(text)

    print(f"Identified causal chain: {matches[0].variables}")
```

## Integration with RAG

### Logical RAG Enhancement

```python
from ragents import RAGEngine, RAGConfig
from ragents.logical_llm import LogicalRAGEnhancer

async def logical_rag():
    # Standard RAG setup
    rag_config = RAGConfig.from_env()
    rag_engine = RAGEngine(rag_config, llm_client)

    # Add logical reasoning enhancement
    logical_enhancer = LogicalRAGEnhancer(
        constraint_engine=ConstraintEngine(),
        pattern_matcher=LogicPattern()
    )

    enhanced_rag = logical_enhancer.enhance(rag_engine)

    # Add documents
    await enhanced_rag.add_document("scientific_papers.pdf")

    # Query with logical reasoning
    response = await enhanced_rag.query(
        "If the hypothesis is correct, what would we expect to observe?",
        use_logical_reasoning=True,
        extract_constraints=True
    )

    print(f"Answer: {response.answer}")
    print(f"Logical chain: {response.reasoning_chain}")
    print(f"Constraints identified: {response.constraints}")
```

### Fact Verification

```python
async def fact_verification():
    enhanced_rag = LogicalRAGEnhancer().enhance(rag_engine)

    # Verify facts using logical reasoning
    fact_to_verify = "All renewable energy sources are carbon-neutral"

    verification = await enhanced_rag.verify_fact(
        fact=fact_to_verify,
        search_evidence=True,
        apply_logical_rules=True
    )

    print(f"Verification result: {verification.is_supported}")
    print(f"Evidence: {verification.supporting_evidence}")
    print(f"Contradictions: {verification.contradicting_evidence}")
    print(f"Logical assessment: {verification.logical_consistency}")
```

## Performance Optimization

### Token Usage Reduction

```python
async def optimized_reasoning():
    # Configure for minimal token usage
    reasoner = LogicalReasoner(
        llm_client,
        optimization_strategy="minimal_tokens",
        use_symbolic_shortcuts=True,
        cache_logical_operations=True
    )

    # Mathematical problem that can be solved symbolically
    problem = "If x = 5 and y = 3x + 2, what is the value of 2y - x?"

    result = await reasoner.solve_optimized(problem)

    print(f"Answer: {result.answer}")
    print(f"Method: {result.solution_method}")  # "symbolic" vs "llm"
    print(f"Tokens saved: {result.tokens_saved}")
```

### Caching and Memoization

```python
async def efficient_reasoning():
    reasoner = LogicalReasoner(
        llm_client,
        enable_caching=True,
        cache_patterns=True,
        memoize_proofs=True
    )

    # Similar problems will reuse cached results
    problems = [
        "If A implies B and B implies C, what can we conclude about A and C?",
        "Given A → B and B → C, what is the relationship between A and C?",
        "A leads to B, B leads to C. What about A and C?"
    ]

    results = []
    for problem in problems:
        result = await reasoner.solve_logical_problem(problem)
        results.append(result)
        print(f"Cache hit: {result.from_cache}")
```

## Advanced Features

### Multi-Step Reasoning

```python
async def multi_step_reasoning():
    reasoner = LogicalReasoner(llm_client)

    # Complex multi-step problem
    complex_problem = """
    Given:
    1. All students who study hard pass their exams
    2. Students who pass their exams graduate
    3. Graduated students get good jobs
    4. Alice studies hard

    Question: Will Alice get a good job?
    """

    result = await reasoner.solve_multi_step(
        problem=complex_problem,
        max_steps=10,
        verify_each_step=True
    )

    print(f"Final conclusion: {result.conclusion}")
    print("\nReasoning steps:")
    for i, step in enumerate(result.steps):
        print(f"{i+1}. {step.statement}")
        print(f"   Rule applied: {step.rule}")
        print(f"   Confidence: {step.confidence}")
```

### Probabilistic Reasoning

```python
async def probabilistic_reasoning():
    reasoner = LogicalReasoner(llm_client)

    # Reasoning with uncertainty
    probabilistic_problem = """
    70% of people who exercise regularly are healthy.
    80% of healthy people live long lives.
    John exercises regularly.
    What's the probability that John lives a long life?
    """

    result = await reasoner.solve_probabilistic(
        problem=probabilistic_problem,
        use_bayesian_inference=True
    )

    print(f"Probability: {result.probability}")
    print(f"Calculation: {result.probability_calculation}")
    print(f"Assumptions: {result.assumptions}")
```

## Monitoring and Debugging

### Reasoning Trace

```python
async def trace_reasoning():
    reasoner = LogicalReasoner(
        llm_client,
        enable_tracing=True,
        trace_level="detailed"
    )

    result = await reasoner.solve_logical_problem(
        "If all cats are mammals and all mammals are animals, are cats animals?"
    )

    # Access detailed trace
    trace = result.reasoning_trace
    print(f"Total steps: {len(trace.steps)}")
    print(f"Logical operations: {trace.logical_operations}")
    print(f"Symbolic shortcuts used: {trace.shortcuts_used}")
    print(f"Token efficiency: {trace.token_efficiency}")
```

### Performance Metrics

```python
from ragents.observability import LogicalReasoningMetrics

async def monitor_performance():
    metrics = LogicalReasoningMetrics(reasoner)

    # Get performance statistics
    stats = metrics.get_performance_stats()
    print(f"Average reasoning time: {stats.avg_reasoning_time}")
    print(f"Token reduction rate: {stats.token_reduction_rate}")
    print(f"Accuracy rate: {stats.accuracy_rate}")
    print(f"Cache hit rate: {stats.cache_hit_rate}")

    # Get reasoning pattern analysis
    patterns = metrics.get_pattern_analysis()
    print(f"Most common patterns: {patterns.frequent_patterns}")
    print(f"Success rate by pattern: {patterns.pattern_success_rates}")
```

## Best Practices

### When to Use Logical Reasoning

1. **Mathematical Problems** - Calculations and algebraic manipulations
2. **Logical Deduction** - Rule-based inference and syllogisms
3. **Constraint Satisfaction** - Problems with explicit constraints
4. **Fact Verification** - Checking logical consistency
5. **Pattern Recognition** - Identifying recurring logical structures

### Configuration Guidelines

1. **Balance Accuracy and Speed** - Configure appropriate confidence thresholds
2. **Use Caching Effectively** - Enable caching for repetitive reasoning patterns
3. **Domain-Specific Optimization** - Customize for your specific use cases
4. **Monitor Token Usage** - Track efficiency gains from logical reasoning

### Integration Tips

1. **Combine with RAG** - Use logical reasoning to enhance retrieval
2. **Gradual Enhancement** - Start with simple logical rules, expand gradually
3. **Validate Results** - Always verify logical reasoning outputs
4. **Handle Edge Cases** - Plan for cases where logical reasoning fails

## Example Applications

### Scientific Research Assistant

```python
class LogicalResearchAssistant:
    def __init__(self):
        self.reasoner = LogicalReasoner(llm_client)
        self.constraint_engine = ConstraintEngine()

    async def analyze_hypothesis(self, hypothesis: str, data: dict):
        # Extract logical implications
        implications = await self.reasoner.extract_implications(hypothesis)

        # Check against data constraints
        constraints = await self.constraint_engine.validate_against_data(
            implications, data
        )

        # Generate predictions
        predictions = await self.reasoner.generate_predictions(
            hypothesis, constraints
        )

        return {
            "implications": implications,
            "data_consistency": constraints.is_consistent,
            "predictions": predictions
        }
```

### Legal Reasoning Assistant

```python
class LegalReasoningAssistant:
    def __init__(self):
        self.reasoner = LogicalReasoner(llm_client)
        self.pattern_matcher = LogicPattern()

    async def analyze_legal_case(self, case_facts: str, relevant_laws: list):
        # Identify legal patterns
        patterns = await self.pattern_matcher.find_legal_patterns(case_facts)

        # Apply legal reasoning
        analysis = await self.reasoner.apply_legal_rules(
            facts=case_facts,
            laws=relevant_laws,
            patterns=patterns
        )

        return {
            "applicable_laws": analysis.relevant_laws,
            "reasoning_chain": analysis.legal_reasoning,
            "conclusion": analysis.legal_conclusion,
            "confidence": analysis.confidence
        }
```

## Next Steps

- **[Query Rewriting](query-rewriting.md)** - Optimize queries with logical patterns
- **[Evaluation](evaluation.md)** - Assess reasoning quality and accuracy
- **[Observability](observability.md)** - Monitor logical reasoning performance
- **[API Reference](../api/llm.md)** - Detailed API documentation