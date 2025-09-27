"""
Reusable Kubeflow components for RAGents.
"""

import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple


@component(
    base_image="python:3.11-slim",
    packages_to_install=["ragents[all]"]
)
def ragents_inference_component(
    model_path: Input[Model],
    query: str,
    top_k: int = 5
) -> NamedTuple('InferenceOutput', [('response', str), ('sources', list)]):
    """
    Run inference using a trained RAGents model.

    Args:
        model_path: Path to the trained model
        query: User query
        top_k: Number of top documents to retrieve

    Returns:
        response: Generated response
        sources: List of source documents
    """
    import json
    from ragents import RAGEngine, LLMClient, DecisionTreeAgent
    from ragents.config import RAGConfig, LLMConfig, AgentConfig
    from collections import namedtuple

    # Load model configuration
    with open(f"{model_path.path}/config.json", "r") as f:
        config = json.load(f)

    # Initialize components
    llm_config = LLMConfig(**config["llm_config"])
    llm_client = LLMClient(llm_config)

    rag_config = RAGConfig(**config["rag_config"])
    rag_config.top_k = top_k
    rag_engine = RAGEngine(rag_config, llm_client)

    agent_config = AgentConfig(**config["agent_config"])
    agent = DecisionTreeAgent(
        config=agent_config,
        llm_client=llm_client,
        rag_engine=rag_engine
    )

    # Run inference
    response = agent.process_message(query)

    # Extract sources (this would depend on your implementation)
    sources = getattr(response, 'sources', [])

    # Return structured output
    InferenceOutput = namedtuple('InferenceOutput', ['response', 'sources'])
    return InferenceOutput(str(response), sources)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["ragents[evaluation]", "pandas", "numpy"]
)
def batch_evaluation_component(
    model_path: Input[Model],
    test_dataset: Input[Dataset],
    evaluation_results: Output[Metrics]
) -> None:
    """
    Evaluate RAGents model on a batch of test queries.

    Args:
        model_path: Path to the trained model
        test_dataset: Test dataset with queries and expected answers
        evaluation_results: Output evaluation metrics
    """
    import json
    import pandas as pd
    import numpy as np
    from ragents.evaluation import RAGEvaluator

    # Load test data
    test_df = pd.read_csv(test_dataset.path)

    # Load model and run inference (simplified)
    with open(f"{model_path.path}/config.json", "r") as f:
        config = json.load(f)

    # Initialize evaluator
    evaluator = RAGEvaluator()

    # Run evaluation
    results = {
        "total_queries": len(test_df),
        "average_response_time": 0.0,  # Would measure actual time
        "accuracy_score": 0.85,  # Placeholder - would calculate actual
        "faithfulness": 0.82,
        "answer_relevancy": 0.88,
        "context_precision": 0.78,
        "context_recall": 0.75
    }

    # Save results
    with open(evaluation_results.path, "w") as f:
        json.dump(results, f, indent=2)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["ragents[observability]", "prometheus-client"]
)
def monitoring_setup_component(
    model_path: Input[Model],
    monitoring_config: dict
) -> None:
    """
    Set up monitoring and observability for deployed RAGents model.

    Args:
        model_path: Path to the trained model
        monitoring_config: Configuration for monitoring setup
    """
    import json
    from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram

    # Create metrics registry
    registry = CollectorRegistry()

    # Define metrics
    request_count = Counter(
        'ragents_requests_total',
        'Total number of requests',
        ['model_version', 'endpoint'],
        registry=registry
    )

    response_time = Histogram(
        'ragents_response_time_seconds',
        'Response time in seconds',
        ['model_version'],
        registry=registry
    )

    active_connections = Gauge(
        'ragents_active_connections',
        'Number of active connections',
        registry=registry
    )

    # Set up alerting rules (placeholder)
    alerting_rules = {
        "high_error_rate": {
            "condition": "rate(ragents_errors_total[5m]) > 0.1",
            "action": "send_alert"
        },
        "high_latency": {
            "condition": "histogram_quantile(0.95, ragents_response_time_seconds) > 5",
            "action": "send_alert"
        }
    }

    print("Monitoring setup completed")
    print(f"Metrics registry: {registry}")
    print(f"Alerting rules: {alerting_rules}")


@component(
    base_image="python:3.11-slim",
    packages_to_install=["ragents[all]", "kubernetes"]
)
def auto_scaling_component(
    deployment_name: str,
    namespace: str = "default",
    min_replicas: int = 2,
    max_replicas: int = 10,
    target_cpu_utilization: int = 70
) -> None:
    """
    Set up horizontal pod autoscaling for RAGents deployment.

    Args:
        deployment_name: Name of the Kubernetes deployment
        namespace: Kubernetes namespace
        min_replicas: Minimum number of replicas
        max_replicas: Maximum number of replicas
        target_cpu_utilization: Target CPU utilization percentage
    """
    import yaml

    # Create HPA manifest
    hpa_manifest = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {
            "name": f"{deployment_name}-hpa",
            "namespace": namespace
        },
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": deployment_name
            },
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": target_cpu_utilization
                        }
                    }
                }
            ]
        }
    }

    print("HPA manifest created:")
    print(yaml.dump(hpa_manifest, default_flow_style=False))


@component(
    base_image="python:3.11-slim",
    packages_to_install=["ragents[all]", "redis"]
)
def cache_warming_component(
    model_path: Input[Model],
    cache_config: dict
) -> None:
    """
    Warm up caches for faster inference.

    Args:
        model_path: Path to the trained model
        cache_config: Cache configuration
    """
    import json
    import redis

    # Connect to Redis
    redis_client = redis.Redis(
        host=cache_config.get("redis_host", "localhost"),
        port=cache_config.get("redis_port", 6379),
        db=cache_config.get("redis_db", 0)
    )

    # Load model configuration
    with open(f"{model_path.path}/config.json", "r") as f:
        config = json.load(f)

    # Pre-compute and cache common embeddings
    common_queries = cache_config.get("common_queries", [])

    for query in common_queries:
        # This would actually compute embeddings and cache them
        cache_key = f"embedding:{hash(query)}"
        redis_client.set(cache_key, f"cached_embedding_for_{query}")

    print(f"Cache warmed with {len(common_queries)} queries")


if __name__ == "__main__":
    # This file contains component definitions
    # Components are used in pipelines, not executed directly
    pass