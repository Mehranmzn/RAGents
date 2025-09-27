"""
Kubeflow Pipeline for RAGents model training and evaluation.
"""

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics


@component(
    base_image="python:3.11-slim",
    packages_to_install=["ragents[all]", "pandas", "scikit-learn"]
)
def data_preparation_component(
    input_data: Input[Dataset],
    processed_data: Output[Dataset],
    chunk_size: int = 1000,
    overlap: int = 100
) -> None:
    """Prepare and chunk documents for RAG training."""
    import pandas as pd
    from ragents.ingestion import DocumentProcessor

    # Load input data
    df = pd.read_csv(input_data.path)

    # Initialize document processor
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    # Process documents
    processed_chunks = []
    for _, row in df.iterrows():
        chunks = processor.process_text(row['content'])
        for chunk in chunks:
            processed_chunks.append({
                'document_id': row['document_id'],
                'chunk_text': chunk.text,
                'metadata': chunk.metadata
            })

    # Save processed data
    processed_df = pd.DataFrame(processed_chunks)
    processed_df.to_csv(processed_data.path, index=False)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["ragents[all]", "chromadb"]
)
def vector_store_setup_component(
    processed_data: Input[Dataset],
    vector_store: Output[Dataset],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> None:
    """Set up vector store with processed documents."""
    import pandas as pd
    from ragents.vector_stores import create_vector_store
    from ragents.config import VectorStoreConfig

    # Load processed data
    df = pd.read_csv(processed_data.path)

    # Configure vector store
    config = VectorStoreConfig(
        store_type="chromadb",
        embedding_model=embedding_model,
        persist_directory=vector_store.path
    )

    # Create and populate vector store
    store = create_vector_store(config)

    # Add documents to vector store
    documents = df['chunk_text'].tolist()
    metadatas = df['metadata'].apply(eval).tolist()
    ids = [f"doc_{i}" for i in range(len(documents))]

    store.add_documents(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )


@component(
    base_image="python:3.11-slim",
    packages_to_install=["ragents[all]", "openai", "anthropic"]
)
def model_training_component(
    vector_store: Input[Dataset],
    trained_model: Output[Model],
    llm_provider: str = "openai",
    model_name: str = "gpt-3.5-turbo"
) -> None:
    """Train RAGents model with custom configurations."""
    import json
    from ragents import RAGEngine, LLMClient, DecisionTreeAgent
    from ragents.config import RAGConfig, LLMConfig, AgentConfig

    # Configure LLM
    llm_config = LLMConfig(
        provider=llm_provider,
        model_name=model_name
    )
    llm_client = LLMClient(llm_config)

    # Configure RAG
    rag_config = RAGConfig(
        vector_store_path=vector_store.path,
        chunk_size=1000,
        top_k=5
    )
    rag_engine = RAGEngine(rag_config, llm_client)

    # Configure Agent
    agent_config = AgentConfig(
        name="Trained RAGent",
        description="Production-ready RAG agent",
        enable_rag=True,
        enable_reasoning=True
    )

    # Create and configure agent
    agent = DecisionTreeAgent(
        config=agent_config,
        llm_client=llm_client,
        rag_engine=rag_engine
    )

    # Save model configuration
    model_config = {
        "llm_config": llm_config.dict(),
        "rag_config": rag_config.dict(),
        "agent_config": agent_config.dict()
    }

    with open(f"{trained_model.path}/config.json", "w") as f:
        json.dump(model_config, f, indent=2)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["ragents[evaluation]", "pandas"]
)
def model_evaluation_component(
    trained_model: Input[Model],
    test_data: Input[Dataset],
    evaluation_metrics: Output[Metrics]
) -> None:
    """Evaluate the trained RAGents model."""
    import json
    import pandas as pd
    from ragents.evaluation import RAGEvaluator

    # Load test data
    test_df = pd.read_csv(test_data.path)

    # Load model configuration
    with open(f"{trained_model.path}/config.json", "r") as f:
        model_config = json.load(f)

    # Initialize evaluator
    evaluator = RAGEvaluator()

    # Prepare evaluation data
    questions = test_df['question'].tolist()
    ground_truths = test_df['ground_truth'].tolist()
    contexts = test_df['context'].tolist()

    # Run evaluation
    results = evaluator.evaluate_batch(
        questions=questions,
        ground_truths=ground_truths,
        contexts=contexts
    )

    # Save metrics
    metrics = {
        "faithfulness": results.faithfulness.mean(),
        "answer_relevancy": results.answer_relevancy.mean(),
        "context_precision": results.context_precision.mean(),
        "context_recall": results.context_recall.mean()
    }

    with open(evaluation_metrics.path, "w") as f:
        json.dump(metrics, f, indent=2)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["ragents[deployment]", "kubernetes"]
)
def model_deployment_component(
    trained_model: Input[Model],
    deployment_config: dict
) -> None:
    """Deploy the trained model to Kubernetes."""
    import json
    import yaml
    from kubernetes import client, config

    # Load model configuration
    with open(f"{trained_model.path}/config.json", "r") as f:
        model_config = json.load(f)

    # Create Kubernetes deployment manifest
    deployment_manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "ragents-model",
            "labels": {"app": "ragents"}
        },
        "spec": {
            "replicas": deployment_config.get("replicas", 3),
            "selector": {"matchLabels": {"app": "ragents"}},
            "template": {
                "metadata": {"labels": {"app": "ragents"}},
                "spec": {
                    "containers": [{
                        "name": "ragents",
                        "image": deployment_config.get("image", "ragents:latest"),
                        "ports": [{"containerPort": 8000}],
                        "env": [
                            {"name": "OPENAI_API_KEY", "valueFrom": {"secretKeyRef": {"name": "ragents-secrets", "key": "openai-api-key"}}},
                            {"name": "RAGENTS_LLM_PROVIDER", "value": model_config["llm_config"]["provider"]}
                        ],
                        "resources": {
                            "requests": {"memory": "1Gi", "cpu": "500m"},
                            "limits": {"memory": "2Gi", "cpu": "1000m"}
                        }
                    }]
                }
            }
        }
    }

    # Apply deployment (in real scenario, you'd use kubectl or K8s Python client)
    print("Deployment manifest created:")
    print(yaml.dump(deployment_manifest, default_flow_style=False))


@pipeline(
    name="ragents-training-pipeline",
    description="Training and deployment pipeline for RAGents"
)
def ragents_training_pipeline(
    input_data: str,
    test_data: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_provider: str = "openai",
    model_name: str = "gpt-3.5-turbo",
    deployment_replicas: int = 3
):
    """
    Complete pipeline for training and deploying RAGents.

    Args:
        input_data: Path to training data CSV
        test_data: Path to test data CSV
        chunk_size: Document chunk size
        overlap: Chunk overlap size
        embedding_model: Embedding model name
        llm_provider: LLM provider (openai/anthropic)
        model_name: LLM model name
        deployment_replicas: Number of deployment replicas
    """

    # Data preparation
    data_prep = data_preparation_component(
        input_data=input_data,
        chunk_size=chunk_size,
        overlap=overlap
    )

    # Vector store setup
    vector_setup = vector_store_setup_component(
        processed_data=data_prep.outputs['processed_data'],
        embedding_model=embedding_model
    )

    # Model training
    training = model_training_component(
        vector_store=vector_setup.outputs['vector_store'],
        llm_provider=llm_provider,
        model_name=model_name
    )

    # Model evaluation
    evaluation = model_evaluation_component(
        trained_model=training.outputs['trained_model'],
        test_data=test_data
    )

    # Model deployment
    deployment = model_deployment_component(
        trained_model=training.outputs['trained_model'],
        deployment_config={
            "replicas": deployment_replicas,
            "image": "ragents:latest"
        }
    )

    # Set dependencies
    vector_setup.after(data_prep)
    training.after(vector_setup)
    evaluation.after(training)
    deployment.after(evaluation)


if __name__ == "__main__":
    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=ragents_training_pipeline,
        package_path="ragents_training_pipeline.yaml"
    )