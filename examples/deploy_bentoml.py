"""Example: Deploy RAGents with BentoML for production scalability."""

import asyncio
from ragents.deployment.bentoml_service import RAGentsService, ServiceConfig
from ragents.llm.types import ModelConfig
from ragents.config.rag_config import RAGConfig
from ragents.rag.cache import CacheConfig
from ragents.agents.checkpointing import CheckpointConfig


async def main():
    """Deploy RAGents with BentoML."""

    # Configure LLM
    model_config = ModelConfig(
        provider="openai",
        model_name="gpt-4",
        temperature=0.7
    )

    # Configure RAG
    rag_config = RAGConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5
    )

    # Configure Redis cache
    cache_config = CacheConfig(
        redis_url="redis://localhost:6379/0",
        embedding_ttl=86400,  # 24 hours
        retrieval_ttl=3600,   # 1 hour
        enable_compression=True
    )

    # Configure distributed checkpointing (choose Redis or PostgreSQL)
    checkpoint_config = CheckpointConfig(
        backend="redis",  # or "postgres"
        redis_url="redis://localhost:6379/1",
        # postgres_url="postgresql://user:pass@localhost:5432/checkpoints",
        ttl_seconds=3600,
        enable_compression=True
    )

    # Create service configuration
    service_config = ServiceConfig(
        model_config=model_config,
        rag_config=rag_config,
        cache_config=cache_config,
        checkpoint_config=checkpoint_config,
        enable_opik=True
    )

    # Initialize service
    service = RAGentsService()
    await service.initialize(service_config)

    print("✅ RAGents BentoML service initialized")
    print(f"✅ {len(service.agents)} agents loaded")
    print(f"✅ Cache status: {service.cache._initialized if service.cache else 'disabled'}")
    print(f"✅ Checkpointing: {checkpoint_config.backend}")
    print("\n📝 To build and serve:")
    print("   bentoml build")
    print("   bentoml serve ragents:latest")
    print("\n🐳 To containerize:")
    print("   bentoml containerize ragents:latest")
    print("   docker run -p 3000:3000 ragents:latest")
    print("\n☸️  To deploy to Kubernetes:")
    print("   bentoml deploy ragents:latest --cluster-name my-cluster")


if __name__ == "__main__":
    asyncio.run(main())
