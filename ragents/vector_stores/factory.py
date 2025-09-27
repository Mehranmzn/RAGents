"""Vector store factory for creating different backend implementations."""

from typing import Type

from .base import VectorStore, VectorStoreConfig, VectorStoreType, MemoryVectorStore
from .chroma_store import ChromaVectorStore
from .elasticsearch_store import ElasticsearchVectorStore
from .pgvector_store import PgVectorStore
from .weaviate_store import WeaviateVectorStore


def create_vector_store(config: VectorStoreConfig) -> VectorStore:
    """Factory function to create vector store instances."""

    store_classes: dict[VectorStoreType, Type[VectorStore]] = {
        VectorStoreType.CHROMADB: ChromaVectorStore,
        VectorStoreType.WEAVIATE: WeaviateVectorStore,
        VectorStoreType.PGVECTOR: PgVectorStore,
        VectorStoreType.ELASTICSEARCH: ElasticsearchVectorStore,
        VectorStoreType.MEMORY: MemoryVectorStore,
    }

    if config.store_type not in store_classes:
        raise ValueError(f"Unsupported vector store type: {config.store_type}")

    store_class = store_classes[config.store_type]
    return store_class(config)


def get_available_backends() -> list[str]:
    """Get list of available vector store backends."""
    available = ["memory"]  # Always available

    # Check for optional dependencies
    try:
        import chromadb
        available.append("chromadb")
    except ImportError:
        pass

    try:
        import weaviate
        available.append("weaviate")
    except ImportError:
        pass

    try:
        import asyncpg
        available.append("pgvector")
    except ImportError:
        pass

    try:
        import elasticsearch
        available.append("elasticsearch")
    except ImportError:
        pass

    return available


def recommend_backend(
    data_size: str = "small",
    use_case: str = "general",
    performance_priority: str = "balanced"
) -> str:
    """Recommend a vector store backend based on requirements."""

    recommendations = {
        ("small", "general", "balanced"): "chromadb",
        ("small", "development", "balanced"): "memory",
        ("medium", "general", "balanced"): "chromadb",
        ("medium", "production", "performance"): "weaviate",
        ("large", "production", "performance"): "elasticsearch",
        ("large", "enterprise", "reliability"): "pgvector",
    }

    key = (data_size, use_case, performance_priority)

    # Find best match
    for (size, case, priority), backend in recommendations.items():
        if (data_size == size or data_size == "any") and \
           (use_case == case or use_case == "any") and \
           (performance_priority == priority or performance_priority == "any"):
            return backend

    # Default fallback
    available = get_available_backends()

    if "chromadb" in available:
        return "chromadb"
    elif "memory" in available:
        return "memory"
    else:
        return available[0] if available else "memory"


def create_vector_store_from_url(url: str, **kwargs) -> VectorStore:
    """Create vector store from connection URL."""

    if url.startswith("postgresql://") or url.startswith("postgres://"):
        config = VectorStoreConfig(
            store_type=VectorStoreType.PGVECTOR,
            database_url=url,
            **kwargs
        )
    elif url.startswith("http://") and ":9200" in url:
        config = VectorStoreConfig(
            store_type=VectorStoreType.ELASTICSEARCH,
            url=url,
            **kwargs
        )
    elif url.startswith("http://") and ":8080" in url:
        config = VectorStoreConfig(
            store_type=VectorStoreType.WEAVIATE,
            url=url,
            **kwargs
        )
    elif url.startswith("chroma://") or url.startswith("file://"):
        persist_dir = url.replace("chroma://", "").replace("file://", "")
        config = VectorStoreConfig(
            store_type=VectorStoreType.CHROMADB,
            persist_directory=persist_dir,
            **kwargs
        )
    else:
        raise ValueError(f"Cannot determine vector store type from URL: {url}")

    return create_vector_store(config)