"""Base vector store interface inspired by Elysia's clean abstractions."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    CHROMADB = "chromadb"
    WEAVIATE = "weaviate"
    PGVECTOR = "pgvector"
    ELASTICSEARCH = "elasticsearch"
    MEMORY = "memory"


class SearchResult(BaseModel):
    """Result from vector similarity search."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = Field(ge=0.0, le=1.0)
    embedding: Optional[List[float]] = None


class VectorStoreConfig(BaseModel):
    """Configuration for vector stores."""
    store_type: VectorStoreType
    collection_name: str = "default"
    embedding_dimension: int = 384

    # Connection settings
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    url: Optional[str] = None

    # Storage settings
    persist_directory: Optional[str] = None

    # Database specific settings
    database_url: Optional[str] = None  # For PostgreSQL
    index_name: Optional[str] = None    # For Elasticsearch

    # Performance settings
    batch_size: int = 100
    max_connections: int = 10
    timeout: int = 30

    # Additional configuration
    extra_config: Dict[str, Any] = Field(default_factory=dict)


class VectorStore(ABC):
    """Abstract base class for vector stores with clean interface."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.collection_name = config.collection_name
        self.embedding_dimension = config.embedding_dimension

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store connection and create collection if needed."""
        pass

    @abstractmethod
    async def add_vectors(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        contents: List[str],
    ) -> bool:
        """Add vectors to the store."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def get_by_id(self, vector_id: str) -> Optional[SearchResult]:
        """Get a vector by its ID."""
        pass

    @abstractmethod
    async def delete_by_id(self, vector_id: str) -> bool:
        """Delete a vector by its ID."""
        pass

    @abstractmethod
    async def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete vectors matching the filter criteria."""
        pass

    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors in the collection."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all vectors from the collection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection to the vector store."""
        pass

    # Optional methods with default implementations
    async def batch_add_vectors(
        self,
        batch_data: List[Tuple[str, List[float], Dict[str, Any], str]],
    ) -> bool:
        """Add vectors in batches for better performance."""
        batch_size = self.config.batch_size

        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            ids = [item[0] for item in batch]
            embeddings = [item[1] for item in batch]
            metadata = [item[2] for item in batch]
            contents = [item[3] for item in batch]

            success = await self.add_vectors(ids, embeddings, metadata, contents)
            if not success:
                return False

        return True

    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.7,  # Weight for semantic vs text search
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Hybrid search combining semantic and text search."""
        # Default implementation just does semantic search
        # Subclasses can override for true hybrid search
        return await self.search(query_embedding, top_k, filters)

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        total_count = await self.count()
        return {
            "collection_name": self.collection_name,
            "total_vectors": total_count,
            "embedding_dimension": self.embedding_dimension,
            "store_type": self.config.store_type,
        }

    def supports_feature(self, feature: str) -> bool:
        """Check if the vector store supports a specific feature."""
        # Base features all stores should support
        base_features = {
            "add_vectors",
            "search",
            "get_by_id",
            "delete_by_id",
            "count",
            "clear"
        }
        return feature in base_features

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class MemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing and development."""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.vectors: Dict[str, Tuple[List[float], Dict[str, Any], str]] = {}

    async def initialize(self) -> None:
        """Initialize the memory store."""
        pass

    async def add_vectors(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        contents: List[str],
    ) -> bool:
        """Add vectors to memory."""
        for i, vector_id in enumerate(ids):
            self.vectors[vector_id] = (embeddings[i], metadata[i], contents[i])
        return True

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search using cosine similarity."""
        import numpy as np

        results = []
        query_vec = np.array(query_embedding)

        for vector_id, (embedding, metadata, content) in self.vectors.items():
            # Apply filters
            if filters:
                if not self._matches_filter(metadata, filters):
                    continue

            # Calculate cosine similarity
            vec = np.array(embedding)
            similarity = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))

            # Apply score threshold
            if score_threshold and similarity < score_threshold:
                continue

            results.append(SearchResult(
                id=vector_id,
                content=content,
                metadata=metadata,
                score=float(similarity),
                embedding=embedding
            ))

        # Sort by score descending and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _matches_filter(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    async def get_by_id(self, vector_id: str) -> Optional[SearchResult]:
        """Get vector by ID."""
        if vector_id in self.vectors:
            embedding, metadata, content = self.vectors[vector_id]
            return SearchResult(
                id=vector_id,
                content=content,
                metadata=metadata,
                score=1.0,
                embedding=embedding
            )
        return None

    async def delete_by_id(self, vector_id: str) -> bool:
        """Delete vector by ID."""
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            return True
        return False

    async def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete vectors matching filters."""
        to_delete = []
        for vector_id, (_, metadata, _) in self.vectors.items():
            if self._matches_filter(metadata, filters):
                to_delete.append(vector_id)

        for vector_id in to_delete:
            del self.vectors[vector_id]

        return len(to_delete)

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors."""
        if not filters:
            return len(self.vectors)

        count = 0
        for _, (_, metadata, _) in self.vectors.items():
            if self._matches_filter(metadata, filters):
                count += 1
        return count

    async def clear(self) -> bool:
        """Clear all vectors."""
        self.vectors.clear()
        return True

    async def close(self) -> None:
        """Close the memory store."""
        pass