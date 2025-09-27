"""ChromaDB vector store implementation."""

import uuid
from typing import Any, Dict, List, Optional

from .base import SearchResult, VectorStore, VectorStoreConfig


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store."""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.client = None
        self.collection = None

    async def initialize(self) -> None:
        """Initialize ChromaDB connection."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")

        # Create client
        if self.config.persist_directory:
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.EphemeralClient()

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We provide embeddings directly
            )
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"}
            )

    async def add_vectors(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        contents: List[str],
    ) -> bool:
        """Add vectors to ChromaDB."""
        try:
            # Convert metadata to strings for ChromaDB compatibility
            chroma_metadata = []
            for meta in metadata:
                chroma_meta = {}
                for key, value in meta.items():
                    # ChromaDB requires string, int, float, or bool values
                    if isinstance(value, (str, int, float, bool)):
                        chroma_meta[key] = value
                    else:
                        chroma_meta[key] = str(value)
                chroma_metadata.append(chroma_meta)

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=chroma_metadata,
                documents=contents
            )
            return True
        except Exception as e:
            print(f"Error adding vectors to ChromaDB: {e}")
            return False

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search ChromaDB for similar vectors."""
        try:
            # Convert filters to ChromaDB format
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    where_clause[key] = {"$eq": value}

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances", "embeddings"]
            )

            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    # Convert distance to similarity score (assuming cosine distance)
                    score = 1.0 - distance

                    # Apply score threshold
                    if score_threshold and score < score_threshold:
                        continue

                    search_results.append(SearchResult(
                        id=doc_id,
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] or {},
                        score=score,
                        embedding=results.get("embeddings", [[]])[0][i] if results.get("embeddings") else None
                    ))

            return search_results
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            return []

    async def get_by_id(self, vector_id: str) -> Optional[SearchResult]:
        """Get a vector by its ID."""
        try:
            result = self.collection.get(
                ids=[vector_id],
                include=["documents", "metadatas", "embeddings"]
            )

            if result["ids"] and result["ids"][0]:
                return SearchResult(
                    id=vector_id,
                    content=result["documents"][0],
                    metadata=result["metadatas"][0] or {},
                    score=1.0,
                    embedding=result.get("embeddings", [[]])[0] if result.get("embeddings") else None
                )
        except Exception as e:
            print(f"Error getting vector by ID: {e}")

        return None

    async def delete_by_id(self, vector_id: str) -> bool:
        """Delete a vector by its ID."""
        try:
            self.collection.delete(ids=[vector_id])
            return True
        except Exception as e:
            print(f"Error deleting vector: {e}")
            return False

    async def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete vectors matching the filter criteria."""
        try:
            # Convert filters to ChromaDB format
            where_clause = {}
            for key, value in filters.items():
                where_clause[key] = {"$eq": value}

            # Get IDs first to count deletions
            results = self.collection.get(
                where=where_clause,
                include=["documents"]
            )

            count = len(results["ids"]) if results["ids"] else 0

            if count > 0:
                self.collection.delete(where=where_clause)

            return count
        except Exception as e:
            print(f"Error deleting by filter: {e}")
            return 0

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors in the collection."""
        try:
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    where_clause[key] = {"$eq": value}

                results = self.collection.get(
                    where=where_clause,
                    include=[]
                )
                return len(results["ids"]) if results["ids"] else 0
            else:
                return self.collection.count()
        except Exception as e:
            print(f"Error counting vectors: {e}")
            return 0

    async def clear(self) -> bool:
        """Clear all vectors from the collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

    async def close(self) -> None:
        """Close the ChromaDB connection."""
        # ChromaDB doesn't require explicit closing
        pass

    def supports_feature(self, feature: str) -> bool:
        """Check if ChromaDB supports a specific feature."""
        chroma_features = {
            "add_vectors", "search", "get_by_id", "delete_by_id",
            "delete_by_filter", "count", "clear", "metadata_filtering"
        }
        return feature in chroma_features