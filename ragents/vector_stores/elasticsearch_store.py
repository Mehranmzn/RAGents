"""Elasticsearch vector store implementation."""

import json
from typing import Any, Dict, List, Optional

from .base import SearchResult, VectorStore, VectorStoreConfig


class ElasticsearchVectorStore(VectorStore):
    """Elasticsearch implementation with vector search capabilities."""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.client = None
        self.index_name = config.index_name or f"vectors_{self.collection_name}"

    async def initialize(self) -> None:
        """Initialize Elasticsearch connection."""
        try:
            from elasticsearch import AsyncElasticsearch
        except ImportError:
            raise ImportError("elasticsearch is required. Install with: pip install elasticsearch")

        # Create client
        client_config = {
            "request_timeout": self.config.timeout,
            "max_retries": 3,
            "retry_on_timeout": True
        }

        if self.config.url:
            client_config["hosts"] = [self.config.url]
        else:
            host = self.config.host or "localhost"
            port = self.config.port or 9200
            client_config["hosts"] = [f"http://{host}:{port}"]

        if self.config.api_key:
            client_config["api_key"] = self.config.api_key

        self.client = AsyncElasticsearch(**client_config)

        # Create index if it doesn't exist
        await self._ensure_index_exists()

    async def _ensure_index_exists(self) -> None:
        """Ensure the Elasticsearch index exists with proper mapping."""
        index_exists = await self.client.indices.exists(index=self.index_name)

        if not index_exists:
            # Create index with vector mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.embedding_dimension,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "metadata": {
                            "type": "object",
                            "dynamic": True
                        },
                        "created_at": {
                            "type": "date"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "index.knn": True
                }
            }

            await self.client.indices.create(
                index=self.index_name,
                body=mapping
            )

    async def add_vectors(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        contents: List[str],
    ) -> bool:
        """Add vectors to Elasticsearch."""
        try:
            from datetime import datetime

            # Prepare bulk operations
            operations = []
            for i, vector_id in enumerate(ids):
                doc = {
                    "embedding": embeddings[i],
                    "content": contents[i],
                    "metadata": metadata[i],
                    "created_at": datetime.utcnow().isoformat()
                }

                operations.extend([
                    {"index": {"_index": self.index_name, "_id": vector_id}},
                    doc
                ])

            # Bulk insert
            response = await self.client.bulk(
                body=operations,
                refresh=True
            )

            # Check for errors
            if response.get("errors"):
                print(f"Some documents failed to index: {response}")
                return False

            return True
        except Exception as e:
            print(f"Error adding vectors to Elasticsearch: {e}")
            return False

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search Elasticsearch for similar vectors."""
        try:
            # Build kNN search query
            knn_query = {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": min(top_k * 10, 1000)
            }

            # Add filters if provided
            query_body = {"knn": knn_query}

            if filters:
                # Convert filters to Elasticsearch query
                filter_query = {"bool": {"must": []}}
                for key, value in filters.items():
                    filter_query["bool"]["must"].append({
                        "term": {f"metadata.{key}": value}
                    })

                if filter_query["bool"]["must"]:
                    knn_query["filter"] = filter_query

            # Add score threshold
            if score_threshold:
                knn_query["min_score"] = score_threshold

            response = await self.client.search(
                index=self.index_name,
                body=query_body,
                size=top_k
            )

            search_results = []
            for hit in response["hits"]["hits"]:
                search_results.append(SearchResult(
                    id=hit["_id"],
                    content=hit["_source"]["content"],
                    metadata=hit["_source"].get("metadata", {}),
                    score=hit["_score"],
                    embedding=hit["_source"].get("embedding")
                ))

            return search_results
        except Exception as e:
            print(f"Error searching Elasticsearch: {e}")
            return []

    async def get_by_id(self, vector_id: str) -> Optional[SearchResult]:
        """Get a vector by its ID."""
        try:
            response = await self.client.get(
                index=self.index_name,
                id=vector_id
            )

            if response["found"]:
                source = response["_source"]
                return SearchResult(
                    id=vector_id,
                    content=source["content"],
                    metadata=source.get("metadata", {}),
                    score=1.0,
                    embedding=source.get("embedding")
                )
        except Exception as e:
            print(f"Error getting vector by ID: {e}")

        return None

    async def delete_by_id(self, vector_id: str) -> bool:
        """Delete a vector by its ID."""
        try:
            response = await self.client.delete(
                index=self.index_name,
                id=vector_id,
                refresh=True
            )
            return response["result"] == "deleted"
        except Exception as e:
            print(f"Error deleting vector: {e}")
            return False

    async def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete vectors matching the filter criteria."""
        try:
            # Build query for filtering
            query = {"bool": {"must": []}}
            for key, value in filters.items():
                query["bool"]["must"].append({
                    "term": {f"metadata.{key}": value}
                })

            response = await self.client.delete_by_query(
                index=self.index_name,
                body={"query": query},
                refresh=True
            )

            return response.get("deleted", 0)
        except Exception as e:
            print(f"Error deleting by filter: {e}")
            return 0

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors in the collection."""
        try:
            query_body = {}
            if filters:
                query = {"bool": {"must": []}}
                for key, value in filters.items():
                    query["bool"]["must"].append({
                        "term": {f"metadata.{key}": value}
                    })
                query_body["query"] = query

            response = await self.client.count(
                index=self.index_name,
                body=query_body
            )

            return response["count"]
        except Exception as e:
            print(f"Error counting vectors: {e}")
            return 0

    async def clear(self) -> bool:
        """Clear all vectors from the collection."""
        try:
            response = await self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"match_all": {}}},
                refresh=True
            )
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

    async def close(self) -> None:
        """Close the Elasticsearch connection."""
        if self.client:
            await self.client.close()

    def supports_feature(self, feature: str) -> bool:
        """Check if Elasticsearch supports a specific feature."""
        elasticsearch_features = {
            "add_vectors", "search", "get_by_id", "delete_by_id",
            "delete_by_filter", "count", "clear", "metadata_filtering",
            "full_text_search", "hybrid_search", "aggregations",
            "complex_queries"
        }
        return feature in elasticsearch_features

    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Hybrid search combining vector similarity and text search."""
        try:
            # Build hybrid query combining kNN and text search
            query_body = {
                "query": {
                    "bool": {
                        "should": [
                            # Vector similarity
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": f"cosineSimilarity(params.query_vector, 'embedding') * {alpha}",
                                        "params": {"query_vector": query_embedding}
                                    }
                                }
                            },
                            # Text search
                            {
                                "match": {
                                    "content": {
                                        "query": query_text,
                                        "boost": 1.0 - alpha
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": top_k
            }

            # Add filters if provided
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append({
                        "term": {f"metadata.{key}": value}
                    })

                if filter_conditions:
                    query_body["query"]["bool"]["filter"] = filter_conditions

            response = await self.client.search(
                index=self.index_name,
                body=query_body
            )

            search_results = []
            for hit in response["hits"]["hits"]:
                search_results.append(SearchResult(
                    id=hit["_id"],
                    content=hit["_source"]["content"],
                    metadata=hit["_source"].get("metadata", {}),
                    score=hit["_score"],
                    embedding=hit["_source"].get("embedding")
                ))

            return search_results
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            # Fallback to vector search
            return await self.search(query_embedding, top_k, filters)

    async def full_text_search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Full-text search without vector similarity."""
        try:
            query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "content": {
                                        "query": query_text,
                                        "operator": "and"
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": top_k,
                "highlight": {
                    "fields": {
                        "content": {}
                    }
                }
            }

            # Add filters
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append({
                        "term": {f"metadata.{key}": value}
                    })

                if filter_conditions:
                    query_body["query"]["bool"]["filter"] = filter_conditions

            response = await self.client.search(
                index=self.index_name,
                body=query_body
            )

            search_results = []
            for hit in response["hits"]["hits"]:
                metadata = hit["_source"].get("metadata", {})

                # Add highlight information to metadata
                if "highlight" in hit:
                    metadata["highlights"] = hit["highlight"]

                search_results.append(SearchResult(
                    id=hit["_id"],
                    content=hit["_source"]["content"],
                    metadata=metadata,
                    score=hit["_score"],
                    embedding=hit["_source"].get("embedding")
                ))

            return search_results
        except Exception as e:
            print(f"Error in full-text search: {e}")
            return []