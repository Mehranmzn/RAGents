"""Weaviate vector store implementation respecting Elysia's patterns."""

import uuid
from typing import Any, Dict, List, Optional

from .base import SearchResult, VectorStore, VectorStoreConfig


class WeaviateVectorStore(VectorStore):
    """Weaviate implementation following Elysia's client patterns."""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.client = None
        self.class_name = self._format_class_name(config.collection_name)

    def _format_class_name(self, name: str) -> str:
        """Format collection name for Weaviate class naming conventions."""
        # Weaviate class names must start with uppercase
        formatted = ''.join(word.capitalize() for word in name.replace('_', ' ').split())
        return formatted or "DefaultCollection"

    async def initialize(self) -> None:
        """Initialize Weaviate connection using Elysia-style patterns."""
        try:
            import weaviate
            from weaviate.auth import AuthApiKey
        except ImportError:
            raise ImportError("weaviate-client is required. Install with: pip install weaviate-client")

        # Create client with authentication similar to Elysia
        auth_config = None
        if self.config.api_key:
            auth_config = AuthApiKey(api_key=self.config.api_key)

        # Support both cloud and local Weaviate instances like Elysia
        if self.config.url:
            self.client = weaviate.Client(
                url=self.config.url,
                auth_client_secret=auth_config,
                timeout_config=(self.config.timeout, self.config.timeout * 2)
            )
        else:
            # Local instance
            host = self.config.host or "localhost"
            port = self.config.port or 8080
            self.client = weaviate.Client(
                url=f"http://{host}:{port}",
                timeout_config=(self.config.timeout, self.config.timeout * 2)
            )

        # Create class if it doesn't exist
        await self._ensure_class_exists()

    async def _ensure_class_exists(self) -> None:
        """Ensure the Weaviate class exists, create if not."""
        class_schema = {
            "class": self.class_name,
            "description": f"Vector collection for {self.collection_name}",
            "vectorizer": "none",  # We provide vectors directly
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The main content of the document"
                },
                {
                    "name": "metadata",
                    "dataType": ["text"],
                    "description": "JSON-encoded metadata"
                }
            ]
        }

        # Check if class exists
        if not self.client.schema.exists(self.class_name):
            self.client.schema.create_class(class_schema)

    async def add_vectors(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        contents: List[str],
    ) -> bool:
        """Add vectors to Weaviate."""
        try:
            import json

            # Batch insert for efficiency
            with self.client.batch as batch:
                batch.batch_size = min(self.config.batch_size, 100)

                for i, vector_id in enumerate(ids):
                    properties = {
                        "content": contents[i],
                        "metadata": json.dumps(metadata[i])
                    }

                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.class_name,
                        uuid=vector_id,
                        vector=embeddings[i]
                    )

            return True
        except Exception as e:
            print(f"Error adding vectors to Weaviate: {e}")
            return False

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search Weaviate for similar vectors."""
        try:
            import json

            # Build query
            query = (
                self.client.query
                .get(self.class_name, ["content", "metadata"])
                .with_near_vector({
                    "vector": query_embedding,
                    "certainty": score_threshold or 0.0
                })
                .with_limit(top_k)
                .with_additional(["id", "certainty"])
            )

            # Add filters if provided
            if filters:
                where_filter = self._build_where_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)

            result = query.do()

            search_results = []
            if "data" in result and "Get" in result["data"]:
                objects = result["data"]["Get"].get(self.class_name, [])

                for obj in objects:
                    # Parse metadata
                    metadata_str = obj.get("metadata", "{}")
                    try:
                        parsed_metadata = json.loads(metadata_str)
                    except:
                        parsed_metadata = {}

                    certainty = obj.get("_additional", {}).get("certainty", 0.0)
                    object_id = obj.get("_additional", {}).get("id", str(uuid.uuid4()))

                    search_results.append(SearchResult(
                        id=object_id,
                        content=obj.get("content", ""),
                        metadata=parsed_metadata,
                        score=certainty
                    ))

            return search_results
        except Exception as e:
            print(f"Error searching Weaviate: {e}")
            return []

    def _build_where_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build Weaviate where filter from simple key-value filters."""
        if not filters:
            return None

        # Simple implementation - could be enhanced for complex filters
        conditions = []
        for key, value in filters.items():
            conditions.append({
                "path": ["metadata"],
                "operator": "Like",
                "valueText": f"*{key}*{value}*"
            })

        if len(conditions) == 1:
            return conditions[0]
        else:
            return {
                "operator": "And",
                "operands": conditions
            }

    async def get_by_id(self, vector_id: str) -> Optional[SearchResult]:
        """Get a vector by its ID."""
        try:
            import json

            result = (
                self.client.query
                .get(self.class_name, ["content", "metadata"])
                .with_where({
                    "path": ["id"],
                    "operator": "Equal",
                    "valueText": vector_id
                })
                .with_additional(["id"])
                .do()
            )

            if "data" in result and "Get" in result["data"]:
                objects = result["data"]["Get"].get(self.class_name, [])
                if objects:
                    obj = objects[0]
                    metadata_str = obj.get("metadata", "{}")
                    try:
                        parsed_metadata = json.loads(metadata_str)
                    except:
                        parsed_metadata = {}

                    return SearchResult(
                        id=vector_id,
                        content=obj.get("content", ""),
                        metadata=parsed_metadata,
                        score=1.0
                    )
        except Exception as e:
            print(f"Error getting vector by ID: {e}")

        return None

    async def delete_by_id(self, vector_id: str) -> bool:
        """Delete a vector by its ID."""
        try:
            self.client.data_object.delete(
                uuid=vector_id,
                class_name=self.class_name
            )
            return True
        except Exception as e:
            print(f"Error deleting vector: {e}")
            return False

    async def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete vectors matching the filter criteria."""
        try:
            # First get the objects to count them
            where_filter = self._build_where_filter(filters)
            if not where_filter:
                return 0

            result = (
                self.client.query
                .get(self.class_name, ["content"])
                .with_where(where_filter)
                .with_additional(["id"])
                .do()
            )

            count = 0
            if "data" in result and "Get" in result["data"]:
                objects = result["data"]["Get"].get(self.class_name, [])
                count = len(objects)

                # Delete each object
                for obj in objects:
                    object_id = obj.get("_additional", {}).get("id")
                    if object_id:
                        await self.delete_by_id(object_id)

            return count
        except Exception as e:
            print(f"Error deleting by filter: {e}")
            return 0

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors in the collection."""
        try:
            query = self.client.query.aggregate(self.class_name).with_meta_count()

            if filters:
                where_filter = self._build_where_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)

            result = query.do()

            if "data" in result and "Aggregate" in result["data"]:
                aggregate_data = result["data"]["Aggregate"].get(self.class_name, [])
                if aggregate_data and "meta" in aggregate_data[0]:
                    return aggregate_data[0]["meta"]["count"]

            return 0
        except Exception as e:
            print(f"Error counting vectors: {e}")
            return 0

    async def clear(self) -> bool:
        """Clear all vectors from the collection."""
        try:
            # Delete and recreate the class
            self.client.schema.delete_class(self.class_name)
            await self._ensure_class_exists()
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

    async def close(self) -> None:
        """Close the Weaviate connection."""
        # Weaviate client doesn't require explicit closing
        pass

    def supports_feature(self, feature: str) -> bool:
        """Check if Weaviate supports a specific feature."""
        weaviate_features = {
            "add_vectors", "search", "get_by_id", "delete_by_id",
            "delete_by_filter", "count", "clear", "metadata_filtering",
            "hybrid_search", "graph_queries"
        }
        return feature in weaviate_features

    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Weaviate native hybrid search combining vector and keyword search."""
        try:
            import json

            # Build hybrid query
            query = (
                self.client.query
                .get(self.class_name, ["content", "metadata"])
                .with_hybrid(
                    query=query_text,
                    vector=query_embedding,
                    alpha=alpha
                )
                .with_limit(top_k)
                .with_additional(["id", "score"])
            )

            # Add filters if provided
            if filters:
                where_filter = self._build_where_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)

            result = query.do()

            search_results = []
            if "data" in result and "Get" in result["data"]:
                objects = result["data"]["Get"].get(self.class_name, [])

                for obj in objects:
                    metadata_str = obj.get("metadata", "{}")
                    try:
                        parsed_metadata = json.loads(metadata_str)
                    except:
                        parsed_metadata = {}

                    score = obj.get("_additional", {}).get("score", 0.0)
                    object_id = obj.get("_additional", {}).get("id", str(uuid.uuid4()))

                    search_results.append(SearchResult(
                        id=object_id,
                        content=obj.get("content", ""),
                        metadata=parsed_metadata,
                        score=score
                    ))

            return search_results
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return await self.search(query_embedding, top_k, filters)