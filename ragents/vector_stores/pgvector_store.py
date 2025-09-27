"""PostgreSQL with pgvector extension implementation."""

import json
import uuid
from typing import Any, Dict, List, Optional

from .base import SearchResult, VectorStore, VectorStoreConfig


class PgVectorStore(VectorStore):
    """PostgreSQL with pgvector extension for vector storage."""

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.connection_pool = None
        self.table_name = f"vectors_{self.collection_name}"

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection and create tables."""
        try:
            import asyncpg
        except ImportError:
            raise ImportError("asyncpg is required. Install with: pip install asyncpg")

        if not self.config.database_url:
            raise ValueError("database_url is required for PgVectorStore")

        # Create connection pool
        self.connection_pool = await asyncpg.create_pool(
            self.config.database_url,
            min_size=1,
            max_size=self.config.max_connections,
            command_timeout=self.config.timeout
        )

        # Create table and indexes
        await self._setup_database()

    async def _setup_database(self) -> None:
        """Set up database tables and indexes."""
        async with self.connection_pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self.embedding_dimension}),
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Create vector index for similarity search
            try:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                    ON {self.table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
            except Exception:
                # Fallback to basic index if IVFFlat fails
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_basic_idx
                    ON {self.table_name}
                    USING ivfflat (embedding vector_cosine_ops);
                """)

            # Create metadata index
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx
                ON {self.table_name}
                USING GIN (metadata);
            """)

    async def add_vectors(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        contents: List[str],
    ) -> bool:
        """Add vectors to PostgreSQL."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Prepare data for batch insert
                rows = []
                for i, vector_id in enumerate(ids):
                    rows.append((
                        vector_id,
                        embeddings[i],
                        contents[i],
                        json.dumps(metadata[i])
                    ))

                # Batch insert with ON CONFLICT handling
                await conn.executemany(f"""
                    INSERT INTO {self.table_name} (id, embedding, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW();
                """, rows)

            return True
        except Exception as e:
            print(f"Error adding vectors to PostgreSQL: {e}")
            return False

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search PostgreSQL for similar vectors."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Build WHERE clause for filters
                where_clause = ""
                params = [query_embedding, top_k]
                param_count = 2

                if filters:
                    filter_conditions = []
                    for key, value in filters.items():
                        param_count += 1
                        filter_conditions.append(f"metadata->>'{key}' = ${param_count}")
                        params.append(str(value))

                    if filter_conditions:
                        where_clause = f"WHERE {' AND '.join(filter_conditions)}"

                # Build score threshold condition
                having_clause = ""
                if score_threshold:
                    param_count += 1
                    having_clause = f"HAVING (1 - (embedding <=> ${1})) >= ${param_count}"
                    params.append(score_threshold)

                # Execute similarity search
                query = f"""
                    SELECT
                        id,
                        content,
                        metadata,
                        (1 - (embedding <=> $1)) as similarity_score,
                        embedding
                    FROM {self.table_name}
                    {where_clause}
                    ORDER BY embedding <=> $1
                    LIMIT $2
                """

                if having_clause:
                    # Insert HAVING clause before ORDER BY
                    query = query.replace("ORDER BY", f"{having_clause} ORDER BY")

                rows = await conn.fetch(query, *params)

                search_results = []
                for row in rows:
                    # Parse metadata
                    try:
                        parsed_metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    except:
                        parsed_metadata = {}

                    search_results.append(SearchResult(
                        id=row['id'],
                        content=row['content'],
                        metadata=parsed_metadata,
                        score=float(row['similarity_score']),
                        embedding=list(row['embedding']) if row['embedding'] else None
                    ))

                return search_results
        except Exception as e:
            print(f"Error searching PostgreSQL: {e}")
            return []

    async def get_by_id(self, vector_id: str) -> Optional[SearchResult]:
        """Get a vector by its ID."""
        try:
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow(f"""
                    SELECT id, content, metadata, embedding
                    FROM {self.table_name}
                    WHERE id = $1;
                """, vector_id)

                if row:
                    try:
                        parsed_metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    except:
                        parsed_metadata = {}

                    return SearchResult(
                        id=row['id'],
                        content=row['content'],
                        metadata=parsed_metadata,
                        score=1.0,
                        embedding=list(row['embedding']) if row['embedding'] else None
                    )
        except Exception as e:
            print(f"Error getting vector by ID: {e}")

        return None

    async def delete_by_id(self, vector_id: str) -> bool:
        """Delete a vector by its ID."""
        try:
            async with self.connection_pool.acquire() as conn:
                result = await conn.execute(f"""
                    DELETE FROM {self.table_name} WHERE id = $1;
                """, vector_id)
                return result.split()[-1] != '0'  # Check if any rows were affected
        except Exception as e:
            print(f"Error deleting vector: {e}")
            return False

    async def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete vectors matching the filter criteria."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Build WHERE clause
                filter_conditions = []
                params = []
                for i, (key, value) in enumerate(filters.items(), 1):
                    filter_conditions.append(f"metadata->>'{key}' = ${i}")
                    params.append(str(value))

                if not filter_conditions:
                    return 0

                where_clause = f"WHERE {' AND '.join(filter_conditions)}"

                result = await conn.execute(f"""
                    DELETE FROM {self.table_name} {where_clause};
                """, *params)

                return int(result.split()[-1])  # Number of deleted rows
        except Exception as e:
            print(f"Error deleting by filter: {e}")
            return 0

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors in the collection."""
        try:
            async with self.connection_pool.acquire() as conn:
                if filters:
                    filter_conditions = []
                    params = []
                    for i, (key, value) in enumerate(filters.items(), 1):
                        filter_conditions.append(f"metadata->>'{key}' = ${i}")
                        params.append(str(value))

                    where_clause = f"WHERE {' AND '.join(filter_conditions)}"

                    result = await conn.fetchval(f"""
                        SELECT COUNT(*) FROM {self.table_name} {where_clause};
                    """, *params)
                else:
                    result = await conn.fetchval(f"""
                        SELECT COUNT(*) FROM {self.table_name};
                    """)

                return result or 0
        except Exception as e:
            print(f"Error counting vectors: {e}")
            return 0

    async def clear(self) -> bool:
        """Clear all vectors from the collection."""
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute(f"TRUNCATE TABLE {self.table_name};")
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

    async def close(self) -> None:
        """Close the PostgreSQL connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()

    def supports_feature(self, feature: str) -> bool:
        """Check if PostgreSQL supports a specific feature."""
        pgvector_features = {
            "add_vectors", "search", "get_by_id", "delete_by_id",
            "delete_by_filter", "count", "clear", "metadata_filtering",
            "full_text_search", "sql_queries", "transactions"
        }
        return feature in pgvector_features

    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Hybrid search combining vector similarity and full-text search."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Build WHERE clause for filters
                where_clause = ""
                params = [query_embedding, query_text, alpha, 1.0 - alpha, top_k]
                param_count = 5

                if filters:
                    filter_conditions = []
                    for key, value in filters.items():
                        param_count += 1
                        filter_conditions.append(f"metadata->>'{key}' = ${param_count}")
                        params.append(str(value))

                    if filter_conditions:
                        where_clause = f"WHERE {' AND '.join(filter_conditions)}"

                # Hybrid search combining vector similarity and text search
                query_sql = f"""
                    SELECT
                        id,
                        content,
                        metadata,
                        (
                            $3 * (1 - (embedding <=> $1)) +
                            $4 * ts_rank(to_tsvector('english', content), plainto_tsquery('english', $2))
                        ) as hybrid_score,
                        embedding
                    FROM {self.table_name}
                    {where_clause}
                    ORDER BY hybrid_score DESC
                    LIMIT $5;
                """

                rows = await conn.fetch(query_sql, *params)

                search_results = []
                for row in rows:
                    try:
                        parsed_metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    except:
                        parsed_metadata = {}

                    search_results.append(SearchResult(
                        id=row['id'],
                        content=row['content'],
                        metadata=parsed_metadata,
                        score=float(row['hybrid_score']),
                        embedding=list(row['embedding']) if row['embedding'] else None
                    ))

                return search_results
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            # Fallback to vector search
            return await self.search(query_embedding, top_k, filters)