"""Distributed checkpointing for LangGraph agents using Redis and PostgreSQL."""

import json
import pickle
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
import logging

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import asyncpg
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CheckpointConfig(BaseModel):
    """Configuration for distributed checkpointing."""

    backend: str = Field(default="redis", description="Backend type: redis or postgres")
    redis_url: str = Field(default="redis://localhost:6379/1", description="Redis URL for checkpoints")
    postgres_url: str = Field(default="postgresql://user:pass@localhost:5432/checkpoints", description="PostgreSQL URL")
    ttl_seconds: int = Field(default=3600, description="TTL for checkpoints (1 hour default)")
    enable_compression: bool = Field(default=True, description="Enable checkpoint compression")
    max_checkpoints_per_thread: int = Field(default=10, description="Max checkpoints to keep per thread")


class RedisCheckpointSaver(BaseCheckpointSaver):
    """
    Redis-based checkpoint saver for LangGraph agents.

    Fast, distributed checkpoint storage for conversation state.
    Ideal for high-throughput scenarios with automatic TTL.
    """

    def __init__(self, config: CheckpointConfig):
        """Initialize Redis checkpoint saver."""
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self._initialized = False

        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                decode_responses=False,
                socket_connect_timeout=5
            )
            await self.redis_client.ping()
            self._initialized = True
            logger.info("Redis checkpoint saver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis checkpoint saver: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self._initialized = False

    def _get_key(self, thread_id: str, checkpoint_id: Optional[str] = None) -> str:
        """Generate Redis key for checkpoint."""
        if checkpoint_id:
            return f"checkpoint:{thread_id}:{checkpoint_id}"
        return f"checkpoint:{thread_id}:latest"

    def _get_list_key(self, thread_id: str) -> str:
        """Generate Redis key for checkpoint list."""
        return f"checkpoint_list:{thread_id}"

    def _serialize(self, checkpoint: Checkpoint) -> bytes:
        """Serialize checkpoint data."""
        data = {
            "v": checkpoint.v,
            "id": checkpoint.id,
            "ts": checkpoint.ts,
            "channel_values": checkpoint.channel_values,
            "channel_versions": checkpoint.channel_versions,
            "versions_seen": checkpoint.versions_seen,
        }

        if self.config.enable_compression:
            import zlib
            return zlib.compress(pickle.dumps(data))
        return pickle.dumps(data)

    def _deserialize(self, data: bytes) -> Checkpoint:
        """Deserialize checkpoint data."""
        if self.config.enable_compression:
            import zlib
            data = zlib.decompress(data)

        checkpoint_dict = pickle.loads(data)
        return Checkpoint(**checkpoint_dict)

    async def aget(self, config: Dict[str, Any]) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a thread."""
        if not self._initialized:
            await self.initialize()

        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return None

        try:
            key = self._get_key(thread_id)
            data = await self.redis_client.get(key)

            if data:
                return self._deserialize(data)
            return None

        except Exception as e:
            logger.error(f"Error getting checkpoint: {e}")
            return None

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save a checkpoint."""
        if not self._initialized:
            await self.initialize()

        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError("thread_id is required in config")

        try:
            # Serialize checkpoint
            serialized = self._serialize(checkpoint)

            # Save as latest
            latest_key = self._get_key(thread_id)
            await self.redis_client.setex(
                latest_key,
                self.config.ttl_seconds,
                serialized
            )

            # Save with checkpoint ID
            checkpoint_key = self._get_key(thread_id, checkpoint.id)
            await self.redis_client.setex(
                checkpoint_key,
                self.config.ttl_seconds,
                serialized
            )

            # Add to checkpoint list (for history)
            list_key = self._get_list_key(thread_id)
            await self.redis_client.lpush(list_key, checkpoint.id)
            await self.redis_client.ltrim(list_key, 0, self.config.max_checkpoints_per_thread - 1)
            await self.redis_client.expire(list_key, self.config.ttl_seconds)

            logger.debug(f"Checkpoint saved for thread {thread_id}")
            return config

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise

    async def alist(self, config: Dict[str, Any]) -> list:
        """List all checkpoints for a thread."""
        if not self._initialized:
            await self.initialize()

        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return []

        try:
            list_key = self._get_list_key(thread_id)
            checkpoint_ids = await self.redis_client.lrange(list_key, 0, -1)

            checkpoints = []
            for checkpoint_id in checkpoint_ids:
                checkpoint_id = checkpoint_id.decode('utf-8')
                key = self._get_key(thread_id, checkpoint_id)
                data = await self.redis_client.get(key)

                if data:
                    checkpoint = self._deserialize(data)
                    checkpoints.append((config, checkpoint, {}))

            return checkpoints

        except Exception as e:
            logger.error(f"Error listing checkpoints: {e}")
            return []


class PostgresCheckpointSaver(BaseCheckpointSaver):
    """
    PostgreSQL-based checkpoint saver for LangGraph agents.

    Persistent, ACID-compliant checkpoint storage.
    Ideal for production deployments requiring durability.
    """

    def __init__(self, config: CheckpointConfig):
        """Initialize PostgreSQL checkpoint saver."""
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False

        if not POSTGRES_AVAILABLE:
            raise ImportError("asyncpg not available. Install with: pip install asyncpg")

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.config.postgres_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )

            # Create table if not exists
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_id TEXT NOT NULL,
                        checkpoint_data BYTEA NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        PRIMARY KEY (thread_id, checkpoint_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_thread_created
                    ON langgraph_checkpoints(thread_id, created_at DESC);
                """)

            self._initialized = True
            logger.info("PostgreSQL checkpoint saver initialized")

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL checkpoint saver: {e}")
            raise

    async def close(self) -> None:
        """Close PostgreSQL connection pool."""
        if self.pool:
            await self.pool.close()
            self._initialized = False

    def _serialize(self, checkpoint: Checkpoint) -> bytes:
        """Serialize checkpoint data."""
        data = {
            "v": checkpoint.v,
            "id": checkpoint.id,
            "ts": checkpoint.ts,
            "channel_values": checkpoint.channel_values,
            "channel_versions": checkpoint.channel_versions,
            "versions_seen": checkpoint.versions_seen,
        }

        if self.config.enable_compression:
            import zlib
            return zlib.compress(pickle.dumps(data))
        return pickle.dumps(data)

    def _deserialize(self, data: bytes) -> Checkpoint:
        """Deserialize checkpoint data."""
        if self.config.enable_compression:
            import zlib
            data = zlib.decompress(data)

        checkpoint_dict = pickle.loads(data)
        return Checkpoint(**checkpoint_dict)

    async def aget(self, config: Dict[str, Any]) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a thread."""
        if not self._initialized:
            await self.initialize()

        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return None

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT checkpoint_data
                    FROM langgraph_checkpoints
                    WHERE thread_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                """, thread_id)

                if row:
                    return self._deserialize(bytes(row['checkpoint_data']))
                return None

        except Exception as e:
            logger.error(f"Error getting checkpoint: {e}")
            return None

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save a checkpoint."""
        if not self._initialized:
            await self.initialize()

        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError("thread_id is required in config")

        try:
            serialized = self._serialize(checkpoint)
            metadata_json = json.dumps(metadata)

            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO langgraph_checkpoints
                    (thread_id, checkpoint_id, checkpoint_data, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (thread_id, checkpoint_id)
                    DO UPDATE SET
                        checkpoint_data = EXCLUDED.checkpoint_data,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW()
                """, thread_id, checkpoint.id, serialized, metadata_json)

                # Clean up old checkpoints
                await conn.execute("""
                    DELETE FROM langgraph_checkpoints
                    WHERE thread_id = $1
                    AND checkpoint_id NOT IN (
                        SELECT checkpoint_id
                        FROM langgraph_checkpoints
                        WHERE thread_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                    )
                """, thread_id, self.config.max_checkpoints_per_thread)

            logger.debug(f"Checkpoint saved for thread {thread_id}")
            return config

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise

    async def alist(self, config: Dict[str, Any]) -> list:
        """List all checkpoints for a thread."""
        if not self._initialized:
            await self.initialize()

        thread_id = config.get("configurable", {}).get("thread_id")
        if not thread_id:
            return []

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT checkpoint_data, metadata
                    FROM langgraph_checkpoints
                    WHERE thread_id = $1
                    ORDER BY created_at DESC
                """, thread_id)

                checkpoints = []
                for row in rows:
                    checkpoint = self._deserialize(bytes(row['checkpoint_data']))
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    checkpoints.append((config, checkpoint, metadata))

                return checkpoints

        except Exception as e:
            logger.error(f"Error listing checkpoints: {e}")
            return []


async def create_checkpoint_saver(config: CheckpointConfig) -> BaseCheckpointSaver:
    """
    Factory function to create appropriate checkpoint saver.

    Args:
        config: CheckpointConfig with backend type and connection details

    Returns:
        Initialized checkpoint saver (Redis or PostgreSQL)
    """
    if config.backend == "redis":
        saver = RedisCheckpointSaver(config)
    elif config.backend == "postgres":
        saver = PostgresCheckpointSaver(config)
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")

    await saver.initialize()
    return saver
