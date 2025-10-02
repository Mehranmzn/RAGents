"""Caching layer for RAG operations to improve performance and reduce costs."""

import asyncio
import hashlib
import json
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Configuration for caching layer."""

    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    embedding_ttl: int = Field(default=86400, description="TTL for embedding cache (24 hours)")
    retrieval_ttl: int = Field(default=3600, description="TTL for retrieval cache (1 hour)")
    query_rewrite_ttl: int = Field(default=7200, description="TTL for query rewrite cache (2 hours)")
    enable_compression: bool = Field(default=True, description="Enable compression for cached data")
    max_cache_size_mb: int = Field(default=1000, description="Maximum cache size in MB")
    enable_stats: bool = Field(default=True, description="Enable cache statistics")


class CacheStats(BaseModel):
    """Cache statistics for monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    cache_size_mb: float = 0.0
    hit_rate: float = 0.0

    def update_hit_rate(self):
        """Calculate hit rate percentage."""
        if self.total_requests > 0:
            self.hit_rate = (self.hits / self.total_requests) * 100


class RAGCache:
    """
    Distributed caching layer for RAG operations.

    Caches:
    - Text embeddings
    - Retrieval results
    - Query rewrites
    - LLM responses (optional)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize the cache with configuration."""
        self.config = config or CacheConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.stats = CacheStats()
        self._initialized = False

        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Install with: pip install redis")

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, caching disabled")
            return

        try:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5
            )
            # Test connection
            await self.redis_client.ping()
            self._initialized = True
            logger.info("RAG cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self.redis_client = None

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self._initialized = False

    def _generate_key(self, prefix: str, content: str) -> str:
        """Generate a cache key from content."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"{prefix}:{content_hash}"

    def _serialize(self, data: Any) -> bytes:
        """Serialize data for caching."""
        if self.config.enable_compression:
            import zlib
            return zlib.compress(pickle.dumps(data))
        return pickle.dumps(data)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize cached data."""
        if self.config.enable_compression:
            import zlib
            return pickle.loads(zlib.decompress(data))
        return pickle.loads(data)

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        if not self._initialized:
            return None

        key = self._generate_key("embedding", text)
        self.stats.total_requests += 1

        try:
            cached = await self.redis_client.get(key)
            if cached:
                self.stats.hits += 1
                self.stats.update_hit_rate()
                return self._deserialize(cached)
            else:
                self.stats.misses += 1
                self.stats.update_hit_rate()
                return None
        except Exception as e:
            logger.error(f"Error getting cached embedding: {e}")
            return None

    async def cache_embedding(self, text: str, embedding: List[float]) -> bool:
        """Cache an embedding for text."""
        if not self._initialized:
            return False

        key = self._generate_key("embedding", text)

        try:
            serialized = self._serialize(embedding)
            await self.redis_client.setex(
                key,
                self.config.embedding_ttl,
                serialized
            )
            return True
        except Exception as e:
            logger.error(f"Error caching embedding: {e}")
            return False

    async def get_retrieval(self, query: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
        """Get cached retrieval results."""
        if not self._initialized:
            return None

        key = self._generate_key(f"retrieval:{top_k}", query)
        self.stats.total_requests += 1

        try:
            cached = await self.redis_client.get(key)
            if cached:
                self.stats.hits += 1
                self.stats.update_hit_rate()
                return self._deserialize(cached)
            else:
                self.stats.misses += 1
                self.stats.update_hit_rate()
                return None
        except Exception as e:
            logger.error(f"Error getting cached retrieval: {e}")
            return None

    async def cache_retrieval(
        self,
        query: str,
        results: Dict[str, Any],
        top_k: int = 5
    ) -> bool:
        """Cache retrieval results."""
        if not self._initialized:
            return False

        key = self._generate_key(f"retrieval:{top_k}", query)

        try:
            serialized = self._serialize(results)
            await self.redis_client.setex(
                key,
                self.config.retrieval_ttl,
                serialized
            )
            return True
        except Exception as e:
            logger.error(f"Error caching retrieval: {e}")
            return False

    async def get_query_rewrite(self, query: str) -> Optional[str]:
        """Get cached query rewrite."""
        if not self._initialized:
            return None

        key = self._generate_key("rewrite", query)
        self.stats.total_requests += 1

        try:
            cached = await self.redis_client.get(key)
            if cached:
                self.stats.hits += 1
                self.stats.update_hit_rate()
                return cached.decode('utf-8')
            else:
                self.stats.misses += 1
                self.stats.update_hit_rate()
                return None
        except Exception as e:
            logger.error(f"Error getting cached query rewrite: {e}")
            return None

    async def cache_query_rewrite(self, original_query: str, rewritten_query: str) -> bool:
        """Cache query rewrite result."""
        if not self._initialized:
            return False

        key = self._generate_key("rewrite", original_query)

        try:
            await self.redis_client.setex(
                key,
                self.config.query_rewrite_ttl,
                rewritten_query.encode('utf-8')
            )
            return True
        except Exception as e:
            logger.error(f"Error caching query rewrite: {e}")
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        if not self._initialized:
            return 0

        try:
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.stats.evictions += deleted
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Error invalidating cache pattern: {e}")
            return 0

    async def clear_all(self) -> bool:
        """Clear all cache entries."""
        if not self._initialized:
            return False

        try:
            await self.redis_client.flushdb()
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        if self._initialized:
            try:
                # Get cache size from Redis
                info = await self.redis_client.info('memory')
                self.stats.cache_size_mb = info.get('used_memory', 0) / (1024 * 1024)
            except Exception as e:
                logger.error(f"Error getting cache stats: {e}")

        return self.stats

    async def health_check(self) -> Dict[str, Any]:
        """Check cache health status."""
        if not self._initialized or not self.redis_client:
            return {
                "status": "unavailable",
                "message": "Redis not initialized"
            }

        try:
            await self.redis_client.ping()
            stats = await self.get_stats()

            return {
                "status": "healthy",
                "hit_rate": f"{stats.hit_rate:.2f}%",
                "total_requests": stats.total_requests,
                "cache_size_mb": f"{stats.cache_size_mb:.2f}",
                "connected": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False
            }


# Singleton instance
_cache_instance: Optional[RAGCache] = None


async def get_cache(config: Optional[CacheConfig] = None) -> RAGCache:
    """Get or create the global cache instance."""
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = RAGCache(config)
        await _cache_instance.initialize()

    return _cache_instance
