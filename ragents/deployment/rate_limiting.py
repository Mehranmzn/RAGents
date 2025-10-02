"""Rate limiting and request prioritization for RAGents."""

import asyncio
import time
from typing import Dict, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import logging

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Priority(str, Enum):
    """Request priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    requests_per_minute: int = Field(default=60, description="Max requests per minute per user")
    requests_per_hour: int = Field(default=1000, description="Max requests per hour per user")
    burst_size: int = Field(default=10, description="Burst capacity for short spikes")
    enable_priority_queue: bool = Field(default=True, description="Enable priority queueing")
    redis_url: Optional[str] = Field(default=None, description="Redis URL for distributed rate limiting")
    use_token_bucket: bool = Field(default=True, description="Use token bucket algorithm")


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time for requested tokens."""
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RateLimiter:
    """
    Rate limiter with token bucket algorithm and priority queueing.

    Features:
    - Per-user rate limiting
    - Token bucket algorithm for smooth rate control
    - Priority queueing for critical requests
    - Distributed rate limiting with Redis
    - Burst capacity handling
    """

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter."""
        self.config = config
        self.buckets: Dict[str, TokenBucket] = {}
        self.redis_client: Optional[redis.Redis] = None
        self._initialized = False

        # Priority queue (priority -> list of pending requests)
        self.priority_queue: Dict[Priority, asyncio.Queue] = {
            Priority.CRITICAL: asyncio.Queue(),
            Priority.HIGH: asyncio.Queue(),
            Priority.NORMAL: asyncio.Queue(),
            Priority.LOW: asyncio.Queue(),
        }

        # Request counters for metrics
        self.request_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"minute": 0, "hour": 0})
        self.last_minute_reset: Dict[str, float] = {}
        self.last_hour_reset: Dict[str, float] = {}

    async def initialize(self):
        """Initialize Redis connection if configured."""
        if self.config.redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(
                    self.config.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                await self.redis_client.ping()
                self._initialized = True
                logger.info("Distributed rate limiter initialized with Redis")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis for rate limiting: {e}")
                self._initialized = False
        else:
            self._initialized = True
            logger.info("Local rate limiter initialized")

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

    def _get_bucket(self, user_id: str) -> TokenBucket:
        """Get or create token bucket for user."""
        if user_id not in self.buckets:
            # Create bucket with per-minute rate
            refill_rate = self.config.requests_per_minute / 60.0  # tokens per second
            self.buckets[user_id] = TokenBucket(
                capacity=self.config.burst_size,
                refill_rate=refill_rate
            )
        return self.buckets[user_id]

    async def _check_redis_limit(self, user_id: str) -> tuple[bool, Optional[float]]:
        """Check rate limit using Redis (distributed)."""
        if not self.redis_client:
            return True, None

        try:
            now = int(time.time())
            minute_key = f"ratelimit:{user_id}:minute:{now // 60}"
            hour_key = f"ratelimit:{user_id}:hour:{now // 3600}"

            # Check minute limit
            minute_count = await self.redis_client.get(minute_key)
            if minute_count and int(minute_count) >= self.config.requests_per_minute:
                wait_time = 60 - (now % 60)
                return False, wait_time

            # Check hour limit
            hour_count = await self.redis_client.get(hour_key)
            if hour_count and int(hour_count) >= self.config.requests_per_hour:
                wait_time = 3600 - (now % 3600)
                return False, wait_time

            # Increment counters
            pipeline = self.redis_client.pipeline()
            pipeline.incr(minute_key)
            pipeline.expire(minute_key, 60)
            pipeline.incr(hour_key)
            pipeline.expire(hour_key, 3600)
            await pipeline.execute()

            return True, None

        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            return True, None  # Allow request on Redis failure

    async def check_rate_limit(
        self,
        user_id: str,
        priority: Priority = Priority.NORMAL
    ) -> tuple[bool, Optional[float]]:
        """
        Check if request is allowed under rate limits.

        Args:
            user_id: User identifier
            priority: Request priority level

        Returns:
            Tuple of (allowed, wait_time_seconds)
        """
        # Use Redis for distributed rate limiting if available
        if self.redis_client:
            return await self._check_redis_limit(user_id)

        # Local token bucket algorithm
        bucket = self._get_bucket(user_id)

        if bucket.consume(1):
            return True, None
        else:
            wait_time = bucket.wait_time(1)
            return False, wait_time

    async def acquire(
        self,
        user_id: str,
        priority: Priority = Priority.NORMAL,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire permission to process request.

        Args:
            user_id: User identifier
            priority: Request priority
            timeout: Maximum wait time in seconds

        Returns:
            True if acquired, False if timeout
        """
        start_time = time.time()

        while True:
            allowed, wait_time = await self.check_rate_limit(user_id, priority)

            if allowed:
                return True

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

                # Adjust wait time based on remaining timeout
                if wait_time:
                    wait_time = min(wait_time, timeout - elapsed)

            # Wait before retrying
            if wait_time:
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0.1)  # Small delay before retry

    async def enqueue_request(
        self,
        user_id: str,
        request_data: Any,
        priority: Priority = Priority.NORMAL
    ):
        """Enqueue a request with priority."""
        if not self.config.enable_priority_queue:
            raise ValueError("Priority queueing is disabled")

        await self.priority_queue[priority].put((user_id, request_data))

    async def dequeue_request(self) -> Optional[tuple[str, Any, Priority]]:
        """Dequeue next request based on priority."""
        if not self.config.enable_priority_queue:
            raise ValueError("Priority queueing is disabled")

        # Check queues in priority order
        for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
            queue = self.priority_queue[priority]
            if not queue.empty():
                user_id, request_data = await queue.get()
                return user_id, request_data, priority

        return None

    async def process_with_rate_limit(
        self,
        user_id: str,
        func: Callable,
        *args,
        priority: Priority = Priority.NORMAL,
        timeout: float = 30.0,
        **kwargs
    ) -> Any:
        """
        Process a function with rate limiting.

        Args:
            user_id: User identifier
            func: Async function to execute
            args: Function arguments
            priority: Request priority
            timeout: Maximum wait time
            kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            TimeoutError: If rate limit wait exceeds timeout
        """
        # Try to acquire rate limit
        acquired = await self.acquire(user_id, priority, timeout)

        if not acquired:
            raise TimeoutError(f"Rate limit timeout for user {user_id}")

        # Execute function
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        if user_id:
            bucket = self.buckets.get(user_id)
            if bucket:
                return {
                    "user_id": user_id,
                    "available_tokens": bucket.tokens,
                    "capacity": bucket.capacity,
                    "refill_rate": bucket.refill_rate,
                }
            return {"error": "User not found"}

        # Global stats
        return {
            "total_users": len(self.buckets),
            "active_queues": {
                priority.value: queue.qsize()
                for priority, queue in self.priority_queue.items()
            },
            "distributed": self.redis_client is not None
        }


# Singleton instance
_rate_limiter_instance: Optional[RateLimiter] = None


async def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter_instance

    if _rate_limiter_instance is None:
        _rate_limiter_instance = RateLimiter(config or RateLimitConfig())
        await _rate_limiter_instance.initialize()

    return _rate_limiter_instance
