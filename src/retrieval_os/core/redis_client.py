"""Async Redis connection pool and helpers."""

import redis.asyncio as aioredis

from retrieval_os.core.config import settings

_pool: aioredis.ConnectionPool | None = None
_client: aioredis.Redis | None = None  # type: ignore[type-arg]


def get_redis_pool() -> aioredis.ConnectionPool:
    global _pool
    if _pool is None:
        _pool = aioredis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_pool_size,
            socket_timeout=settings.redis_socket_timeout,
            decode_responses=True,
        )
    return _pool


def get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    global _client
    if _client is None:
        _client = aioredis.Redis(connection_pool=get_redis_pool())
    return _client


async def check_redis_connection() -> bool:
    """Returns True if Redis is reachable."""
    try:
        await get_redis().ping()
        return True
    except Exception:
        return False


async def close_redis() -> None:
    global _pool, _client
    if _client:
        await _client.aclose()
        _client = None
    if _pool:
        await _pool.aclose()
        _pool = None
