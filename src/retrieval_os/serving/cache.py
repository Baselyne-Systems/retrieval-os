"""Semantic query cache backed by Redis.

Cache key = SHA-256( plan_name | version_num | query_text | top_k )
Value      = JSON-serialised list[RetrievedChunk]
TTL        = plan's cache_ttl_seconds (0 = disabled)
"""

from __future__ import annotations

import hashlib
import json
import logging

from retrieval_os.core.redis_client import get_redis

log = logging.getLogger(__name__)

_PREFIX = "ros:qcache:"


def _cache_key(plan_name: str, version: int, query: str, top_k: int) -> str:
    raw = f"{plan_name}|{version}|{query}|{top_k}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"{_PREFIX}{digest}"


async def cache_get(
    plan_name: str, version: int, query: str, top_k: int
) -> list[dict] | None:
    """Return cached chunks or None on miss."""
    key = _cache_key(plan_name, version, query, top_k)
    redis = await get_redis()
    try:
        raw = await redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        log.warning("cache.get_error", extra={"key": key})
        return None


async def cache_set(
    plan_name: str,
    version: int,
    query: str,
    top_k: int,
    chunks: list[dict],
    ttl_seconds: int,
) -> None:
    """Store chunks; skip if ttl_seconds == 0."""
    if ttl_seconds <= 0:
        return
    key = _cache_key(plan_name, version, query, top_k)
    redis = await get_redis()
    try:
        await redis.set(key, json.dumps(chunks, default=str), ex=ttl_seconds)
    except Exception:
        log.warning("cache.set_error", extra={"key": key})


async def cache_invalidate_plan(plan_name: str) -> int:
    """Delete all cached entries for a plan (used on new deployment).

    Returns the number of keys deleted.
    """
    redis = await get_redis()
    pattern = f"{_PREFIX}*"
    # Scan instead of KEYS to avoid blocking Redis on large keyspaces.
    deleted = 0
    async for key in redis.scan_iter(pattern):
        await redis.delete(key)
        deleted += 1
    return deleted
