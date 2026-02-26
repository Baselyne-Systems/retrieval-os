"""Benchmark: Cost Efficiency — cache hits eliminate embedding and vector-search costs.

Customer claims
---------------
1. On a cache hit, no embedding API call and no vector-database query is made —
   the entire pipeline is short-circuited at the Redis layer.
2. Setting cache_ttl_seconds=0 fully disables cache writes — useful when a team
   wants to guarantee fresh results for every query without permanently disabling
   the feature.
3. Setting cache_enabled=False skips both the cache-read and the cache-write
   phases, so tokens are billed for every query but results are always fresh.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from retrieval_os.serving.cache import cache_set
from retrieval_os.serving.executor import execute_retrieval

# ── Shared fixtures ───────────────────────────────────────────────────────────


def _base_kwargs(**overrides: object) -> dict:
    return {
        "project_name": "my-docs",
        "version": 1,
        "query": "what is retrieval-augmented generation?",
        "embedding_provider": "sentence_transformers",
        "embedding_model": "BAAI/bge-m3",
        "embedding_normalize": True,
        "embedding_batch_size": 32,
        "index_backend": "qdrant",
        "index_collection": "docs_v1",
        "distance_metric": "cosine",
        "top_k": 10,
        "reranker": None,
        "rerank_top_k": None,
        "metadata_filters": None,
        "cache_enabled": True,
        "cache_ttl_seconds": 3600,
        **overrides,
    }


_CACHED_CHUNKS = [
    {
        "id": "doc-1",
        "score": 0.95,
        "text": "RAG combines retrieval with generation.",
        "metadata": {},
    },
    {
        "id": "doc-2",
        "score": 0.88,
        "text": "Retrieval augmentation reduces hallucination.",
        "metadata": {},
    },
]


# ── Cache hit eliminates external API calls ───────────────────────────────────


class TestCacheHitSkipsExternalCalls:
    @pytest.mark.asyncio
    async def test_cache_hit_never_calls_embed_text(self) -> None:
        """On a cache hit, the embedding provider is never invoked."""
        embed_mock = AsyncMock()
        vector_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.serving.executor.cache_get",
                new=AsyncMock(return_value=_CACHED_CHUNKS),
            ),
            patch("retrieval_os.serving.executor.embed_text", new=embed_mock),
            patch("retrieval_os.serving.executor.vector_search", new=vector_mock),
        ):
            chunks, cache_hit = await execute_retrieval(**_base_kwargs())

        assert cache_hit is True
        embed_mock.assert_not_called()
        vector_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_hit_returns_correct_chunks(self) -> None:
        """Cached chunks are returned exactly as stored — no data loss."""
        with (
            patch(
                "retrieval_os.serving.executor.cache_get",
                new=AsyncMock(return_value=_CACHED_CHUNKS),
            ),
            patch("retrieval_os.serving.executor.embed_text", new=AsyncMock()),
            patch("retrieval_os.serving.executor.vector_search", new=AsyncMock()),
        ):
            chunks, cache_hit = await execute_retrieval(**_base_kwargs())

        assert cache_hit is True
        assert len(chunks) == len(_CACHED_CHUNKS)
        assert chunks[0].id == "doc-1"
        assert chunks[0].score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_cache_miss_calls_embed_and_vector_search(self) -> None:
        """On a cache miss, both embed and vector-search are invoked."""
        embed_mock = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        vector_mock = AsyncMock(return_value=[])

        with (
            patch("retrieval_os.serving.executor.cache_get", new=AsyncMock(return_value=None)),
            patch("retrieval_os.serving.executor.embed_text", new=embed_mock),
            patch("retrieval_os.serving.executor.vector_search", new=vector_mock),
            patch("retrieval_os.serving.executor.cache_set", new=AsyncMock()),
        ):
            chunks, cache_hit = await execute_retrieval(**_base_kwargs())

        assert cache_hit is False
        embed_mock.assert_called_once()
        vector_mock.assert_called_once()


# ── TTL=0 disables cache writes ───────────────────────────────────────────────


class TestTTLZeroDisablesCacheWrite:
    @pytest.mark.asyncio
    async def test_ttl_zero_skips_redis_set(self) -> None:
        """cache_set with ttl_seconds=0 must not write to Redis."""
        redis_mock = AsyncMock()
        redis_mock.set = AsyncMock()

        with patch("retrieval_os.serving.cache.get_redis", new=AsyncMock(return_value=redis_mock)):
            await cache_set("my-docs", 1, "test query", 10, _CACHED_CHUNKS, ttl_seconds=0)

        redis_mock.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_ttl_zero_in_execute_retrieval_passes_zero_to_cache_set(self) -> None:
        """End-to-end: cache_ttl_seconds=0 is forwarded to cache_set as ttl_seconds=0.

        cache_set is responsible for the skip-write logic when ttl <= 0; the
        executor delegates this decision by passing the exact ttl value through.
        """
        cache_set_mock = AsyncMock()

        with (
            patch("retrieval_os.serving.executor.cache_get", new=AsyncMock(return_value=None)),
            patch(
                "retrieval_os.serving.executor.embed_text", new=AsyncMock(return_value=[[0.1, 0.2]])
            ),
            patch("retrieval_os.serving.executor.vector_search", new=AsyncMock(return_value=[])),
            patch("retrieval_os.serving.executor.cache_set", new=cache_set_mock),
        ):
            await execute_retrieval(**_base_kwargs(cache_ttl_seconds=0))

        # cache_set must be called with ttl_seconds=0 so it can apply the skip logic
        cache_set_mock.assert_called_once()
        assert cache_set_mock.call_args.kwargs["ttl_seconds"] == 0


# ── cache_enabled=False bypass ────────────────────────────────────────────────


class TestCacheDisabledSkipsBothReadAndWrite:
    @pytest.mark.asyncio
    async def test_cache_disabled_skips_cache_get(self) -> None:
        """When cache_enabled=False, the cache lookup is skipped entirely."""
        cache_get_mock = AsyncMock()

        with (
            patch("retrieval_os.serving.executor.cache_get", new=cache_get_mock),
            patch("retrieval_os.serving.executor.embed_text", new=AsyncMock(return_value=[[0.1]])),
            patch("retrieval_os.serving.executor.vector_search", new=AsyncMock(return_value=[])),
            patch("retrieval_os.serving.executor.cache_set", new=AsyncMock()),
        ):
            _, cache_hit = await execute_retrieval(**_base_kwargs(cache_enabled=False))

        assert cache_hit is False
        cache_get_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_disabled_skips_cache_set(self) -> None:
        """When cache_enabled=False, no cache entry is written after retrieval."""
        cache_set_mock = AsyncMock()

        with (
            patch("retrieval_os.serving.executor.cache_get", new=AsyncMock()),
            patch("retrieval_os.serving.executor.embed_text", new=AsyncMock(return_value=[[0.1]])),
            patch("retrieval_os.serving.executor.vector_search", new=AsyncMock(return_value=[])),
            patch("retrieval_os.serving.executor.cache_set", new=cache_set_mock),
        ):
            await execute_retrieval(**_base_kwargs(cache_enabled=False))

        cache_set_mock.assert_not_called()
