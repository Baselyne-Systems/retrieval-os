"""Load test: Cache eliminates Qdrant load on repeated queries.

Proves:
1. Warm Qdrant calls == 0 (cache fully absorbs repeated queries).
2. Warm QPS ≥ 5× cold QPS (cache is substantially faster than Qdrant).

Infrastructure required: Postgres + Redis + Qdrant.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import retrieval_os.serving.index_proxy as index_proxy
from retrieval_os.core.redis_client import get_redis
from retrieval_os.serving.executor import execute_retrieval
from tests.load.conftest import random_unit_vector


async def _clear_query_cache() -> None:
    redis = await get_redis()
    async for key in redis.scan_iter("ros:qcache:*"):
        await redis.delete(key)


def _exec_kwargs(project_name: str, collection: str, query: str) -> dict:
    return dict(
        project_name=project_name,
        version=1,
        query=query,
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        embedding_normalize=True,
        embedding_batch_size=32,
        index_backend="qdrant",
        index_collection=collection,
        distance_metric="cosine",
        top_k=10,
        reranker=None,
        rerank_top_k=None,
        metadata_filters=None,
        cache_enabled=True,
        cache_ttl_seconds=3600,
    )


class TestCacheEliminatesQdrantLoad:
    """Cache hit path must bypass Qdrant entirely."""

    async def test_cold_vs_warm_qdrant_call_ratio(
        self, load_project, load_collection, record_load
    ) -> None:
        """Warm cache must result in 0 Qdrant calls and ≥ 5× higher QPS.

        Cold: 50 unique queries → 50 Qdrant searches, results cached.
        Warm: same 50 queries again → 0 Qdrant searches, served from Redis.
        """
        stub_vector = random_unit_vector()
        n_queries = 50
        queries = [f"cache efficiency query {i}" for i in range(n_queries)]

        await _clear_query_cache()

        qdrant_cold_calls = 0
        qdrant_warm_calls = 0
        _original_search = index_proxy._qdrant_search

        async def counting_search_cold(**kwargs):
            nonlocal qdrant_cold_calls
            qdrant_cold_calls += 1
            return await _original_search(**kwargs)

        async def counting_search_warm(**kwargs):
            nonlocal qdrant_warm_calls
            qdrant_warm_calls += 1
            return await _original_search(**kwargs)

        # ── Cold pass ─────────────────────────────────────────────────────────
        with (
            patch(
                "retrieval_os.serving.executor.embed_text",
                new=AsyncMock(return_value=[stub_vector]),
            ),
            patch.object(index_proxy, "_qdrant_search", side_effect=counting_search_cold),
        ):
            cold_start = time.perf_counter()
            for q in queries:
                _, hit = await execute_retrieval(**_exec_kwargs(load_project, load_collection, q))
                assert not hit, f"Unexpected cache hit on cold query: {q}"
            cold_elapsed = time.perf_counter() - cold_start

        cold_qps = n_queries / cold_elapsed if cold_elapsed > 0 else 0

        record_load(
            "Cache cold pass — 50 unique queries",
            samples=[cold_elapsed / n_queries * 1000] * n_queries,
            qps=cold_qps,
            note="embed+Qdrant+Redis SET per query",
        )

        # ── Warm pass ─────────────────────────────────────────────────────────
        with (
            patch(
                "retrieval_os.serving.executor.embed_text",
                new=AsyncMock(return_value=[stub_vector]),
            ),
            patch.object(index_proxy, "_qdrant_search", side_effect=counting_search_warm),
        ):
            warm_start = time.perf_counter()
            for q in queries:
                _, hit = await execute_retrieval(**_exec_kwargs(load_project, load_collection, q))
                assert hit, f"Expected cache hit on warm query: {q}"
            warm_elapsed = time.perf_counter() - warm_start

        warm_qps = n_queries / warm_elapsed if warm_elapsed > 0 else 0

        record_load(
            "Cache warm pass — 50 repeated queries",
            samples=[warm_elapsed / n_queries * 1000] * n_queries,
            qps=warm_qps,
            note="Redis GET only, Qdrant bypassed",
        )

        assert qdrant_warm_calls == 0, (
            f"Expected 0 Qdrant calls on warm pass; got {qdrant_warm_calls}. "
            "Cache is not absorbing repeated queries."
        )
        assert warm_qps >= 5 * cold_qps, (
            f"Warm QPS ({warm_qps:.0f}) should be ≥ 5× cold QPS ({cold_qps:.0f}). "
            "Cache speedup is lower than expected."
        )
