"""Load test: Query latency and throughput.

Measures four points on the serving stack:
1. Pure Qdrant ANN search latency  (no embed, no cache, no Postgres)
2. Full stack cache miss            (embed stub → Qdrant → cache write)
3. Full stack cache hit             (Redis GET → deserialise → return)
4. Concurrent burst throughput      (50 concurrent distinct queries)

Infrastructure required (auto-skipped when any service is unreachable):
  - Qdrant  (10 000-vector collection created by ``load_collection`` fixture)
  - Redis   (semantic query cache)
  - Postgres (project + deployment config)

Embedding is intentionally stubbed with random unit vectors throughout.
Add your model's inference latency (~2–150 ms) for real end-to-end estimates.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

from retrieval_os.core.redis_client import get_redis
from retrieval_os.serving.executor import execute_retrieval
from retrieval_os.serving.index_proxy import vector_search
from tests.load.conftest import random_unit_vector

# ── Helpers ───────────────────────────────────────────────────────────────────


def _exec_kwargs(project_name: str, collection: str, query: str) -> dict:
    """Build execute_retrieval kwargs using config that matches the load fixture."""
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


async def _clear_query_cache() -> None:
    """Delete all query-cache keys so subsequent tests start cold."""
    redis = await get_redis()
    async for key in redis.scan_iter("ros:qcache:*"):
        await redis.delete(key)


# ── Pure Qdrant ANN ───────────────────────────────────────────────────────────


class TestQdrantANNLatency:
    """Direct Qdrant gRPC search — no Redis, no embedding, no Postgres.

    This is the floor latency of the serving stack.  Every query will be *at
    least* this fast, regardless of the embedding model chosen.
    """

    async def test_ann_search_p99_under_50ms(self, load_collection, record_load) -> None:
        """ANN p99 on a 10k-vector collection must be < 50 ms.

        At p99 < 50 ms the downstream app has a > 150 ms latency budget for
        embedding + business logic while still hitting a 200 ms SLA.
        """
        n_queries = 100
        latencies: list[float] = []

        for _ in range(n_queries):
            t0 = time.perf_counter()
            await vector_search(
                backend="qdrant",
                collection=load_collection,
                vector=random_unit_vector(),
                top_k=10,
                distance_metric="cosine",
            )
            latencies.append((time.perf_counter() - t0) * 1000)

        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            "Qdrant ANN (10k vectors, top_k=10)",
            samples=latencies,
            qps=qps,
            note="direct gRPC, no cache/embed/Postgres",
        )

        p99 = sorted(latencies)[int(n_queries * 0.99)]
        assert p99 < 50.0, (
            f"Qdrant ANN p99={p99:.1f} ms on 10k vectors; expected < 50 ms. "
            "Check Qdrant resources or HNSW ef_search setting."
        )

    async def test_ann_search_returns_correct_count(self, load_collection) -> None:
        """Every ANN call returns exactly top_k results from the 10k collection."""
        for top_k in (1, 5, 10):
            hits = await vector_search(
                backend="qdrant",
                collection=load_collection,
                vector=random_unit_vector(),
                top_k=top_k,
                distance_metric="cosine",
            )
            assert len(hits) == top_k, f"Expected {top_k} hits, got {len(hits)}"


# ── Full stack cache miss ─────────────────────────────────────────────────────


class TestFullStackCacheMiss:
    """execute_retrieval on a cold cache — embed stub → Qdrant → cache write."""

    async def test_cache_miss_p95_under_100ms(
        self, load_project, load_collection, record_load
    ) -> None:
        """Full stack cache miss p95 must be < 100 ms.

        Measured path: stub-embed (≈0 ms) → Qdrant ANN → Redis SET.
        100 ms p95 leaves a comfortable margin for a real embedding model
        (add ~5–50 ms for sentence-transformers) to meet a 200 ms SLA.
        """
        n_queries = 50
        latencies: list[float] = []
        stub_vector = random_unit_vector()

        await _clear_query_cache()

        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            for i in range(n_queries):
                # Unique query string → guaranteed cache miss every time
                kwargs = _exec_kwargs(load_project, load_collection, f"unique load query {i}")
                t0 = time.perf_counter()
                _, cache_hit = await execute_retrieval(**kwargs)
                latencies.append((time.perf_counter() - t0) * 1000)
                assert not cache_hit, f"Unexpected cache hit on query {i}"

        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            "Full stack cache miss (embed stub → Qdrant → Redis SET)",
            samples=latencies,
            qps=qps,
            note="50 unique queries, embed stubbed",
        )

        p95 = sorted(latencies)[int(n_queries * 0.95)]
        assert p95 < 100.0, (
            f"Full stack cache miss p95={p95:.1f} ms; expected < 100 ms. "
            "Check Qdrant and Redis latency."
        )

    async def test_cache_miss_returns_valid_results(self, load_project, load_collection) -> None:
        """Cache miss path returns 10 results with valid cosine scores."""
        stub_vector = random_unit_vector()
        await _clear_query_cache()

        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            chunks, cache_hit = await execute_retrieval(
                **_exec_kwargs(load_project, load_collection, "sanity check query abc123")
            )

        assert not cache_hit
        assert len(chunks) == 10
        # Cosine scores should be in [-1, 1]; all should be finite
        assert all(-1.1 <= c.score <= 1.1 for c in chunks)


# ── Full stack cache hit ──────────────────────────────────────────────────────


class TestFullStackCacheHit:
    """execute_retrieval on a warm cache — Redis GET → deserialise → return."""

    async def test_cache_hit_p99_under_20ms(
        self, load_project, load_collection, record_load
    ) -> None:
        """Cached query p99 must be < 20 ms.

        Cache hit path skips embedding and Qdrant entirely.  Latency is
        dominated by Redis RTT (~0.5–2 ms localhost) and JSON deserialisation.
        20 ms p99 leaves headroom for network and application overhead.
        """
        n_queries = 100
        stub_vector = random_unit_vector()
        warm_query = "what is retrieval-augmented generation for enterprise search?"
        kwargs = _exec_kwargs(load_project, load_collection, warm_query)

        await _clear_query_cache()

        # Warm the cache with one real retrieval
        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            await execute_retrieval(**kwargs)

        # Measure only the cache hit path (embed not called)
        latencies: list[float] = []
        for _ in range(n_queries):
            t0 = time.perf_counter()
            _, cache_hit = await execute_retrieval(**kwargs)
            latencies.append((time.perf_counter() - t0) * 1000)
            assert cache_hit, "Expected cache hit on repeated query"

        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            "Full stack cache hit (Redis GET → JSON deserialise)",
            samples=latencies,
            qps=qps,
            note="100 identical queries after cache warm",
        )

        p99 = sorted(latencies)[int(n_queries * 0.99)]
        assert p99 < 20.0, (
            f"Cache hit p99={p99:.1f} ms; expected < 20 ms. "
            "Check Redis latency or result payload size."
        )


# ── Concurrent burst throughput ───────────────────────────────────────────────


class TestConcurrentThroughput:
    """Concurrent query bursts — wall-clock QPS under realistic concurrency."""

    async def test_50_concurrent_cache_miss_queries_qps_above_50(
        self, load_project, load_collection, record_load
    ) -> None:
        """50 concurrent distinct queries (all cache misses) must exceed 50 QPS.

        This simulates a burst of novel questions hitting a freshly deployed
        project.  50 QPS on a single-node Qdrant over localhost is a conservative
        floor; production deployments typically exceed 200 QPS on the miss path.
        """
        n_concurrent = 50
        stub_vector = random_unit_vector()

        await _clear_query_cache()

        async def _one_query(i: int) -> float:
            kwargs = _exec_kwargs(
                load_project, load_collection, f"concurrent burst query unique {i}"
            )
            t0 = time.perf_counter()
            await execute_retrieval(**kwargs)
            return (time.perf_counter() - t0) * 1000

        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            wall_start = time.perf_counter()
            latencies = await asyncio.gather(*[_one_query(i) for i in range(n_concurrent)])
            wall_elapsed = time.perf_counter() - wall_start

        qps = n_concurrent / wall_elapsed

        record_load(
            "Concurrent burst — 50 distinct queries (cache miss)",
            samples=list(latencies),
            qps=qps,
            note="50 concurrent, all cache misses",
        )

        assert qps >= 50.0, (
            f"Concurrent cache-miss throughput = {qps:.0f} QPS; expected >= 50 QPS. "
            "Check Qdrant gRPC connection pool or asyncio event loop contention."
        )

    async def test_50_concurrent_cached_queries_qps_above_500(
        self, load_project, load_collection, record_load
    ) -> None:
        """50 concurrent identical queries (all cache hits) must exceed 500 QPS.

        Cache hit path: Redis GET + JSON deserialise.  At < 1 ms per query on
        localhost, 50 concurrent requests should complete in < 100 ms (> 500 QPS).
        This ceiling validates that Redis is not a throughput bottleneck.
        """
        stub_vector = random_unit_vector()
        warm_query = "shared cached query for concurrent throughput load test"
        kwargs = _exec_kwargs(load_project, load_collection, warm_query)

        await _clear_query_cache()

        # Warm the cache
        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            await execute_retrieval(**kwargs)

        async def _cached_query() -> float:
            t0 = time.perf_counter()
            _, cache_hit = await execute_retrieval(**kwargs)
            assert cache_hit, "Expected cache hit"
            return (time.perf_counter() - t0) * 1000

        wall_start = time.perf_counter()
        latencies = await asyncio.gather(*[_cached_query() for _ in range(50)])
        wall_elapsed = time.perf_counter() - wall_start

        qps = 50 / wall_elapsed

        record_load(
            "Concurrent cached queries — 50 identical (cache hit)",
            samples=list(latencies),
            qps=qps,
            note="50 concurrent, all cache hits",
        )

        assert qps >= 500.0, (
            f"Cached concurrent throughput = {qps:.0f} QPS; expected >= 500 QPS. "
            "Redis cache should handle this easily on localhost."
        )
