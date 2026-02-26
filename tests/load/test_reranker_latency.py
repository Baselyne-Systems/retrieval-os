"""Load test: Reranker latency overhead in the hot path.

What this proves
----------------
Reranking adds predictable, bounded overhead. Teams can use these numbers to
set realistic SLA budgets before choosing a reranker model.

Approach
--------
The ``rerank`` function in ``retrieval_os.serving.executor`` is patched with a
stub that sleeps for the specified duration and returns the top hits unchanged.
This isolates reranker overhead without loading an actual cross-encoder model.

``execute_retrieval()`` is called directly (not via ``route_query``) so that
reranker config can be injected without a Deployment row in Postgres.

Simulated workloads
-------------------
- No reranker       : baseline for comparison
- 50 ms sleep stub  : lightweight CPU cross-encoder (e.g. ms-marco-MiniLM)
- 120 ms sleep stub : full GPU cross-encoder (e.g. ms-marco-distilbert)

Infrastructure required: Postgres + Redis + Qdrant.
Auto-skipped when any service is unreachable.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

from retrieval_os.core.redis_client import get_redis
from retrieval_os.serving.executor import execute_retrieval
from retrieval_os.serving.index_proxy import IndexHit
from tests.load.conftest import random_unit_vector

# ── Shared state ───────────────────────────────────────────────────────────────

_baseline_p99: float = 0.0  # populated by test_baseline_no_reranker

# ── Helpers ────────────────────────────────────────────────────────────────────


def _exec_kwargs(
    project_name: str,
    collection: str,
    query: str,
    *,
    reranker: str | None = None,
    rerank_top_k: int | None = None,
) -> dict:
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
        reranker=reranker,
        rerank_top_k=rerank_top_k,
        metadata_filters=None,
        cache_enabled=False,  # disable cache — measure reranker overhead cleanly
        cache_ttl_seconds=3600,
    )


async def _clear_query_cache() -> None:
    redis = await get_redis()
    async for key in redis.scan_iter("ros:qcache:*"):
        await redis.delete(key)


def _percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = int(len(s) * p)
    return s[min(idx, len(s) - 1)]


def _make_rerank_stub(latency_s: float, top_k: int):
    """Return an async rerank stub that sleeps ``latency_s`` seconds."""

    async def _stub(
        hits: list[IndexHit], *, query: str, reranker: str, top_k: int = top_k
    ) -> list[IndexHit]:
        await asyncio.sleep(latency_s)
        return hits[:top_k]

    return _stub


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestRerankerLatency:
    """Quantifies reranker latency overhead relative to the no-reranker baseline."""

    async def test_baseline_no_reranker(self, load_project, load_collection, record_load) -> None:
        """30 queries with no reranker — establishes p99 reference row.

        Cache is disabled so every query exercises the full embed→ANN path.
        This is the floor latency the reranker overhead is added on top of.
        """
        global _baseline_p99

        n_queries = 30
        stub_vector = random_unit_vector()
        await _clear_query_cache()

        latencies: list[float] = []

        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            for i in range(n_queries):
                t0 = time.perf_counter()
                await execute_retrieval(
                    **_exec_kwargs(load_project, load_collection, f"reranker baseline query {i}")
                )
                latencies.append((time.perf_counter() - t0) * 1000)

        _baseline_p99 = _percentile(latencies, 0.99)

        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            "Reranker baseline — no reranker",
            samples=latencies,
            qps=qps,
            note=f"p99={_baseline_p99:.1f}ms",
        )

    async def test_reranker_50ms_overhead(self, load_project, load_collection, record_load) -> None:
        """Lightweight CPU cross-encoder stub: 50 ms sleep.

        Simulates the latency of a small cross-encoder model such as
        ``cross-encoder/ms-marco-MiniLM-L-6-v2`` running on CPU.

        Assertions
        ----------
        - p50 > 50 ms  (stub is actually running)
        - p99 < baseline_p99 + 100 ms  (overhead is bounded)
        """
        global _baseline_p99

        stub_vector = random_unit_vector()
        n_queries = 30
        overhead_s = 0.050
        top_k = 5

        latencies: list[float] = []

        with (
            patch(
                "retrieval_os.serving.executor.embed_text",
                new_callable=AsyncMock,
                return_value=[stub_vector],
            ),
            patch(
                "retrieval_os.serving.executor.rerank",
                side_effect=_make_rerank_stub(overhead_s, top_k),
            ),
        ):
            for i in range(n_queries):
                t0 = time.perf_counter()
                await execute_retrieval(
                    **_exec_kwargs(
                        load_project,
                        load_collection,
                        f"reranker 50ms query {i}",
                        reranker="cross-encoder",
                        rerank_top_k=top_k,
                    )
                )
                latencies.append((time.perf_counter() - t0) * 1000)

        p50 = _percentile(latencies, 0.50)
        p99 = _percentile(latencies, 0.99)
        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            "Reranker 50 ms stub (CPU cross-encoder)",
            samples=latencies,
            qps=qps,
            note=f"p50={p50:.1f}ms, p99={p99:.1f}ms, overhead_s={overhead_s}",
        )

        assert p50 > overhead_s * 1000 * 0.9, (
            f"p50={p50:.1f}ms should be > {overhead_s * 1000:.0f}ms (stub running). "
            "Check that rerank is being called — patch may not have applied."
        )

        limit = _baseline_p99 + 100.0
        assert p99 < limit, (
            f"Reranker 50 ms stub: p99={p99:.1f}ms exceeds baseline_p99 + 100ms "
            f"(limit={limit:.1f}ms). Reranker overhead is unbounded — check for lock contention."
        )

    async def test_reranker_120ms_overhead(
        self, load_project, load_collection, record_load
    ) -> None:
        """GPU cross-encoder stub: 120 ms sleep.

        Simulates the latency of a larger cross-encoder model such as
        ``cross-encoder/ms-marco-distilbert-base-v3`` on GPU.

        Assertions
        ----------
        - p50 > 120 ms  (stub is actually running)
        - p99 < baseline_p99 + 200 ms  (overhead is bounded)
        """
        global _baseline_p99

        stub_vector = random_unit_vector()
        n_queries = 30
        overhead_s = 0.120
        top_k = 5

        latencies: list[float] = []

        with (
            patch(
                "retrieval_os.serving.executor.embed_text",
                new_callable=AsyncMock,
                return_value=[stub_vector],
            ),
            patch(
                "retrieval_os.serving.executor.rerank",
                side_effect=_make_rerank_stub(overhead_s, top_k),
            ),
        ):
            for i in range(n_queries):
                t0 = time.perf_counter()
                await execute_retrieval(
                    **_exec_kwargs(
                        load_project,
                        load_collection,
                        f"reranker 120ms query {i}",
                        reranker="cross-encoder",
                        rerank_top_k=top_k,
                    )
                )
                latencies.append((time.perf_counter() - t0) * 1000)

        p50 = _percentile(latencies, 0.50)
        p99 = _percentile(latencies, 0.99)
        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            "Reranker 120 ms stub (GPU cross-encoder)",
            samples=latencies,
            qps=qps,
            note=f"p50={p50:.1f}ms, p99={p99:.1f}ms, overhead_s={overhead_s}",
        )

        assert p50 > overhead_s * 1000 * 0.9, (
            f"p50={p50:.1f}ms should be > {overhead_s * 1000:.0f}ms (stub running). "
            "Check that rerank is being called — patch may not have applied."
        )

        limit = _baseline_p99 + 200.0
        assert p99 < limit, (
            f"Reranker 120 ms stub: p99={p99:.1f}ms exceeds baseline_p99 + 200ms "
            f"(limit={limit:.1f}ms). Reranker overhead is unbounded."
        )
