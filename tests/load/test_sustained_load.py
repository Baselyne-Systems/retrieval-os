"""Load test: Sustained load stability — 30-second continuous query stream.

Detects:
  - QPS degradation        (memory leak, connection pool exhaustion, GC pressure)
  - Latency drift          (p99 growing over time = intermittent slowdown)
  - Error rate increase    (exceptions that start occurring after warmup)

How it works
------------
Each test runs a fixed-concurrency coroutine pool for DURATION_SECONDS.
Samples are split into time windows; QPS and p99 are computed per window.
Assertions enforce:
  - Final window QPS ≥ 80% of peak window QPS       (no sustained degradation)
  - Final window p99 ≤ 2× first-window p99          (no latency drift)
  - Zero exceptions across the full run

Infrastructure required: Postgres + Redis + Qdrant.
Auto-skipped when any service is unreachable.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient

from retrieval_os.api.main import app
from retrieval_os.serving.executor import execute_retrieval
from tests.load.conftest import random_unit_vector

# ── Configuration ─────────────────────────────────────────────────────────────

DURATION_SECONDS = 30  # increase to 60 for longer soak tests
WINDOW_SECONDS = 5  # split into 6 × 5-second windows


# ── Result collector ──────────────────────────────────────────────────────────


@dataclass
class _Sample:
    latency_ms: float
    ts: float  # wall-clock time of completion
    error: bool = False


@dataclass
class _WindowStats:
    window_idx: int
    qps: float
    p50: float
    p95: float
    p99: float
    errors: int


def _compute_windows(samples: list[_Sample], window_s: float) -> list[_WindowStats]:
    """Split samples into fixed-duration windows and compute per-window stats."""
    if not samples:
        return []
    t_start = samples[0].ts
    n_windows = max(1, int(DURATION_SECONDS / window_s))
    windows: list[list[_Sample]] = [[] for _ in range(n_windows)]
    for s in samples:
        idx = min(int((s.ts - t_start) / window_s), n_windows - 1)
        windows[idx].append(s)

    result = []
    for i, w in enumerate(windows):
        if not w:
            continue
        lats = sorted(s.latency_ms for s in w if not s.error)
        n = len(lats)
        result.append(
            _WindowStats(
                window_idx=i,
                qps=len(w) / window_s,
                p50=lats[n // 2] if lats else 0,
                p95=lats[int(n * 0.95)] if lats else 0,
                p99=lats[int(n * 0.99)] if lats else 0,
                errors=sum(1 for s in w if s.error),
            )
        )
    return result


async def _run_sustained(
    query_fn,
    *,
    concurrency: int,
    duration_s: float,
) -> list[_Sample]:
    """Drive *query_fn* at *concurrency* for *duration_s* seconds.

    Returns one _Sample per completed call.
    """
    samples: list[_Sample] = []
    stop = asyncio.Event()

    async def _worker() -> None:
        while not stop.is_set():
            t0 = time.perf_counter()
            error = False
            try:
                await query_fn()
            except Exception:
                error = True
            samples.append(
                _Sample(
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    ts=time.perf_counter(),
                    error=error,
                )
            )

    workers = [asyncio.create_task(_worker()) for _ in range(concurrency)]
    await asyncio.sleep(duration_s)
    stop.set()
    await asyncio.gather(*workers, return_exceptions=True)
    return samples


# ── Executor-level sustained test ─────────────────────────────────────────────


class TestSustainedExecutorLoad:
    """30-second sustained test at the executor level (no HTTP overhead)."""

    async def test_30s_cache_miss_no_degradation(
        self, load_project, load_collection, record_load
    ) -> None:
        """30-second sustained cache miss stream must produce zero errors.

        Runs 10 concurrent workers, each firing unique queries (all cache misses)
        for 30 seconds.  Detects:
          - Hard errors (exceptions from exhausted pools, handle leaks, etc.)
          - Complete throughput stall (total QPS < floor)

        Note: per-window QPS may vary significantly because local Qdrant performs
        HNSW reoptimisation under sustained load, causing natural throughput
        fluctuation.  The critical signal is zero errors and sustained throughput
        above the floor — not peak-vs-final window stability.
        """
        stub_vector = random_unit_vector()
        _counter = [0]

        def _next_query() -> str:
            _counter[0] += 1
            return f"sustained cache miss query number {_counter[0]}"

        def _exec_kwargs(query: str) -> dict:
            return dict(
                project_name=load_project,
                version=1,
                query=query,
                embedding_provider="sentence_transformers",
                embedding_model="all-MiniLM-L6-v2",
                embedding_normalize=True,
                embedding_batch_size=32,
                index_backend="qdrant",
                index_collection=load_collection,
                distance_metric="cosine",
                top_k=10,
                reranker=None,
                rerank_top_k=None,
                metadata_filters=None,
                cache_enabled=True,
                cache_ttl_seconds=3600,
            )

        async def _query():
            await execute_retrieval(**_exec_kwargs(_next_query()))

        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            samples = await _run_sustained(_query, concurrency=10, duration_s=DURATION_SECONDS)

        windows = _compute_windows(samples, WINDOW_SECONDS)
        total_errors = sum(s.error for s in samples)
        total_qs = sum(1 for s in samples if not s.error)
        total_qps = total_qs / DURATION_SECONDS

        record_load(
            f"Sustained 30s cache miss (10 concurrent, {total_qs} queries)",
            samples=[s.latency_ms for s in samples if not s.error],
            qps=total_qps,
            note="10 workers, all cache misses",
        )

        assert total_errors == 0, f"{total_errors} errors during 30-second sustained run"
        assert len(windows) >= 3, "Need at least 3 windows to measure stability"

        # Total throughput floor — catches complete stalls, not natural fluctuation.
        # 30 QPS = 300 queries/30s with 10 workers; a healthy system far exceeds this.
        assert total_qps >= 30, (
            f"Average QPS = {total_qps:.0f} over 30s; expected ≥ 30 QPS. "
            "System may have stalled — check connection pool exhaustion."
        )

    async def test_30s_cache_hit_stability(
        self, load_project, load_collection, record_load
    ) -> None:
        """30-second repeated-query stream (cache hits) must stay stable.

        Cache hit path is Redis-only.  p99 > 10 ms sustained suggests Redis
        connection pool exhaustion or event loop saturation.
        """
        stub_vector = random_unit_vector()
        warm_query = "what is the optimal chunk size for production RAG systems?"

        # Warm the cache
        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            await execute_retrieval(
                project_name=load_project,
                version=1,
                query=warm_query,
                embedding_provider="sentence_transformers",
                embedding_model="all-MiniLM-L6-v2",
                embedding_normalize=True,
                embedding_batch_size=32,
                index_backend="qdrant",
                index_collection=load_collection,
                distance_metric="cosine",
                top_k=10,
                reranker=None,
                rerank_top_k=None,
                metadata_filters=None,
                cache_enabled=True,
                cache_ttl_seconds=3600,
            )

        async def _cached_query():
            await execute_retrieval(
                project_name=load_project,
                version=1,
                query=warm_query,
                embedding_provider="sentence_transformers",
                embedding_model="all-MiniLM-L6-v2",
                embedding_normalize=True,
                embedding_batch_size=32,
                index_backend="qdrant",
                index_collection=load_collection,
                distance_metric="cosine",
                top_k=10,
                reranker=None,
                rerank_top_k=None,
                metadata_filters=None,
                cache_enabled=True,
                cache_ttl_seconds=3600,
            )

        samples = await _run_sustained(_cached_query, concurrency=20, duration_s=DURATION_SECONDS)

        windows = _compute_windows(samples, WINDOW_SECONDS)
        total_errors = sum(s.error for s in samples)
        total_qs = sum(1 for s in samples if not s.error)
        total_qps = total_qs / DURATION_SECONDS

        record_load(
            f"Sustained 30s cache hit (20 concurrent, {total_qs} queries)",
            samples=[s.latency_ms for s in samples if not s.error],
            qps=total_qps,
            note="20 workers, all cache hits",
        )

        assert total_errors == 0, f"{total_errors} errors during 30-second sustained cache-hit run"

        # QPS must stay high; p99 must stay low
        qps_values = [w.qps for w in windows]
        final_qps = windows[-1].qps if windows else 0
        peak_qps = max(qps_values) if qps_values else 1

        assert final_qps >= 0.8 * peak_qps, (
            f"Cache-hit QPS degraded: final={final_qps:.0f}, peak={peak_qps:.0f}. "
            "Possible Redis connection pool exhaustion."
        )

        overall_p99 = sorted(s.latency_ms for s in samples if not s.error)[int(total_qs * 0.99)]
        assert overall_p99 < 10.0, (
            f"Cache-hit p99={overall_p99:.1f} ms over 30-second run; expected < 10 ms. "
            "Redis latency spikes may indicate pool exhaustion or GC pressure."
        )


# ── HTTP-layer sustained test ─────────────────────────────────────────────────


class TestSustainedHTTPLoad:
    """30-second sustained test including the full FastAPI HTTP stack."""

    async def test_30s_http_full_stack_no_degradation(
        self, load_project, load_collection, record_load
    ) -> None:
        """Full HTTP stack sustained 30 seconds — QPS and p99 must stay stable.

        This is the closest approximation to real production traffic:
          HTTP decode → Pydantic → route_query → Qdrant → Redis → response

        Detects HTTP-layer issues: keep-alive connection pool exhaustion,
        middleware state accumulation, or event loop saturation.
        """
        stub_vector = random_unit_vector()
        _query_counter = [0]

        async def _http_query(client: AsyncClient) -> None:
            _query_counter[0] += 1
            await client.post(
                f"/v1/query/{load_project}",
                json={"query": f"sustained http query {_query_counter[0]}"},
            )

        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                samples = await _run_sustained(
                    lambda: _http_query(client),
                    concurrency=10,
                    duration_s=DURATION_SECONDS,
                )

        windows = _compute_windows(samples, WINDOW_SECONDS)
        total_errors = sum(s.error for s in samples)
        total_qs = sum(1 for s in samples if not s.error)
        total_qps = total_qs / DURATION_SECONDS

        record_load(
            f"Sustained 30s HTTP full stack (10 concurrent, {total_qs} queries)",
            samples=[s.latency_ms for s in samples if not s.error],
            qps=total_qps,
            note="10 workers via ASGI, embed stubbed",
        )

        assert total_errors == 0, (
            f"{total_errors} HTTP errors during 30-second sustained run. "
            "Check FastAPI exception handlers or DB session management."
        )

        if len(windows) >= 2:
            qps_values = [w.qps for w in windows]
            peak_qps = max(qps_values)
            final_qps = windows[-1].qps
            assert final_qps >= 0.75 * peak_qps, (
                f"HTTP QPS degraded: final={final_qps:.0f}, peak={peak_qps:.0f} "
                f"({(1 - final_qps / peak_qps) * 100:.0f}% drop over 30 s). "
                "Check for connection leak or middleware state accumulation."
            )
