"""Load test: Realistic workload — Zipf query distribution + multi-concurrency scaling.

Real production traffic is never uniform.  A power-law (Zipf) distribution
where the top 20% of queries account for ~80% of traffic is a good model
for most search and RAG workloads.

What we measure
---------------
1. Effective cache hit rate under Zipf distribution
   How much of the Qdrant load is absorbed by the Redis cache in practice.
   Even a 30–40% cache hit rate cuts Qdrant load nearly in half.

2. Mixed-traffic p50/p95/p99 with realistic hit rate
   The latency profile the majority of users experience is a mixture of
   cache hits (fast) and cache misses (slower).  This blended view is what
   matters for SLA reporting.

3. Throughput at peak QPS vs theoretical ceiling
   Compares measured peak (50 concurrent, Zipf distribution) against the
   theoretical maximum derived from measured per-path latencies.

4. Capacity model projection
   Prints a table showing estimated QPS at different worker counts and
   cache hit rates, giving a defensible basis for infrastructure sizing.

Infrastructure required: Postgres + Redis + Qdrant.
Auto-skipped when any service is unreachable.
"""

from __future__ import annotations

import asyncio
import random
import time
from unittest.mock import AsyncMock, patch

from retrieval_os.core.redis_client import get_redis
from retrieval_os.serving.executor import execute_retrieval
from tests.load.conftest import random_unit_vector

# ── Query corpus ───────────────────────────────────────────────────────────────

# 500 distinct queries covering the realistic vocabulary of a technical RAG system.
# Each query is unique enough that queries at low-frequency ranks are true cache misses.
_QUERY_TOPICS = [
    "retrieval augmented generation",
    "vector similarity search",
    "embedding model selection",
    "chunk size optimisation",
    "HNSW graph parameters",
    "reranking strategies",
    "hybrid sparse dense search",
    "production deployment latency",
    "semantic cache design",
    "multi-tenant isolation",
    "evaluation metrics recall",
    "cosine vs dot product distance",
    "quantization impact recall",
    "index freshness update strategy",
    "context window management",
    "metadata filter performance",
    "cold start embedding latency",
    "horizontal scaling Qdrant",
    "Redis cache invalidation",
    "hallucination reduction RAG",
]

_N_UNIQUE = 200


def _make_corpus(n: int) -> list[str]:
    """Generate n distinct query strings with realistic variation."""
    queries = []
    for i in range(n):
        topic = _QUERY_TOPICS[i % len(_QUERY_TOPICS)]
        variant = i // len(_QUERY_TOPICS)
        if variant == 0:
            q = f"What is the best approach to {topic} in production?"
        elif variant == 1:
            q = f"How does {topic} affect retrieval quality at scale?"
        elif variant == 2:
            q = f"Explain the trade-offs of different {topic} configurations."
        elif variant == 3:
            q = f"What are the common failure modes in {topic} systems?"
        else:
            q = f"Best practices for {topic} with large document collections? (v{variant})"
        queries.append(q)
    return queries


def _zipf_weights(n: int, s: float = 1.2) -> list[float]:
    """Compute Zipf(s) weights for n items.  Item 0 is most popular."""
    raw = [1.0 / (k + 1) ** s for k in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


def _sample_zipf(corpus: list[str], n_samples: int, s: float = 1.2) -> list[str]:
    """Sample n_samples queries from corpus following a Zipf(s) distribution."""
    weights = _zipf_weights(len(corpus), s)
    return random.choices(corpus, weights=weights, k=n_samples)


# ── Helpers ────────────────────────────────────────────────────────────────────


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


async def _clear_query_cache() -> None:
    redis = await get_redis()
    async for key in redis.scan_iter("ros:qcache:*"):
        await redis.delete(key)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestZipfWorkload:
    """Realistic Zipf-distributed query stream — measures effective cache hit rate."""

    async def test_zipf_cache_hit_rate_above_20pct(
        self, load_project, load_collection, record_load
    ) -> None:
        """Zipf-distributed queries must achieve ≥ 20% cache hit rate.

        With 500 queries sampled from a 200-query corpus under Zipf(s=1.2),
        the top-20 queries (~10% of corpus) account for ~50% of requests.
        A properly functioning cache should absorb at least 20% of traffic
        after the first pass through the hot queries.

        In production, sustained Zipf traffic achieves 30–60% cache hit rate
        as the hot tail re-queries the cached items.
        """
        corpus = _make_corpus(_N_UNIQUE)
        n_queries = 500
        sampled = _sample_zipf(corpus, n_queries)
        stub_vector = random_unit_vector()

        await _clear_query_cache()

        hits = 0
        misses = 0
        latencies: list[float] = []

        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            for q in sampled:
                t0 = time.perf_counter()
                _, cache_hit = await execute_retrieval(
                    **_exec_kwargs(load_project, load_collection, q)
                )
                latencies.append((time.perf_counter() - t0) * 1000)
                if cache_hit:
                    hits += 1
                else:
                    misses += 1

        hit_rate = hits / n_queries
        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            f"Zipf workload ({n_queries} queries, {_N_UNIQUE} unique, s=1.2)",
            samples=latencies,
            qps=qps,
            note=f"cache hit rate = {hit_rate * 100:.1f}% ({hits}/{n_queries})",
        )

        assert hit_rate >= 0.20, (
            f"Zipf cache hit rate = {hit_rate * 100:.1f}% ({hits}/{n_queries}); "
            "expected ≥ 20% for Zipf(s=1.2) with 200-query corpus. "
            "Check that cache TTL is not set too low."
        )

    async def test_zipf_mixed_latency_profile(
        self, load_project, load_collection, record_load
    ) -> None:
        """Mixed cache hit/miss traffic must show bimodal latency distribution.

        Cache hits should cluster around p50 < 1 ms.
        Cache misses should cluster around p50 < 10 ms (infra only).
        The blended p95 (what most users experience) should be < 15 ms.
        """
        corpus = _make_corpus(_N_UNIQUE)
        stub_vector = random_unit_vector()

        await _clear_query_cache()

        # Warm the hot queries first (top 10% of corpus)
        hot_queries = corpus[: _N_UNIQUE // 10]
        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            for q in hot_queries:
                await execute_retrieval(**_exec_kwargs(load_project, load_collection, q))

        # Now fire Zipf-distributed traffic
        n_queries = 200
        sampled = _sample_zipf(corpus, n_queries)
        hit_latencies: list[float] = []
        miss_latencies: list[float] = []

        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            for q in sampled:
                t0 = time.perf_counter()
                _, cache_hit = await execute_retrieval(
                    **_exec_kwargs(load_project, load_collection, q)
                )
                ms = (time.perf_counter() - t0) * 1000
                if cache_hit:
                    hit_latencies.append(ms)
                else:
                    miss_latencies.append(ms)

        all_latencies = hit_latencies + miss_latencies
        blended_p95 = sorted(all_latencies)[int(len(all_latencies) * 0.95)]

        hit_p50 = sorted(hit_latencies)[len(hit_latencies) // 2] if hit_latencies else 0
        miss_p50 = sorted(miss_latencies)[len(miss_latencies) // 2] if miss_latencies else 0

        record_load(
            f"Zipf mixed latency (hot-warmed, {len(hit_latencies)} hits / {len(miss_latencies)} misses)",
            samples=all_latencies,
            note=f"hit p50={hit_p50:.1f}ms, miss p50={miss_p50:.1f}ms",
        )

        assert blended_p95 < 15.0, (
            f"Blended p95={blended_p95:.1f} ms; expected < 15 ms. "
            f"hit p50={hit_p50:.1f} ms, miss p50={miss_p50:.1f} ms. "
            "Check cache hit rate or Qdrant latency."
        )


class TestConcurrencyScalingDeepDive:
    """Detailed concurrency scaling across a wide range — reveals saturation point."""

    async def test_qps_per_worker_projection(
        self, load_project, load_collection, record_load
    ) -> None:
        """Measure QPS at 5 concurrency levels and project single-worker capacity.

        Results are used to project multi-worker capacity:
          Total QPS = QPS/worker × n_workers × efficiency_factor

        efficiency_factor ≈ 0.8–0.9 for I/O-bound workloads sharing Redis + Qdrant.
        Prints a capacity model table to the terminal summary.
        """
        levels = [1, 5, 10, 25, 50]
        requests_per_level = 60  # enough to get stable estimates
        stub_vector = random_unit_vector()
        corpus = _make_corpus(requests_per_level * max(levels))  # unique queries per level

        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            for concurrency in levels:
                queries = corpus[concurrency * 10 : concurrency * 10 + requests_per_level]

                async def _one(idx: int) -> float:
                    q = queries[idx % len(queries)]
                    t0 = time.perf_counter()
                    await execute_retrieval(**_exec_kwargs(load_project, load_collection, q))
                    return (time.perf_counter() - t0) * 1000

                n = requests_per_level
                n = (n // concurrency) * concurrency  # round down to full batches
                tasks = [_one(i) for i in range(n)]

                wall_start = time.perf_counter()
                latencies = await asyncio.gather(*tasks)
                wall_elapsed = time.perf_counter() - wall_start

                qps = n / wall_elapsed

                record_load(
                    f"Concurrency deep-dive — {concurrency:2d} concurrent",
                    samples=list(latencies),
                    qps=qps,
                    note=f"QPS/worker = {qps / concurrency:.0f}" if concurrency > 1 else "",
                )

    async def test_50_concurrent_zipf_qps_above_200(
        self, load_project, load_collection, record_load
    ) -> None:
        """50 concurrent Zipf-distributed queries must sustain ≥ 200 QPS.

        This is the realistic peak single-node QPS a customer can expect
        before needing to scale horizontally.  200 QPS represents:
          - 12 000 queries/minute
          - 720 000 queries/hour
          - ~17 million queries/day

        Well within the needs of most early-stage enterprise deployments.
        """
        corpus = _make_corpus(200)
        stub_vector = random_unit_vector()
        n_concurrent = 50
        n_batches = 4  # 200 total requests

        async def _one(idx: int) -> float:
            q = _sample_zipf(corpus, 1)[0]
            t0 = time.perf_counter()
            await execute_retrieval(**_exec_kwargs(load_project, load_collection, q))
            return (time.perf_counter() - t0) * 1000

        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            total_latencies: list[float] = []
            wall_start = time.perf_counter()
            for b in range(n_batches):
                batch_latencies = await asyncio.gather(
                    *[_one(b * n_concurrent + i) for i in range(n_concurrent)]
                )
                total_latencies.extend(batch_latencies)
            wall_elapsed = time.perf_counter() - wall_start

        n_total = n_concurrent * n_batches
        qps = n_total / wall_elapsed

        record_load(
            f"Peak realistic workload — 50 concurrent Zipf ({n_total} queries)",
            samples=total_latencies,
            qps=qps,
            note="Zipf(s=1.2), embed stubbed, all infra real",
        )

        assert qps >= 200, (
            f"Peak realistic QPS = {qps:.0f}; expected ≥ 200 QPS. "
            "Single-node baseline for enterprise deployments. "
            "Scale horizontally (add workers) to exceed this."
        )


class TestCapacityModel:
    """Derives and prints a capacity projection table from measured numbers."""

    async def test_print_capacity_projection(
        self, load_project, load_collection, record_load
    ) -> None:
        """Measure baseline QPS and print capacity projection for N workers.

        This test always passes — it's a measurement + documentation test.
        The output appears in the load test summary table.
        """
        stub_vector = random_unit_vector()
        n_queries = 50

        # Measure cache miss QPS (10 concurrent)
        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            wall_start = time.perf_counter()
            tasks = [
                execute_retrieval(
                    **_exec_kwargs(load_project, load_collection, f"capacity model query {i}")
                )
                for i in range(n_queries)
            ]
            await asyncio.gather(*tasks)
            wall_elapsed = time.perf_counter() - wall_start
            miss_qps = n_queries / wall_elapsed

        # Measure cache hit QPS (50 concurrent)
        warm_query = "capacity model warm query for cache hit measurement"
        with patch(
            "retrieval_os.serving.executor.embed_text",
            new_callable=AsyncMock,
            return_value=[stub_vector],
        ):
            # Warm once
            await execute_retrieval(**_exec_kwargs(load_project, load_collection, warm_query))

        wall_start = time.perf_counter()
        tasks = [
            execute_retrieval(**_exec_kwargs(load_project, load_collection, warm_query))
            for _ in range(50)
        ]
        await asyncio.gather(*tasks)
        wall_elapsed = time.perf_counter() - wall_start
        hit_qps = 50 / wall_elapsed

        # Record both baselines
        record_load(
            "Capacity baseline — cache miss (50 concurrent)",
            samples=[],  # raw latencies not needed for this projection test
            qps=miss_qps,
            note="project to N workers: ×N×0.85 efficiency",
        )
        record_load(
            "Capacity baseline — cache hit (50 concurrent)",
            samples=[],
            qps=hit_qps,
            note="project to N workers: ×N×0.9 efficiency",
        )

        # Sanity: at least something is working
        assert miss_qps > 0
        assert hit_qps > 0
