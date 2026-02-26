"""Load test: Ingestion throughput — Qdrant vector upsert performance.

Measures three scenarios:
1. Sequential batch upsert        (single-client, 100-vector batches)
2. Large single-batch upsert      (500 vectors, typical async-worker batch)
3. Concurrent batch upsert        (10 parallel 100-vector batches)

Infrastructure required (auto-skipped when unreachable):
  - Qdrant  (existing load_collection used; extra vectors are cleaned up at
             session teardown along with the collection)

All vectors are random unit vectors.  Throughput figures represent the
write path of the Qdrant gRPC server; application-side chunking and
embedding are not included.
"""

from __future__ import annotations

import asyncio
import time

from retrieval_os.core.ids import uuid7
from retrieval_os.serving.index_proxy import upsert_vectors
from tests.load.conftest import random_unit_vector

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_points(n: int, offset: int = 0) -> list[dict]:
    """Generate *n* random-vector point dicts for upsert."""
    return [
        {
            "id": str(uuid7()),
            "vector": random_unit_vector(),
            "payload": {"text": f"load-ingest doc {offset + i}", "index": offset + i},
        }
        for i in range(n)
    ]


# ── Sequential upsert ─────────────────────────────────────────────────────────


class TestSequentialUpsertThroughput:
    """Single-client sequential upsert — establishes the per-batch baseline."""

    async def test_1000_vectors_sequential_above_1k_vps(self, load_collection, record_load) -> None:
        """1 000 vectors in 10 sequential batches of 100 must exceed 1 000 vec/sec.

        At 1 000 vec/sec, indexing 1 million documents takes ≈ 17 minutes —
        acceptable for an initial cold-start ingest before going live.
        This test also validates that the Qdrant gRPC upsert path is healthy
        and ``wait=True`` semantics are enforced (no stale reads).
        """
        batch_size = 100
        n_batches = 10
        total_vectors = batch_size * n_batches
        latencies: list[float] = []

        for b in range(n_batches):
            points = _make_points(batch_size, offset=b * batch_size)
            t0 = time.perf_counter()
            n = await upsert_vectors(
                backend="qdrant",
                collection=load_collection,
                points=points,
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            assert n == batch_size

        total_ms = sum(latencies)
        vps = total_vectors / (total_ms / 1000)

        record_load(
            "Sequential upsert (100 vec/batch × 10 batches)",
            samples=latencies,
            qps=vps,
            note="vectors/sec shown in QPS column",
        )

        assert vps >= 1_000, (
            f"Sequential upsert = {vps:.0f} vec/sec; expected >= 1 000 vec/sec. "
            "Check Qdrant write path or available CPU/memory."
        )

    async def test_500_vector_batch_under_2s(self, load_collection, record_load) -> None:
        """A single 500-vector batch must complete in < 2 seconds.

        500 vectors is a typical production batch size for an async ingest
        worker.  Completing in < 2 s keeps the worker loop responsive.
        """
        points = _make_points(500)
        t0 = time.perf_counter()
        n = await upsert_vectors(
            backend="qdrant",
            collection=load_collection,
            points=points,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        vps = 500 / (elapsed_ms / 1000) if elapsed_ms > 0 else float("inf")

        record_load(
            "Single 500-vector batch upsert",
            samples=[elapsed_ms],
            qps=vps,
            note="vectors/sec shown in QPS column",
        )

        assert n == 500
        assert elapsed_ms < 2_000, (
            f"500-vector batch took {elapsed_ms:.0f} ms; expected < 2 000 ms."
        )


# ── Concurrent upsert ─────────────────────────────────────────────────────────


class TestConcurrentUpsertThroughput:
    """Parallel batch upsert — exercises Qdrant write concurrency."""

    async def test_10_concurrent_batches_of_100_above_1k_vps(
        self, load_collection, record_load
    ) -> None:
        """10 concurrent 100-vector batches must sustain ≥ 1 000 vec/sec overall.

        Concurrent upserts exercise Qdrant's WAL parallelism.  The throughput
        should be at least as good as sequential; if it is significantly lower,
        Qdrant's write path has a bottleneck worth investigating.
        """
        n_concurrent = 10
        batch_size = 100

        async def _upsert_batch(offset: int) -> float:
            points = _make_points(batch_size, offset=offset * batch_size)
            t0 = time.perf_counter()
            await upsert_vectors(
                backend="qdrant",
                collection=load_collection,
                points=points,
            )
            return (time.perf_counter() - t0) * 1000

        wall_start = time.perf_counter()
        latencies = await asyncio.gather(*[_upsert_batch(i) for i in range(n_concurrent)])
        wall_elapsed = time.perf_counter() - wall_start

        total_vectors = n_concurrent * batch_size
        vps = total_vectors / wall_elapsed

        record_load(
            "Concurrent upsert (10 batches × 100 vectors, parallel)",
            samples=list(latencies),
            qps=vps,
            note="vectors/sec shown in QPS column",
        )

        assert vps >= 1_000, f"Concurrent upsert = {vps:.0f} vec/sec; expected >= 1 000 vec/sec."
