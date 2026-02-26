"""Infrastructure setup and result collection for load tests.

Load tests exercise the full production stack:
  - PostgreSQL  (project / deployment config)
  - Redis       (serving config cache)
  - Qdrant      (ANN vector search)

All tests skip automatically when any service is unreachable.
Run `make infra && make migrate` to enable load tests.

Embedding is intentionally stubbed with random unit vectors throughout.
This isolates the retrieval infrastructure latency that Retrieval-OS controls.
Real embedding latency (2–150 ms depending on model and hardware) is additive
and documented separately.
"""

from __future__ import annotations

import math
import random
import statistics
import uuid
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sqlalchemy import text

from retrieval_os.core.database import async_session_factory, check_db_connection
from retrieval_os.core.ids import uuid7
from retrieval_os.core.redis_client import check_redis_connection, get_redis
from retrieval_os.deployments.schemas import CreateDeploymentRequest
from retrieval_os.deployments.service import create_deployment
from retrieval_os.plans.schemas import CreateProjectRequest, IndexConfigInput
from retrieval_os.plans.service import create_project

# ── Constants ─────────────────────────────────────────────────────────────────

DIMS = 384  # matches all-MiniLM-L6-v2 (fast CPU model) — swap for your model
N_VECTORS = 10_000  # realistic collection size

# ── Vector helpers ─────────────────────────────────────────────────────────────


def random_unit_vector(dims: int = DIMS) -> list[float]:
    """Random unit vector for use as query or document embedding stub."""
    v = [random.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


# ── Infrastructure guard ───────────────────────────────────────────────────────


@pytest.fixture(scope="session")
async def check_load_infra() -> None:
    """Skip all load tests if any required service is unreachable."""
    if not await check_db_connection():
        pytest.skip("Postgres not reachable — run `make infra && make migrate`")
    if not await check_redis_connection():
        pytest.skip("Redis not reachable — run `make infra`")
    try:
        client = AsyncQdrantClient(url="http://localhost:6333", check_compatibility=False)
        await client.get_collections()
        await client.close()
    except Exception:
        pytest.skip("Qdrant not reachable — run `make infra`")


# ── Qdrant collection with 10k random vectors ─────────────────────────────────


@pytest.fixture(scope="session")
async def load_collection(check_load_infra) -> str:  # type: ignore[misc]
    """Create a 10k-vector Qdrant collection for load tests; delete on teardown."""
    collection = f"load-{uuid.uuid4().hex[:8]}"
    client = AsyncQdrantClient(url="http://localhost:6333", check_compatibility=False)

    await client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE),
    )

    # Insert 10k vectors in batches of 500
    batch_size = 500
    for batch_start in range(0, N_VECTORS, batch_size):
        n = min(batch_size, N_VECTORS - batch_start)
        points = [
            PointStruct(
                id=str(uuid7()),
                vector=random_unit_vector(),
                payload={"text": f"document {batch_start + i}", "doc_id": f"doc-{batch_start + i}"},
            )
            for i in range(n)
        ]
        await client.upsert(collection_name=collection, points=points, wait=True)

    yield collection

    await client.delete_collection(collection)
    await client.close()


# ── Postgres project + active deployment ──────────────────────────────────────


@pytest.fixture(scope="session")
async def load_project(load_collection) -> str:  # type: ignore[misc]
    """Create a project + index config + active deployment; clean up on teardown."""
    name = f"load-{uuid.uuid4().hex[:8]}"

    async with async_session_factory() as session:
        await create_project(
            session,
            CreateProjectRequest(
                name=name,
                description="Load test project",
                config=IndexConfigInput(
                    embedding_provider="sentence_transformers",
                    embedding_model="all-MiniLM-L6-v2",
                    embedding_dimensions=DIMS,
                    index_collection=load_collection,
                    index_backend="qdrant",
                    distance_metric="cosine",
                    change_comment="load test",
                ),
                created_by="load-test",
            ),
        )
        await session.commit()

    async with async_session_factory() as session:
        await create_deployment(
            session,
            name,
            CreateDeploymentRequest(
                index_config_version=1,
                top_k=10,
                cache_enabled=True,
                cache_ttl_seconds=3600,
                created_by="load-test",
            ),
        )
        await session.commit()

    yield name

    # Best-effort cleanup
    try:
        redis = await get_redis()
        await redis.delete(f"ros:project:{name}:active", f"ros:deployment:{name}:active")
    except Exception:
        pass
    try:
        async with async_session_factory() as session:
            await session.execute(
                text("DELETE FROM deployments WHERE project_name = :n"), {"n": name}
            )
            await session.execute(
                text(
                    "DELETE FROM index_configs "
                    "WHERE project_id = (SELECT id FROM projects WHERE name = :n)"
                ),
                {"n": name},
            )
            await session.execute(text("DELETE FROM projects WHERE name = :n"), {"n": name})
            await session.commit()
    except Exception:
        pass


# ── Result collection + terminal summary ──────────────────────────────────────

_LOAD_RESULTS: list[dict[str, Any]] = []


@pytest.fixture
def record_load():
    """Record a load test result for the terminal summary table."""

    def _record(
        label: str,
        *,
        samples: list[float],  # latency samples in ms
        qps: float | None = None,
        note: str = "",
    ) -> None:
        s = sorted(samples)
        n = len(s)
        _LOAD_RESULTS.append(
            {
                "label": label,
                "n": n,
                "p50": statistics.median(s) if s else 0.0,
                "p95": s[int(n * 0.95)] if s else 0.0,
                "p99": s[int(n * 0.99)] if s else 0.0,
                "qps": qps,
                "note": note,
            }
        )

    return _record


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if not _LOAD_RESULTS:
        return

    tw = terminalreporter
    tw.write_sep(
        "=", "load test results  (embedding latency excluded — add ~2–150 ms for your model)"
    )
    tw.write_line(f"  {'Test':<50} {'p50':>7}  {'p95':>7}  {'p99':>7}  {'QPS':>8}  {'n':>5}")
    tw.write_line("  " + "-" * 92)
    for r in _LOAD_RESULTS:
        qps_str = f"{r['qps']:>7.0f}" if r["qps"] is not None else "      —"
        note = f"  ({r['note']})" if r["note"] else ""
        tw.write_line(
            f"  {r['label']:<50} {r['p50']:>5.1f} ms  {r['p95']:>5.1f} ms"
            f"  {r['p99']:>5.1f} ms  {qps_str}  {r['n']:>5}{note}"
        )
    tw.write_line("\n  NOTE: All queries use random unit vectors (no embedding model loaded).")
    tw.write_line(
        "        Add your model's inference latency to p50/p95/p99 for end-to-end estimates."
    )
    tw.write_line("")
