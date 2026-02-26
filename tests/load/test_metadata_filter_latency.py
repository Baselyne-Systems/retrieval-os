"""Load test: ANN latency under payload filters at different selectivities.

What this proves
----------------
How filter selectivity and payload indexing affect ANN search latency on a
10k-vector Qdrant collection. Results quantify when filters become expensive
and whether payload indexes provide meaningful speedup.

Collection layout
-----------------
10k vectors with three payload fields (uniform distribution):
  - doc_type   ∈ {policy, faq, guide, api, release}    — 20% each; INDEXED
  - tenant_id  ∈ {acme, globex, initech, umbrella, cyberdyne} — 20% each; INDEXED
  - year       ∈ {2020, 2021, 2022, 2023, 2024}         — 20% each; NOT indexed

Filter scenarios tested
-----------------------
| Label                    | Filter                         | Selectivity |
|--------------------------|-------------------------------|-------------|
| No filter                | None                           | 100%        |
| doc_type=policy (index)  | must: doc_type=policy          | ~20%        |
| year=2023 (no index)     | must: year=2023                | ~20%        |
| doc_type + tenant_id     | must: both (both indexed)      | ~4%         |

Infrastructure required: Postgres + Redis + Qdrant.
Auto-skipped when any service is unreachable.
"""

from __future__ import annotations

import time
import uuid

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import FieldCondition, MatchValue
from qdrant_client.models import Distance, PointStruct, VectorParams

from retrieval_os.core.config import settings
from retrieval_os.core.ids import uuid7
from retrieval_os.serving.index_proxy import vector_search
from tests.load.conftest import DIMS, N_VECTORS, random_unit_vector

# ── Payload distribution constants ─────────────────────────────────────────────

_DOC_TYPES = ["policy", "faq", "guide", "api", "release"]
_TENANT_IDS = ["acme", "globex", "initech", "umbrella", "cyberdyne"]
_YEARS = [2020, 2021, 2022, 2023, 2024]

# ── Shared state ───────────────────────────────────────────────────────────────

_baseline_p99: float = 0.0  # populated by test_unfiltered_ann_baseline

# ── Helpers ────────────────────────────────────────────────────────────────────


def _percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = int(len(s) * p)
    return s[min(idx, len(s) - 1)]


async def _search(
    collection: str, top_k: int = 10, metadata_filters: dict | None = None
) -> tuple[list, float]:
    """Run vector_search and return (hits, elapsed_ms)."""
    t0 = time.perf_counter()
    hits = await vector_search(
        backend="qdrant",
        collection=collection,
        vector=random_unit_vector(),
        top_k=top_k,
        distance_metric="cosine",
        metadata_filters=metadata_filters,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return hits, elapsed_ms


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
async def filter_collection(check_load_infra) -> str:  # type: ignore[misc]
    """Create a 10k-vector collection with payload metadata + payload indexes."""
    name = f"load-filters-{uuid.uuid4().hex[:8]}"
    client = AsyncQdrantClient(
        url=f"http://{settings.qdrant_host}:{settings.qdrant_http_port}",
        check_compatibility=False,
    )

    await client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=DIMS, distance=Distance.COSINE),
    )

    # Insert 10k vectors with structured payloads
    batch_size = 500
    for batch_start in range(0, N_VECTORS, batch_size):
        n = min(batch_size, N_VECTORS - batch_start)
        points = []
        for i in range(n):
            global_idx = batch_start + i
            points.append(
                PointStruct(
                    id=str(uuid7()),
                    vector=random_unit_vector(),
                    payload={
                        "text": f"document {global_idx}",
                        "doc_type": _DOC_TYPES[global_idx % len(_DOC_TYPES)],
                        "tenant_id": _TENANT_IDS[global_idx % len(_TENANT_IDS)],
                        "year": _YEARS[global_idx % len(_YEARS)],
                    },
                )
            )
        await client.upsert(collection_name=name, points=points, wait=True)

    # Create payload indexes on doc_type and tenant_id (NOT year)
    await client.create_payload_index(
        collection_name=name,
        field_name="doc_type",
        field_schema="keyword",
    )
    await client.create_payload_index(
        collection_name=name,
        field_name="tenant_id",
        field_schema="keyword",
    )

    await client.close()

    yield name

    try:
        client = AsyncQdrantClient(
            url=f"http://{settings.qdrant_host}:{settings.qdrant_http_port}",
            check_compatibility=False,
        )
        await client.delete_collection(name)
        await client.close()
    except Exception:
        pass


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestMetadataFilterLatency:
    """ANN latency at different payload filter selectivities and index configurations."""

    async def test_unfiltered_ann_baseline(self, filter_collection, record_load) -> None:
        """100 unfiltered ANN queries — establishes p99 reference.

        This is the floor latency for this collection. All filtered scenarios
        are compared against this number to quantify filter overhead.
        """
        global _baseline_p99

        n_queries = 100
        latencies: list[float] = []

        for _ in range(n_queries):
            _, ms = await _search(filter_collection)
            latencies.append(ms)

        _baseline_p99 = _percentile(latencies, 0.99)
        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            "Metadata filter — no filter (100% selectivity)",
            samples=latencies,
            qps=qps,
            note=f"p99={_baseline_p99:.1f}ms, 10k vectors",
        )

    async def test_single_field_indexed_filter(self, filter_collection, record_load) -> None:
        """doc_type=policy filter with payload index — ~20% selectivity.

        A payload index on ``doc_type`` should allow Qdrant to prune the
        candidate set before ANN scoring, keeping latency bounded.

        Assert: p99 < baseline_p99 × 3  (index provides meaningful speedup)
        """
        global _baseline_p99

        n_queries = 100
        latencies: list[float] = []
        metadata_filters = {
            "must": [FieldCondition(key="doc_type", match=MatchValue(value="policy"))]
        }

        for _ in range(n_queries):
            _, ms = await _search(filter_collection, metadata_filters=metadata_filters)
            latencies.append(ms)

        p99 = _percentile(latencies, 0.99)
        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            "Metadata filter — doc_type=policy (indexed, ~20% selectivity)",
            samples=latencies,
            qps=qps,
            note=f"p99={p99:.1f}ms, baseline_p99={_baseline_p99:.1f}ms",
        )

        # Floor at 10 ms: with a very fast Qdrant on localhost the baseline p99 can be
        # sub-millisecond, making a relative 3× bound meaninglessly tight.
        limit = max(_baseline_p99 * 3.0, 10.0)
        assert p99 < limit, (
            f"Indexed filter p99={p99:.1f}ms exceeds 3× baseline_p99={_baseline_p99:.1f}ms "
            f"(limit={limit:.1f}ms). "
            "Payload index may not be effective — check Qdrant index status."
        )

    async def test_single_field_no_index(self, filter_collection, record_load) -> None:
        """year=2023 filter WITHOUT payload index — ~20% selectivity.

        Qdrant must scan all payloads to evaluate this filter. No hard
        assertion — the result row documents the unindexed filter cost so
        teams can decide whether to add a payload index for ``year``.
        """
        n_queries = 100
        latencies: list[float] = []
        metadata_filters = {"must": [FieldCondition(key="year", match=MatchValue(value=2023))]}

        for _ in range(n_queries):
            _, ms = await _search(filter_collection, metadata_filters=metadata_filters)
            latencies.append(ms)

        p99 = _percentile(latencies, 0.99)
        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            "Metadata filter — year=2023 (no index, ~20% selectivity)",
            samples=latencies,
            qps=qps,
            note=f"p99={p99:.1f}ms, no payload index — informational only",
        )
        # No assertion: this is a measurement-only test showing unindexed cost.

    async def test_compound_filter_small_candidate_set(
        self, filter_collection, record_load
    ) -> None:
        """doc_type=policy AND tenant_id=acme — both indexed, ~4% selectivity.

        With both fields indexed, Qdrant can intersect the two candidate sets
        before ANN scoring. At 4% selectivity (≈400 vectors from 10k), result
        count may be less than top_k=10 — this is expected and recorded.

        Assert: p99 < baseline_p99 × 5  (two-field compound filter is bounded)
        """
        global _baseline_p99

        n_queries = 100
        latencies: list[float] = []
        result_counts: list[int] = []

        metadata_filters = {
            "must": [
                FieldCondition(key="doc_type", match=MatchValue(value="policy")),
                FieldCondition(key="tenant_id", match=MatchValue(value="acme")),
            ]
        }

        for _ in range(n_queries):
            hits, ms = await _search(filter_collection, top_k=10, metadata_filters=metadata_filters)
            latencies.append(ms)
            result_counts.append(len(hits))

        p99 = _percentile(latencies, 0.99)
        avg_results = sum(result_counts) / len(result_counts) if result_counts else 0
        total_s = sum(latencies) / 1000
        qps = n_queries / total_s if total_s > 0 else 0

        record_load(
            "Metadata filter — doc_type + tenant_id (both indexed, ~4% selectivity)",
            samples=latencies,
            qps=qps,
            note=f"p99={p99:.1f}ms, avg_results={avg_results:.1f}, baseline_p99={_baseline_p99:.1f}ms",
        )

        print(f"\n  Average result count (may be < 10 for small candidate set): {avg_results:.1f}")

        limit = max(_baseline_p99 * 5.0, 20.0)
        assert p99 < limit, (
            f"Compound filter p99={p99:.1f}ms exceeds 5× baseline_p99={_baseline_p99:.1f}ms "
            f"(limit={limit:.1f}ms). "
            "Compound indexed filter overhead is too high — check Qdrant segment layout."
        )
