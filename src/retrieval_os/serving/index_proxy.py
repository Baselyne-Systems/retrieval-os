"""Index Proxy — thin async wrapper over Qdrant gRPC.

Abstracts the vector index backend so the retrieval executor never imports
Qdrant types directly. A pgvector backend can be added here in Phase 8.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from retrieval_os.core import metrics
from retrieval_os.core.circuit_breaker import get_index_breaker
from retrieval_os.core.config import settings
from retrieval_os.core.exceptions import IndexBackendError

log = logging.getLogger(__name__)

_qdrant_client: Any = None


def _get_qdrant() -> Any:
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from qdrant_client import AsyncQdrantClient  # type: ignore[import]

            _qdrant_client = AsyncQdrantClient(
                host=settings.qdrant_host,
                grpc_port=settings.qdrant_grpc_port,
                prefer_grpc=True,
                api_key=settings.qdrant_api_key or None,
            )
        except ImportError:
            raise IndexBackendError(
                "qdrant-client is not installed. "
                "Install with: uv add qdrant-client"
            )
    return _qdrant_client


# ── Canonical result type ─────────────────────────────────────────────────────


class IndexHit:
    """A single result from a vector index query."""

    __slots__ = ("id", "score", "payload")

    def __init__(self, id: str, score: float, payload: dict) -> None:
        self.id = id
        self.score = score
        self.payload = payload

    def to_dict(self) -> dict:
        return {"id": self.id, "score": self.score, "payload": self.payload}


# ── Public interface ──────────────────────────────────────────────────────────


async def vector_search(
    *,
    backend: str,
    collection: str,
    vector: list[float],
    top_k: int,
    distance_metric: str,
    metadata_filters: dict | None = None,
    score_threshold: float | None = None,
) -> list[IndexHit]:
    """Return top-k nearest neighbours from the specified index backend.

    Args:
        backend:          "qdrant" or "pgvector" (pgvector in Phase 8).
        collection:       Collection / table name in the index.
        vector:           Dense query embedding.
        top_k:            Number of results to return.
        distance_metric:  "cosine", "dot", or "euclidean" (informational; the
                          collection is pre-configured with the metric at creation).
        metadata_filters: Optional Qdrant filter dict (passed through as-is).
        score_threshold:  Minimum score; None = no threshold.

    Returns:
        List of IndexHit sorted by score descending.
    """
    breaker = get_index_breaker(backend)
    if backend == "qdrant":
        return await breaker.call(
            _qdrant_search,
            collection=collection,
            vector=vector,
            top_k=top_k,
            metadata_filters=metadata_filters,
            score_threshold=score_threshold,
        )
    if backend == "pgvector":
        raise IndexBackendError(
            "pgvector backend is not yet implemented. Planned for Phase 9."
        )
    raise IndexBackendError(f"Unknown index backend: '{backend}'")


async def _qdrant_search(
    *,
    collection: str,
    vector: list[float],
    top_k: int,
    metadata_filters: dict | None,
    score_threshold: float | None,
) -> list[IndexHit]:
    from qdrant_client.http.models import Filter  # type: ignore[import]

    client = _get_qdrant()
    start = time.perf_counter()
    try:
        results = await client.search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k,
            query_filter=Filter(**metadata_filters) if metadata_filters else None,
            score_threshold=score_threshold,
            with_payload=True,
        )
    except Exception as exc:
        metrics.index_errors_total.labels(backend="qdrant").inc()
        raise IndexBackendError(
            f"Qdrant search failed on collection '{collection}': {exc}"
        ) from exc
    finally:
        elapsed = time.perf_counter() - start
        metrics.index_latency_seconds.labels(backend="qdrant").observe(elapsed)

    return [
        IndexHit(
            id=str(r.id),
            score=r.score,
            payload=r.payload or {},
        )
        for r in results
    ]
