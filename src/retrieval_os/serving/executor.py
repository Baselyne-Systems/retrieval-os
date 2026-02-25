"""Retrieval Executor — orchestrates the full query pipeline.

Flow:
  1. Cache lookup (Redis, keyed by plan+version+query+top_k)
  2. Embed query text
  3. Vector search
  4. Optional rerank (stub — Phase 8)
  5. Cache set
  6. Return results

The executor never reads Postgres on the hot path. Plan config is passed in
by the query router, which fetches it from Redis (refreshed async from Postgres).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from retrieval_os.core import metrics
from retrieval_os.serving.cache import cache_get, cache_set
from retrieval_os.serving.embed_router import embed_text
from retrieval_os.serving.fusion import reciprocal_rank_fusion, sparse_search
from retrieval_os.serving.index_proxy import IndexHit, vector_search

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    id: str
    score: float
    text: str
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }


async def execute_retrieval(
    *,
    plan_name: str,
    version: int,
    query: str,
    # Plan config (passed in — never loaded from DB here)
    embedding_provider: str,
    embedding_model: str,
    embedding_normalize: bool,
    embedding_batch_size: int,
    index_backend: str,
    index_collection: str,
    distance_metric: str,
    top_k: int,
    reranker: str | None,
    rerank_top_k: int | None,
    metadata_filters: dict | None,
    cache_enabled: bool,
    cache_ttl_seconds: int,
    hybrid_alpha: float | None = None,
) -> tuple[list[RetrievedChunk], bool]:
    """Execute the retrieval pipeline for a single query.

    Returns:
        (chunks, cache_hit) — chunks sorted by score descending.
    """
    start = time.perf_counter()

    # 1. Cache lookup ──────────────────────────────────────────────────────────
    cache_hit = False
    if cache_enabled:
        cached = await cache_get(plan_name, version, query, top_k)
        if cached is not None:
            metrics.cache_hits_total.labels(plan_name=plan_name).inc()
            cache_hit = True
            chunks = [
                RetrievedChunk(
                    id=c["id"],
                    score=c["score"],
                    text=c.get("text", ""),
                    metadata=c.get("metadata", {}),
                )
                for c in cached
            ]
            _record_latency(plan_name, start)
            return chunks, cache_hit

    metrics.cache_misses_total.labels(plan_name=plan_name).inc()

    # 2. Embed ─────────────────────────────────────────────────────────────────
    vectors = await embed_text(
        [query],
        provider=embedding_provider,
        model=embedding_model,
        normalize=embedding_normalize,
        batch_size=embedding_batch_size,
    )
    query_vector = vectors[0]

    # 3. Vector search ─────────────────────────────────────────────────────────
    dense_hits: list[IndexHit] = await vector_search(
        backend=index_backend,
        collection=index_collection,
        vector=query_vector,
        top_k=top_k,
        distance_metric=distance_metric,
        metadata_filters=metadata_filters,
    )

    # 3b. Hybrid: fuse dense + sparse via RRF when hybrid_alpha is configured.
    # hybrid_alpha=1.0 → pure dense; 0.0 → pure sparse (sparse stub returns []).
    if hybrid_alpha is not None and hybrid_alpha < 1.0:
        sparse_hits = await sparse_search(
            collection=index_collection,
            query=query,
            top_k=top_k,
        )
        hits = reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=top_k)
    else:
        hits = dense_hits

    # 4. Rerank (stub) ─────────────────────────────────────────────────────────
    if reranker and rerank_top_k:
        hits = _rerank_stub(hits, query=query, reranker=reranker, top_k=rerank_top_k)

    # 5. Build result ──────────────────────────────────────────────────────────
    chunks = [
        RetrievedChunk(
            id=h.id,
            score=h.score,
            text=h.payload.get("text", ""),
            metadata={k: v for k, v in h.payload.items() if k != "text"},
        )
        for h in hits
    ]

    # 6. Cache set ─────────────────────────────────────────────────────────────
    if cache_enabled:
        await cache_set(
            plan_name, version, query, top_k,
            [c.to_dict() for c in chunks],
            ttl_seconds=cache_ttl_seconds,
        )

    _record_latency(plan_name, start)
    return chunks, cache_hit


def _record_latency(plan_name: str, start: float) -> None:
    elapsed = time.perf_counter() - start
    metrics.retrieval_latency_seconds.labels(plan_name=plan_name).observe(elapsed)
    metrics.retrieval_requests_total.labels(plan_name=plan_name).inc()


def _rerank_stub(
    hits: list[IndexHit],
    *,
    query: str,
    reranker: str,
    top_k: int,
) -> list[IndexHit]:
    """Reranking placeholder — Phase 8 will integrate a cross-encoder or Cohere."""
    log.debug("rerank.stub", extra={"reranker": reranker, "top_k": top_k})
    return hits[:top_k]
