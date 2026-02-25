"""Result fusion utilities — Reciprocal Rank Fusion (RRF) and sparse search stub.

RRF formula (Cormack et al., 2009):
    score(d) = Σ_r  1 / (k + rank_r(d))

where rank_r(d) is the 1-based rank of document d in ranked list r, and k is
a smoothing constant (default 60 per the original paper).

Usage::

    from retrieval_os.serving.fusion import reciprocal_rank_fusion

    dense_hits  = await vector_search(...)
    sparse_hits = await sparse_search(...)        # returns IndexHit with BM25 scores
    fused       = reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=top_k)
"""

from __future__ import annotations

import logging
from collections import defaultdict

from retrieval_os.serving.index_proxy import IndexHit

log = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[IndexHit]],
    *,
    top_k: int,
    k: int = 60,
) -> list[IndexHit]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists:  Each inner list must be ordered from highest score to
                       lowest (as returned by vector_search / sparse_search).
        top_k:         Maximum number of results to return.
        k:             RRF smoothing constant (default 60).

    Returns:
        New list of IndexHit sorted by RRF score descending, length ≤ top_k.
        The score field on each hit contains the RRF score (not the original).
        Payload is taken from the first ranked_list in which the id appears.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    payloads: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank_0, hit in enumerate(ranked):
            rank_1 = rank_0 + 1  # 1-based
            rrf_scores[hit.id] += 1.0 / (k + rank_1)
            if hit.id not in payloads:
                payloads[hit.id] = hit.payload

    sorted_ids = sorted(rrf_scores, key=lambda doc_id: rrf_scores[doc_id], reverse=True)

    return [
        IndexHit(id=doc_id, score=rrf_scores[doc_id], payload=payloads[doc_id])
        for doc_id in sorted_ids[:top_k]
    ]


# ── Sparse search stub ────────────────────────────────────────────────────────


async def sparse_search(
    *,
    collection: str,
    query: str,
    top_k: int,
) -> list[IndexHit]:
    """BM25 / sparse vector search — planned for Phase 9 (full-text index).

    Currently returns an empty list so that RRF degrades gracefully to
    pure dense retrieval when called with a sparse list.
    """
    log.debug(
        "sparse_search.stub",
        extra={"collection": collection, "top_k": top_k},
    )
    return []
