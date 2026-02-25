"""Pure-Python retrieval quality metrics.

All functions operate on plain Python types (no ORM, no async) so they
can be unit-tested without any infrastructure.

Notation:
  retrieved_ids  — ordered list of document IDs returned by the index (best first)
  relevant_ids   — set of IDs that are ground-truth relevant for the query
  relevance_scores — dict[id, float] for graded-relevance NDCG; binary if absent
"""

from __future__ import annotations

import math


def compute_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of relevant documents found in the top-k results.

    Capped at 1.0: if there are more relevant documents than k, the maximum
    achievable recall is k / |relevant|, but we normalise against |relevant|.
    """
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return min(hits / len(relevant_ids), 1.0)


def compute_mrr(
    results: list[tuple[list[str], set[str]]],
) -> float:
    """Mean Reciprocal Rank across a list of (retrieved_ids, relevant_ids) pairs.

    For each query, the reciprocal rank is 1/rank of the first relevant hit,
    or 0 if no relevant document appears in the retrieved list.
    """
    if not results:
        return 0.0
    rrs: list[float] = []
    for retrieved_ids, relevant_ids in results:
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return sum(rrs) / len(rrs)


def compute_ndcg_at_k(
    retrieved_ids: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain at k.

    Uses log base 2. Position i (0-indexed) has discount 1 / log2(i + 2).
    relevance_scores maps document ID → relevance grade (e.g. 0, 1, or 0.5–1.0).
    Documents not in relevance_scores are treated as irrelevant (score=0).
    """
    if not relevance_scores:
        return 0.0

    dcg = sum(
        relevance_scores.get(rid, 0.0) / math.log2(i + 2)
        for i, rid in enumerate(retrieved_ids[:k])
    )
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(score / math.log2(i + 2) for i, score in enumerate(ideal_scores))
    return dcg / idcg if idcg > 0 else 0.0


def check_regression(
    current: dict[str, float],
    previous: dict[str, float],
    threshold: float = 0.05,
) -> list[dict]:
    """Return a list of regression events where a metric dropped by > threshold.

    Args:
        current:   metric_name → value for the new job
        previous:  metric_name → value for the reference (previous) job
        threshold: fractional drop that triggers a regression (default 5%)

    Returns:
        List of dicts: {metric, prev_value, curr_value, drop_pct}
        Empty list means no regression.
    """
    regressions: list[dict] = []
    for metric, curr_val in current.items():
        prev_val = previous.get(metric)
        if prev_val is None or prev_val == 0.0:
            continue
        drop_pct = (prev_val - curr_val) / prev_val
        if drop_pct > threshold:
            regressions.append(
                {
                    "metric": metric,
                    "prev_value": round(prev_val, 6),
                    "curr_value": round(curr_val, 6),
                    "drop_pct": round(drop_pct * 100, 2),
                }
            )
    return regressions
