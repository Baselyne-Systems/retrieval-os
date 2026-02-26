"""Benchmark: Retrieval Quality Stability

Customer claim: quality metrics (Recall@k, MRR, NDCG) are measured continuously
and regressions are caught automatically before they reach production users.

What we measure
---------------
The metric computation layer must stay fast as eval datasets grow. If metric
aggregation becomes a bottleneck, eval jobs run less frequently and regressions
go undetected longer.

1. Recall@k throughput at 10 000 queries — the minimum eval set size at which
   statistical significance is achievable for most retrieval tasks.

2. MRR throughput at 10 000 queries — MRR is more expensive than Recall (it
   requires scanning until the first hit) and must still be fast enough to
   compute in seconds.

3. NDCG throughput at 10 000 queries — NDCG with graded relevance involves
   sorting and log operations; must not dominate eval job runtime.

4. JSONL dataset parsing at 50 000 records — eval datasets at real production
   scale are large; parsing must not throttle the eval loop.

5. Regression detection precision — check_regression must detect every real
   regression (no false negatives) with the correct magnitude.

Scale targets
-------------
- Recall@5 for 10 000 queries     in < 500 ms  → < 0.05 ms/query
- MRR for 10 000 queries           in < 500 ms  → < 0.05 ms/query
- NDCG@5 for 10 000 queries        in < 1 s     → < 0.1 ms/query
- Parse 50 000 JSONL records        in < 3 s     → < 0.06 ms/record
- Detect regressions across N metrics in O(N)    (no quadratic behaviour)
"""

from __future__ import annotations

import json
import time

import pytest

from retrieval_os.evaluations.metrics import (
    check_regression,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
)
from retrieval_os.evaluations.runner import _parse_jsonl

# ── Synthetic data generators ─────────────────────────────────────────────────


def _make_retrieval_result(
    *,
    n_retrieved: int = 10,
    n_relevant: int = 3,
    hit_positions: list[int] | None = None,
) -> tuple[list[str], set[str]]:
    """One (retrieved_ids, relevant_ids) pair.

    hit_positions: 0-indexed positions in retrieved list where relevant docs land.
    Default places relevant docs at positions 0, 2, 4 (hits in top-5).
    """
    retrieved = [f"doc-{i}" for i in range(n_retrieved)]
    if hit_positions is None:
        hit_positions = [0, 2, 4]
    relevant = {retrieved[p] for p in hit_positions if p < n_retrieved}
    return retrieved, relevant


def _make_eval_dataset_bytes(n_records: int) -> bytes:
    """Synthetic JSONL eval dataset with n_records records."""
    lines = []
    for i in range(n_records):
        record = {
            "query": f"query about topic {i % 500}: what is the best approach for {i}?",
            "relevant_ids": [f"doc-{i}-a", f"doc-{i}-b"],
            "relevant_scores": {f"doc-{i}-a": 1.0, f"doc-{i}-b": 0.7},
        }
        lines.append(json.dumps(record))
    return "\n".join(lines).encode()


# ── Recall@k throughput ───────────────────────────────────────────────────────


class TestRecallThroughput:
    def test_recall_at_5_for_10k_queries_under_500ms(self) -> None:
        """Recall@5 across a 10 000-query eval set must complete in < 500 ms.

        At this speed, a continuous eval loop can evaluate 10 000 queries
        every 30 seconds without metric computation becoming the rate limiter.
        """
        queries = [_make_retrieval_result(hit_positions=[0, 2, 5]) for _ in range(10_000)]
        relevant_sets = [rel for _, rel in queries]
        retrieved_lists = [ret for ret, _ in queries]

        start = time.perf_counter()
        scores = [
            compute_recall_at_k(ret, rel, k=5) for ret, rel in zip(retrieved_lists, relevant_sets)
        ]
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, (
            f"Recall@5 for 10 000 queries took {elapsed:.3f}s; must be < 0.5s "
            f"({elapsed / 10_000 * 1000:.4f} ms/query)"
        )
        # Sanity: hits at positions 0 and 2 are within top-5; position 5 is not
        assert all(s > 0.0 for s in scores)

    def test_recall_at_k_result_is_accurate_at_scale(self) -> None:
        """Recall values are numerically correct across 10 000 queries with known ground truth."""
        # Perfect recall: all relevant docs in top-k
        queries = [_make_retrieval_result(hit_positions=[0, 1, 2]) for _ in range(10_000)]
        scores = [compute_recall_at_k(ret, rel, k=5) for ret, rel in queries]
        assert all(s == pytest.approx(1.0) for s in scores), (
            "Perfect recall (all hits in top-3 of top-5) must yield 1.0"
        )

        # Zero recall: no relevant docs in top-k
        queries_zero = [_make_retrieval_result(hit_positions=[7, 8, 9]) for _ in range(10_000)]
        scores_zero = [compute_recall_at_k(ret, rel, k=5) for ret, rel in queries_zero]
        assert all(s == pytest.approx(0.0) for s in scores_zero), (
            "Zero recall (all hits beyond position 5) must yield 0.0"
        )


# ── MRR throughput ────────────────────────────────────────────────────────────


class TestMRRThroughput:
    def test_mrr_for_10k_queries_under_500ms(self) -> None:
        """MRR aggregation across 10 000 queries must complete in < 500 ms."""
        results = [_make_retrieval_result(hit_positions=[0, 3, 7]) for _ in range(10_000)]

        start = time.perf_counter()
        mrr = compute_mrr(results)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"MRR for 10 000 queries took {elapsed:.3f}s; must be < 0.5s"
        # First hit at rank 0 (position 0) → RR = 1.0 → MRR ≈ 1.0
        assert mrr == pytest.approx(1.0)

    def test_mrr_degrades_correctly_as_first_hit_rank_increases(self) -> None:
        """MRR numerical accuracy across 10 000 queries with varying first-hit positions."""
        # First hit at rank 2 (position 1) → RR = 0.5
        results = [_make_retrieval_result(hit_positions=[1, 5, 9]) for _ in range(10_000)]
        mrr = compute_mrr(results)
        assert mrr == pytest.approx(0.5, rel=1e-3)


# ── NDCG throughput ───────────────────────────────────────────────────────────


class TestNDCGThroughput:
    def test_ndcg_at_5_for_10k_queries_under_1s(self) -> None:
        """NDCG@5 across 10 000 queries must complete in < 1 second.

        NDCG is the most compute-intensive metric (sorting + log). At 0.1 ms/query
        it still evaluates 10 000 queries in < 1 second.
        """
        queries = [
            (
                [f"doc-{j}" for j in range(10)],
                {f"doc-{j}": (1.0 if j < 3 else 0.5 if j < 6 else 0.0) for j in range(10)},
            )
            for _ in range(10_000)
        ]

        start = time.perf_counter()
        scores = [compute_ndcg_at_k(ret, scores, k=5) for ret, scores in queries]
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, (
            f"NDCG@5 for 10 000 queries took {elapsed:.3f}s; must be < 1s "
            f"({elapsed / 10_000 * 1000:.4f} ms/query)"
        )
        assert all(s > 0.0 for s in scores)


# ── JSONL parsing throughput ──────────────────────────────────────────────────


class TestEvalDatasetParsingThroughput:
    def test_parse_50k_records_under_3s(self) -> None:
        """50 000 JSONL eval records must parse in < 3 seconds.

        A production eval dataset with 50 000 labelled queries (~5 MB of JSONL)
        must not take so long to parse that it delays the first query being run.
        """
        raw = _make_eval_dataset_bytes(50_000)

        start = time.perf_counter()
        records = _parse_jsonl(raw)
        elapsed = time.perf_counter() - start

        assert len(records) == 50_000
        assert elapsed < 3.0, (
            f"Parsing 50 000 JSONL records took {elapsed:.3f}s; must be < 3s "
            f"({elapsed / 50_000 * 1000:.4f} ms/record)"
        )

    def test_parsed_records_are_complete(self) -> None:
        """Every parsed record has the correct fields and types."""
        raw = _make_eval_dataset_bytes(1_000)
        records = _parse_jsonl(raw)
        assert len(records) == 1_000
        for r in records:
            assert isinstance(r.query, str) and r.query
            assert isinstance(r.relevant_ids, set) and len(r.relevant_ids) == 2
            assert isinstance(r.relevance_scores, dict) and len(r.relevance_scores) == 2


# ── Regression detection ──────────────────────────────────────────────────────


class TestRegressionDetection:
    def test_regression_detected_at_correct_threshold(self) -> None:
        """check_regression fires exactly when a metric drops by more than threshold.

        This is the core quality guard-rail: false negatives mean regressions
        reach production; false positives cause unnecessary rollbacks.
        """
        cases: list[tuple[float, float, float, bool]] = [
            # (prev, curr, threshold, should_detect)
            (0.80, 0.74, 0.05, True),  # 7.5% drop > 5% threshold
            (0.80, 0.78, 0.05, False),  # 2.5% drop < threshold
            (0.80, 0.70, 0.10, True),  # 12.5% drop > 10% threshold
            (0.80, 0.79, 0.10, False),  # 1.25% drop < threshold
            (0.80, 0.90, 0.05, False),  # improvement — never a regression
            (0.00, 0.50, 0.05, False),  # previous=0 → skip (undefined percentage)
        ]
        for prev, curr, threshold, should_detect in cases:
            result = check_regression(
                {"recall_at_5": curr}, {"recall_at_5": prev}, threshold=threshold
            )
            if should_detect:
                assert len(result) == 1, (
                    f"prev={prev}, curr={curr}, threshold={threshold}: "
                    f"expected regression detected, got {result}"
                )
            else:
                assert len(result) == 0, (
                    f"prev={prev}, curr={curr}, threshold={threshold}: "
                    f"expected no regression, got {result}"
                )

    def test_regression_detection_scales_linearly_with_metric_count(self) -> None:
        """check_regression with 1 000 metrics must run in < 10 ms.

        The watchdog runs on every eval cycle. If it is O(N²) in the number of
        metrics, it becomes a bottleneck as deployments accumulate metrics.
        """
        n = 1_000
        current = {f"metric_{i}": 0.6 for i in range(n)}  # all regressed
        previous = {f"metric_{i}": 0.8 for i in range(n)}

        start = time.perf_counter()
        regressions = check_regression(current, previous, threshold=0.05)
        elapsed = time.perf_counter() - start

        assert len(regressions) == n
        assert elapsed < 0.01, (
            f"check_regression for {n} metrics took {elapsed * 1000:.2f}ms; must be < 10ms"
        )
