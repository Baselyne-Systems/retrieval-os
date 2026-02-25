"""Unit tests for the Evaluation Engine domain (no live DB or S3)."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime

import pytest

from retrieval_os.evaluations.metrics import (
    check_regression,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
)
from retrieval_os.evaluations.models import EvalJob, EvalJobStatus
from retrieval_os.evaluations.runner import EvalRecord, _parse_jsonl

# ── EvalJobStatus enum ─────────────────────────────────────────────────────────


class TestEvalJobStatus:
    def test_all_statuses_exist(self) -> None:
        values = {s.value for s in EvalJobStatus}
        assert values == {"QUEUED", "RUNNING", "COMPLETED", "FAILED"}

    def test_str_comparison(self) -> None:
        assert EvalJobStatus.QUEUED == "QUEUED"
        assert EvalJobStatus.COMPLETED == "COMPLETED"


# ── EvalJob ORM model ──────────────────────────────────────────────────────────


class TestEvalJobModel:
    def test_constructor(self) -> None:
        now = datetime.now(UTC)
        job = EvalJob(
            id="test-id",
            plan_name="wiki-search",
            plan_version=3,
            status=EvalJobStatus.QUEUED,
            dataset_uri="s3://bucket/eval/wiki.jsonl",
            top_k=10,
            created_at=now,
            created_by="alice",
        )
        assert job.plan_name == "wiki-search"
        assert job.plan_version == 3
        assert job.status == "QUEUED"
        assert job.recall_at_5 is None
        assert job.regression_detected is None


# ── Schemas ────────────────────────────────────────────────────────────────────


class TestQueueEvalJobRequest:
    def test_valid_request(self) -> None:
        from retrieval_os.evaluations.schemas import QueueEvalJobRequest

        req = QueueEvalJobRequest(
            plan_name="wiki-search",
            plan_version=1,
            dataset_uri="s3://bucket/eval/wiki.jsonl",
            top_k=10,
            created_by="alice",
        )
        assert req.plan_name == "wiki-search"
        assert req.top_k == 10

    def test_top_k_must_be_positive(self) -> None:
        from retrieval_os.evaluations.schemas import QueueEvalJobRequest

        with pytest.raises(Exception):
            QueueEvalJobRequest(
                plan_name="x",
                plan_version=1,
                dataset_uri="s3://bucket/x.jsonl",
                top_k=0,
                created_by="alice",
            )

    def test_plan_version_must_be_positive(self) -> None:
        from retrieval_os.evaluations.schemas import QueueEvalJobRequest

        with pytest.raises(Exception):
            QueueEvalJobRequest(
                plan_name="x",
                plan_version=0,
                dataset_uri="s3://bucket/x.jsonl",
                top_k=10,
                created_by="alice",
            )


# ── Metrics: compute_recall_at_k ──────────────────────────────────────────────


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert compute_recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_zero_recall(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert compute_recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_recall(self) -> None:
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b", "c", "d"}
        # 2 hits out of 4 relevant
        assert compute_recall_at_k(retrieved, relevant, k=4) == 0.5

    def test_recall_capped_at_1(self) -> None:
        # More relevant than k — max achievable recall is k/|relevant|
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c", "d", "e"}
        recall = compute_recall_at_k(retrieved, relevant, k=2)
        assert recall == pytest.approx(2 / 5)

    def test_k_limits_window(self) -> None:
        retrieved = ["x", "a", "b"]
        relevant = {"a", "b"}
        assert compute_recall_at_k(retrieved, relevant, k=1) == 0.0
        assert compute_recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)
        assert compute_recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_empty_relevant_returns_zero(self) -> None:
        assert compute_recall_at_k(["a", "b"], set(), k=2) == 0.0

    def test_empty_retrieved_returns_zero(self) -> None:
        assert compute_recall_at_k([], {"a", "b"}, k=10) == 0.0


# ── Metrics: compute_mrr ──────────────────────────────────────────────────────


class TestMRR:
    def test_first_hit_at_rank_1(self) -> None:
        results = [(["a", "b", "c"], {"a"})]
        assert compute_mrr(results) == pytest.approx(1.0)

    def test_first_hit_at_rank_2(self) -> None:
        results = [(["x", "a", "b"], {"a"})]
        assert compute_mrr(results) == pytest.approx(0.5)

    def test_no_hit(self) -> None:
        results = [(["x", "y", "z"], {"a"})]
        assert compute_mrr(results) == pytest.approx(0.0)

    def test_mean_over_multiple_queries(self) -> None:
        results = [
            (["a", "b"], {"a"}),  # RR = 1.0
            (["x", "a"], {"a"}),  # RR = 0.5
        ]
        assert compute_mrr(results) == pytest.approx(0.75)

    def test_empty_results_returns_zero(self) -> None:
        assert compute_mrr([]) == 0.0

    def test_uses_first_relevant_hit(self) -> None:
        results = [(["x", "a", "b"], {"a", "b"})]
        # First hit is "a" at rank 2 → RR = 0.5
        assert compute_mrr(results) == pytest.approx(0.5)


# ── Metrics: compute_ndcg_at_k ────────────────────────────────────────────────


class TestNDCG:
    def test_perfect_ndcg(self) -> None:
        retrieved = ["a", "b", "c"]
        scores = {"a": 1.0, "b": 1.0, "c": 1.0}
        # Ideal order matches retrieved → NDCG = 1.0
        assert compute_ndcg_at_k(retrieved, scores, k=3) == pytest.approx(1.0)

    def test_zero_ndcg(self) -> None:
        retrieved = ["x", "y", "z"]
        scores = {"a": 1.0, "b": 1.0}
        assert compute_ndcg_at_k(retrieved, scores, k=3) == pytest.approx(0.0)

    def test_graded_relevance(self) -> None:
        # "b" is more relevant than "a" but "a" is retrieved first
        retrieved = ["a", "b"]
        scores = {"a": 0.5, "b": 1.0}
        ndcg = compute_ndcg_at_k(retrieved, scores, k=2)
        # DCG = 0.5/log2(2) + 1.0/log2(3) = 0.5 + 0.631 = 1.131
        # IDCG = 1.0/log2(2) + 0.5/log2(3) = 1.0 + 0.315 = 1.315
        # NDCG = 1.131 / 1.315 ≈ 0.86
        assert ndcg < 1.0
        assert ndcg > 0.7

    def test_empty_scores_returns_zero(self) -> None:
        assert compute_ndcg_at_k(["a", "b"], {}, k=5) == 0.0

    def test_k_limits_window(self) -> None:
        retrieved = ["x", "a", "b"]
        scores = {"a": 1.0, "b": 1.0}
        ndcg_k1 = compute_ndcg_at_k(retrieved, scores, k=1)
        ndcg_k3 = compute_ndcg_at_k(retrieved, scores, k=3)
        assert ndcg_k1 == pytest.approx(0.0)
        assert ndcg_k3 > 0.0


# ── Metrics: check_regression ─────────────────────────────────────────────────


class TestCheckRegression:
    def test_no_regression(self) -> None:
        current = {"recall_at_5": 0.8, "mrr": 0.7}
        previous = {"recall_at_5": 0.8, "mrr": 0.7}
        assert check_regression(current, previous) == []

    def test_regression_detected(self) -> None:
        current = {"recall_at_5": 0.7, "mrr": 0.6}
        previous = {"recall_at_5": 0.8, "mrr": 0.7}
        regressions = check_regression(current, previous, threshold=0.05)
        assert len(regressions) == 2
        metrics = {r["metric"] for r in regressions}
        assert "recall_at_5" in metrics
        assert "mrr" in metrics

    def test_small_drop_below_threshold(self) -> None:
        current = {"recall_at_5": 0.78}
        previous = {"recall_at_5": 0.80}
        # Drop = 2.5%, threshold = 5% → no regression
        assert check_regression(current, previous, threshold=0.05) == []

    def test_improvement_not_regression(self) -> None:
        current = {"recall_at_5": 0.9}
        previous = {"recall_at_5": 0.8}
        assert check_regression(current, previous) == []

    def test_missing_previous_metric_skipped(self) -> None:
        current = {"recall_at_5": 0.5, "new_metric": 0.3}
        previous = {"recall_at_5": 0.8}
        regressions = check_regression(current, previous, threshold=0.05)
        # new_metric has no previous → skipped
        assert all(r["metric"] == "recall_at_5" for r in regressions)

    def test_previous_zero_skipped(self) -> None:
        current = {"recall_at_5": 0.1}
        previous = {"recall_at_5": 0.0}
        assert check_regression(current, previous) == []

    def test_regression_detail_fields(self) -> None:
        current = {"mrr": 0.5}
        previous = {"mrr": 0.8}
        regressions = check_regression(current, previous, threshold=0.05)
        assert len(regressions) == 1
        r = regressions[0]
        assert r["metric"] == "mrr"
        assert r["prev_value"] == pytest.approx(0.8)
        assert r["curr_value"] == pytest.approx(0.5)
        assert r["drop_pct"] == pytest.approx(37.5, abs=0.1)


# ── Dataset parsing ───────────────────────────────────────────────────────────


class TestParseJsonl:
    def _jsonl(self, records: list[dict]) -> bytes:
        return "\n".join(json.dumps(r) for r in records).encode()

    def test_basic_parsing(self) -> None:
        raw = self._jsonl([
            {"query": "what is X?", "relevant_ids": ["id1", "id2"]},
        ])
        records = _parse_jsonl(raw)
        assert len(records) == 1
        assert records[0].query == "what is X?"
        assert records[0].relevant_ids == {"id1", "id2"}

    def test_default_relevance_scores(self) -> None:
        raw = self._jsonl([
            {"query": "q", "relevant_ids": ["a", "b"]},
        ])
        records = _parse_jsonl(raw)
        assert records[0].relevance_scores == {"a": 1.0, "b": 1.0}

    def test_custom_relevance_scores(self) -> None:
        raw = self._jsonl([
            {
                "query": "q",
                "relevant_ids": ["a", "b"],
                "relevant_scores": {"a": 1.0, "b": 0.5},
            }
        ])
        records = _parse_jsonl(raw)
        assert records[0].relevance_scores == {"a": 1.0, "b": 0.5}

    def test_gzip_compressed(self) -> None:
        raw = self._jsonl([{"query": "q", "relevant_ids": ["id1"]}])
        compressed = gzip.compress(raw)
        records = _parse_jsonl(compressed)
        assert len(records) == 1

    def test_skips_missing_query(self) -> None:
        raw = self._jsonl([
            {"relevant_ids": ["id1"]},
            {"query": "good q", "relevant_ids": ["id2"]},
        ])
        records = _parse_jsonl(raw)
        assert len(records) == 1
        assert records[0].query == "good q"

    def test_skips_missing_relevant_ids(self) -> None:
        raw = self._jsonl([
            {"query": "bad q", "relevant_ids": []},
            {"query": "good q", "relevant_ids": ["id1"]},
        ])
        records = _parse_jsonl(raw)
        assert len(records) == 1

    def test_skips_blank_lines(self) -> None:
        raw = b'{"query": "q", "relevant_ids": ["id1"]}\n\n'
        records = _parse_jsonl(raw)
        assert len(records) == 1

    def test_multiple_records(self) -> None:
        raw = self._jsonl([
            {"query": f"q{i}", "relevant_ids": [f"id{i}"]}
            for i in range(5)
        ])
        records = _parse_jsonl(raw)
        assert len(records) == 5


# ── EvalRecord dataclass ──────────────────────────────────────────────────────


class TestEvalRecord:
    def test_fields(self) -> None:
        record = EvalRecord(
            query="how does X work?",
            relevant_ids={"doc1", "doc2"},
            relevance_scores={"doc1": 1.0, "doc2": 0.8},
        )
        assert "doc1" in record.relevant_ids
        assert record.relevance_scores["doc2"] == 0.8
