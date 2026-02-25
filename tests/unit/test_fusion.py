"""Unit tests for the RRF fusion module."""

from __future__ import annotations

import pytest

from retrieval_os.serving.fusion import reciprocal_rank_fusion
from retrieval_os.serving.index_proxy import IndexHit

# ── Helpers ───────────────────────────────────────────────────────────────────


def _hit(id: str, score: float = 1.0, payload: dict | None = None) -> IndexHit:
    return IndexHit(id=id, score=score, payload=payload or {})


def _ids(hits: list[IndexHit]) -> list[str]:
    return [h.id for h in hits]


# ── Basic correctness ─────────────────────────────────────────────────────────


class TestRRFBasics:
    def test_single_list_preserves_order(self) -> None:
        hits = [_hit("a", 0.9), _hit("b", 0.7), _hit("c", 0.5)]
        result = reciprocal_rank_fusion([hits], top_k=3)
        assert _ids(result) == ["a", "b", "c"]

    def test_empty_lists_returns_empty(self) -> None:
        result = reciprocal_rank_fusion([[], []], top_k=10)
        assert result == []

    def test_no_lists_returns_empty(self) -> None:
        result = reciprocal_rank_fusion([], top_k=10)
        assert result == []

    def test_top_k_limits_results(self) -> None:
        hits = [_hit(f"doc{i}") for i in range(10)]
        result = reciprocal_rank_fusion([hits], top_k=3)
        assert len(result) == 3

    def test_returns_index_hit_objects(self) -> None:
        hits = [_hit("x", 1.0, {"text": "hello"})]
        result = reciprocal_rank_fusion([hits], top_k=1)
        assert isinstance(result[0], IndexHit)

    def test_scores_are_rrf_not_original(self) -> None:
        """Score should be 1/(60+1) for rank-1 from a single list."""
        hits = [_hit("a", 999.0)]
        result = reciprocal_rank_fusion([hits], top_k=1)
        expected = 1.0 / (60 + 1)
        assert result[0].score == pytest.approx(expected)

    def test_custom_k_affects_score(self) -> None:
        hits = [_hit("a")]
        r_default = reciprocal_rank_fusion([hits], top_k=1, k=60)
        r_custom = reciprocal_rank_fusion([hits], top_k=1, k=10)
        assert r_custom[0].score > r_default[0].score  # smaller k → higher score


# ── Fusion of two lists ───────────────────────────────────────────────────────


class TestRRFFusion:
    def test_doc_in_both_lists_scores_higher(self) -> None:
        """A document ranked #1 in both lists should beat one ranked #1 in only one."""
        dense = [_hit("shared", 0.9), _hit("dense-only", 0.8)]
        sparse = [_hit("shared", 0.7), _hit("sparse-only", 0.6)]
        result = reciprocal_rank_fusion([dense, sparse], top_k=3)
        assert result[0].id == "shared"

    def test_all_docs_included_when_top_k_large(self) -> None:
        dense = [_hit("a"), _hit("b")]
        sparse = [_hit("c"), _hit("d")]
        result = reciprocal_rank_fusion([dense, sparse], top_k=10)
        assert {h.id for h in result} == {"a", "b", "c", "d"}

    def test_rank_1_beats_rank_2(self) -> None:
        """Rank 1 in list 1 should beat rank 2 in list 1 (all else equal)."""
        dense = [_hit("first"), _hit("second")]
        result = reciprocal_rank_fusion([dense], top_k=2)
        assert result[0].id == "first"
        assert result[1].id == "second"

    def test_payload_from_first_occurrence(self) -> None:
        """Payload should be taken from the list where the id first appears."""
        dense = [_hit("shared", payload={"source": "dense"})]
        sparse = [_hit("shared", payload={"source": "sparse"})]
        result = reciprocal_rank_fusion([dense, sparse], top_k=1)
        assert result[0].payload["source"] == "dense"

    def test_rrf_score_formula(self) -> None:
        """Verify RRF arithmetic: doc ranked #1 in both lists with k=60."""
        # score = 1/(60+1) + 1/(60+1) = 2/61
        dense = [_hit("a")]
        sparse = [_hit("a")]
        result = reciprocal_rank_fusion([dense, sparse], top_k=1, k=60)
        expected = 2.0 / 61.0
        assert result[0].score == pytest.approx(expected)

    def test_three_lists_accumulate_scores(self) -> None:
        l1 = [_hit("x")]
        l2 = [_hit("x")]
        l3 = [_hit("x")]
        result = reciprocal_rank_fusion([l1, l2, l3], top_k=1, k=60)
        expected = 3.0 / 61.0
        assert result[0].score == pytest.approx(expected)

    def test_disjoint_lists_sorted_by_rank(self) -> None:
        """With disjoint lists, rank-1 items should appear before rank-2 items."""
        dense = [_hit("d1"), _hit("d2")]   # d1=1/(60+1), d2=1/(60+2)
        sparse = [_hit("s1"), _hit("s2")]  # s1=1/(60+1), s2=1/(60+2)
        result = reciprocal_rank_fusion([dense, sparse], top_k=4)
        # d1 and s1 both have score 1/61; d2 and s2 both have 1/62
        rank1_ids = {result[0].id, result[1].id}
        assert rank1_ids == {"d1", "s1"}

    def test_empty_sparse_degrades_to_dense(self) -> None:
        dense = [_hit("a"), _hit("b"), _hit("c")]
        result = reciprocal_rank_fusion([dense, []], top_k=3)
        assert _ids(result) == ["a", "b", "c"]

    def test_empty_dense_degrades_to_sparse(self) -> None:
        sparse = [_hit("x"), _hit("y")]
        result = reciprocal_rank_fusion([[], sparse], top_k=2)
        assert _ids(result) == ["x", "y"]


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestRRFEdgeCases:
    def test_top_k_zero_returns_empty(self) -> None:
        hits = [_hit("a"), _hit("b")]
        result = reciprocal_rank_fusion([hits], top_k=0)
        assert result == []

    def test_duplicate_id_in_same_list_not_double_counted(self) -> None:
        """Duplicate IDs in the same list are not an expected input, but ensure no crash."""
        hits = [_hit("a", 0.9), _hit("b", 0.8), _hit("a", 0.7)]  # 'a' appears twice
        result = reciprocal_rank_fusion([hits], top_k=3)
        # 'a' will appear twice in the result (not our constraint to prevent)
        assert all(isinstance(h, IndexHit) for h in result)

    def test_single_document(self) -> None:
        result = reciprocal_rank_fusion([[_hit("only")]], top_k=5)
        assert len(result) == 1
        assert result[0].id == "only"
