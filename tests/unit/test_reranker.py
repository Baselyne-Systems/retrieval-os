"""Unit tests for the Reranker module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval_os.serving.index_proxy import IndexHit
from retrieval_os.serving.reranker import _parse_reranker, rerank

# ── _parse_reranker ───────────────────────────────────────────────────────────


class TestParseReranker:
    def test_provider_and_model(self) -> None:
        provider, model = _parse_reranker("cross_encoder:my-model")
        assert provider == "cross_encoder"
        assert model == "my-model"

    def test_bare_cross_encoder_uses_default(self) -> None:
        provider, model = _parse_reranker("cross_encoder")
        assert provider == "cross_encoder"
        assert "ms-marco" in model

    def test_bare_cohere_uses_default(self) -> None:
        provider, model = _parse_reranker("cohere")
        assert provider == "cohere"
        assert "rerank" in model

    def test_colon_in_model_name(self) -> None:
        # Only the first colon is treated as separator
        provider, model = _parse_reranker("cross_encoder:org/model:variant")
        assert provider == "cross_encoder"
        assert model == "org/model:variant"

    def test_whitespace_stripped(self) -> None:
        provider, model = _parse_reranker("  cross_encoder : my-model  ")
        assert provider == "cross_encoder"
        assert model == "my-model"


# ── rerank — empty / trivial ──────────────────────────────────────────────────


class TestRerankEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_hits_returns_empty(self) -> None:
        result = await rerank([], query="q", reranker="cross_encoder", top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_unknown_provider_falls_back(self) -> None:
        hits = [
            IndexHit(id="a", score=0.9, payload={"text": "alpha"}),
            IndexHit(id="b", score=0.8, payload={"text": "beta"}),
            IndexHit(id="c", score=0.7, payload={"text": "gamma"}),
        ]
        result = await rerank(hits, query="q", reranker="unknown_provider", top_k=2)
        assert len(result) == 2
        assert result[0].id == "a"

    @pytest.mark.asyncio
    async def test_top_k_respected_on_fallback(self) -> None:
        hits = [IndexHit(id=str(i), score=float(i), payload={}) for i in range(10)]
        result = await rerank(hits, query="q", reranker="unknown_provider", top_k=3)
        assert len(result) == 3


# ── rerank — cross_encoder ────────────────────────────────────────────────────


class TestRerankCrossEncoder:
    @pytest.mark.asyncio
    async def test_scores_replaced_and_sorted(self) -> None:
        hits = [
            IndexHit(id="a", score=0.9, payload={"text": "alpha"}),
            IndexHit(id="b", score=0.8, payload={"text": "beta"}),
            IndexHit(id="c", score=0.7, payload={"text": "gamma"}),
        ]
        # CE returns low score for a, high for c → order should flip
        fake_ce = MagicMock()
        fake_ce.predict = MagicMock(return_value=[0.1, 0.5, 0.9])

        with patch(
            "retrieval_os.serving.reranker._get_cross_encoder",
            return_value=fake_ce,
        ):
            result = await rerank(
                hits, query="query", reranker="cross_encoder:test-model", top_k=3
            )

        assert result[0].id == "c"
        assert result[1].id == "b"
        assert result[2].id == "a"

    @pytest.mark.asyncio
    async def test_top_k_truncates(self) -> None:
        hits = [IndexHit(id=str(i), score=float(i), payload={"text": f"text{i}"}) for i in range(5)]
        fake_ce = MagicMock()
        fake_ce.predict = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])

        with patch(
            "retrieval_os.serving.reranker._get_cross_encoder",
            return_value=fake_ce,
        ):
            result = await rerank(
                hits, query="q", reranker="cross_encoder:test-model", top_k=2
            )

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_scores_are_floats(self) -> None:
        hits = [
            IndexHit(id="x", score=0.5, payload={"text": "something"}),
        ]
        import numpy as np

        fake_ce = MagicMock()
        fake_ce.predict = MagicMock(return_value=np.array([0.42]))

        with patch(
            "retrieval_os.serving.reranker._get_cross_encoder",
            return_value=fake_ce,
        ):
            result = await rerank(
                hits, query="q", reranker="cross_encoder:test-model", top_k=1
            )

        assert isinstance(result[0].score, float)
        assert abs(result[0].score - 0.42) < 1e-6

    @pytest.mark.asyncio
    async def test_missing_sentence_transformers_raises_provider_error(self) -> None:
        from retrieval_os.core.exceptions import EmbeddingProviderError

        hits = [IndexHit(id="a", score=0.9, payload={"text": "text"})]

        with patch(
            "retrieval_os.serving.reranker._get_cross_encoder",
            side_effect=EmbeddingProviderError("not installed"),
        ):
            with pytest.raises(EmbeddingProviderError):
                await rerank(
                    hits, query="q", reranker="cross_encoder:test-model", top_k=1
                )

    @pytest.mark.asyncio
    async def test_runtime_error_triggers_graceful_degradation(self) -> None:
        hits = [
            IndexHit(id="a", score=0.9, payload={"text": "text"}),
            IndexHit(id="b", score=0.8, payload={"text": "other"}),
        ]
        fake_ce = MagicMock()
        fake_ce.predict = MagicMock(side_effect=RuntimeError("CUDA OOM"))

        with patch(
            "retrieval_os.serving.reranker._get_cross_encoder",
            return_value=fake_ce,
        ):
            result = await rerank(
                hits, query="q", reranker="cross_encoder:test-model", top_k=2
            )

        # Graceful degradation: returns original hits
        assert len(result) == 2
        assert result[0].id == "a"

    @pytest.mark.asyncio
    async def test_uses_text_from_payload(self) -> None:
        """The cross-encoder must receive the text field from each hit's payload."""
        hits = [
            IndexHit(id="a", score=0.9, payload={"text": "first passage"}),
            IndexHit(id="b", score=0.8, payload={"text": "second passage"}),
        ]
        captured_pairs: list = []

        def fake_predict(pairs):  # noqa: ANN001
            captured_pairs.extend(pairs)
            return [0.5, 0.5]

        fake_ce = MagicMock()
        fake_ce.predict = fake_predict

        with patch(
            "retrieval_os.serving.reranker._get_cross_encoder",
            return_value=fake_ce,
        ):
            await rerank(hits, query="my query", reranker="cross_encoder:m", top_k=2)

        assert captured_pairs[0] == ("my query", "first passage")
        assert captured_pairs[1] == ("my query", "second passage")


# ── rerank — cohere ───────────────────────────────────────────────────────────


class TestRerankCohere:
    @pytest.mark.asyncio
    async def test_cohere_scores_applied(self) -> None:
        hits = [
            IndexHit(id="a", score=0.9, payload={"text": "alpha"}),
            IndexHit(id="b", score=0.8, payload={"text": "beta"}),
        ]

        mock_result_item_0 = MagicMock()
        mock_result_item_0.index = 1
        mock_result_item_0.relevance_score = 0.95

        mock_result_item_1 = MagicMock()
        mock_result_item_1.index = 0
        mock_result_item_1.relevance_score = 0.3

        mock_rerank_response = MagicMock()
        mock_rerank_response.results = [mock_result_item_0, mock_result_item_1]

        mock_cohere_client = MagicMock()
        mock_cohere_client.rerank = AsyncMock(return_value=mock_rerank_response)

        mock_cohere_module = MagicMock()
        mock_cohere_module.AsyncClientV2 = MagicMock(return_value=mock_cohere_client)

        with patch.dict("sys.modules", {"cohere": mock_cohere_module}):
            result = await rerank(
                hits, query="test query", reranker="cohere:rerank-english-v3.0", top_k=2
            )

        assert result[0].id == "b"  # index=1 → hits[1] = b
        assert result[1].id == "a"  # index=0 → hits[0] = a
        assert abs(result[0].score - 0.95) < 1e-6

    @pytest.mark.asyncio
    async def test_cohere_import_error_raises_provider_error(self) -> None:
        from retrieval_os.core.exceptions import EmbeddingProviderError

        hits = [IndexHit(id="a", score=0.9, payload={"text": "text"})]

        with patch.dict("sys.modules", {"cohere": None}):
            with pytest.raises(EmbeddingProviderError, match="cohere"):
                await rerank(hits, query="q", reranker="cohere:rerank-english-v3.0", top_k=1)
