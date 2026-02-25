"""Reranker — second-pass scoring of ANN hits.

Supported providers
-------------------
``cross_encoder:<model>``
    Runs a sentence-transformers CrossEncoder locally in a thread pool.
    Requires the ``ml`` extras: ``uv sync --extra ml``.
    Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``

``cohere:<model>``
    Calls the Cohere Rerank v2 API (async).
    Requires ``pip install cohere`` and ``COHERE_API_KEY`` in the environment.
    Default model: ``rerank-english-v3.0``

Usage
-----
The ``reranker`` field on a PlanVersion is a ``"provider:model"`` string,
e.g. ``"cross_encoder:cross-encoder/ms-marco-MiniLM-L-6-v2"``.
A bare provider name (``"cross_encoder"``) uses the default model.

Graceful degradation
--------------------
If reranking fails for any reason (model not installed, API error, network
timeout) the function logs a warning and returns the original hits truncated
to ``top_k``.  The serving path stays live.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from retrieval_os.core.exceptions import EmbeddingProviderError
from retrieval_os.serving.index_proxy import IndexHit

log = logging.getLogger(__name__)

# Lazy singleton cache: model_name → CrossEncoder instance
_ce_models: dict[str, Any] = {}

_DEFAULT_MODELS: dict[str, str] = {
    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cohere": "rerank-english-v3.0",
}


# ── Provider dispatch ─────────────────────────────────────────────────────────


def _parse_reranker(reranker: str) -> tuple[str, str]:
    """Return ``(provider, model)`` from a ``"provider:model"`` string."""
    provider, _, model = reranker.partition(":")
    if not model:
        model = _DEFAULT_MODELS.get(provider, "")
    return provider.strip(), model.strip()


async def rerank(
    hits: list[IndexHit],
    *,
    query: str,
    reranker: str,
    top_k: int,
) -> list[IndexHit]:
    """Re-score *hits* with a cross-encoder or Cohere and return top *top_k*.

    Falls back to original score-sorted truncation on any error.
    """
    if not hits:
        return hits

    provider, model = _parse_reranker(reranker)

    try:
        if provider == "cross_encoder":
            return await _rerank_cross_encoder(hits, query=query, model=model, top_k=top_k)
        if provider == "cohere":
            return await _rerank_cohere(hits, query=query, model=model, top_k=top_k)
    except EmbeddingProviderError:
        raise
    except Exception as exc:
        log.warning(
            "rerank.failed",
            extra={"provider": provider, "model": model, "error": str(exc)},
        )
        return hits[:top_k]

    log.warning("rerank.unknown_provider", extra={"provider": provider})
    return hits[:top_k]


# ── Cross-encoder ─────────────────────────────────────────────────────────────


def _get_cross_encoder(model_name: str) -> Any:
    if model_name not in _ce_models:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
        except ImportError:
            raise EmbeddingProviderError(
                "sentence_transformers is not installed. "
                "Install with: uv sync --extra ml"
            )
        _ce_models[model_name] = CrossEncoder(model_name)
    return _ce_models[model_name]


async def _rerank_cross_encoder(
    hits: list[IndexHit],
    *,
    query: str,
    model: str,
    top_k: int,
) -> list[IndexHit]:
    """Score query-passage pairs with a cross-encoder in a thread pool."""

    def _score() -> list[float]:
        ce = _get_cross_encoder(model)
        pairs = [(query, h.payload.get("text", "")) for h in hits]
        raw = ce.predict(pairs)
        # CrossEncoder.predict returns ndarray or list depending on version
        try:
            return raw.tolist()
        except AttributeError:
            return list(raw)

    raw_scores = await asyncio.to_thread(_score)

    reranked = [
        IndexHit(id=h.id, score=float(s), payload=h.payload)
        for h, s in zip(hits, raw_scores)
    ]
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked[:top_k]


# ── Cohere ────────────────────────────────────────────────────────────────────


async def _rerank_cohere(
    hits: list[IndexHit],
    *,
    query: str,
    model: str,
    top_k: int,
) -> list[IndexHit]:
    """Call the Cohere Rerank v2 API."""
    try:
        import cohere  # type: ignore[import]
    except ImportError:
        raise EmbeddingProviderError(
            "cohere package is not installed. Install with: pip install cohere"
        )

    from retrieval_os.core.config import settings

    api_key: str | None = getattr(settings, "cohere_api_key", None)
    client = cohere.AsyncClientV2(api_key=api_key)

    documents = [h.payload.get("text", "") for h in hits]
    result = await client.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_k,
    )

    return [
        IndexHit(
            id=hits[r.index].id,
            score=float(r.relevance_score),
            payload=hits[r.index].payload,
        )
        for r in result.results
    ]
