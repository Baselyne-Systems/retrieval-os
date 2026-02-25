"""Embed Router — dispatches embed requests to the correct provider.

Each provider is instantiated lazily and cached at module level.
Phase 3 implements sentence_transformers and openai.
clip, whisper, video_frame stubs are included; full implementations in Phase 8.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from retrieval_os.core import metrics
from retrieval_os.core.circuit_breaker import get_embed_breaker
from retrieval_os.core.config import settings
from retrieval_os.core.exceptions import EmbeddingProviderError

log = logging.getLogger(__name__)

# ── Provider registry ────────────────────────────────────────────────────────

_st_model: Any = None   # sentence_transformers SentenceTransformer
_oa_client: Any = None  # openai AsyncOpenAI


def _get_st_model(model_name: str) -> Any:
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            _st_model = SentenceTransformer(model_name)
        except ImportError:
            raise EmbeddingProviderError(
                "sentence_transformers is not installed. "
                "Install with: uv sync --extra ml"
            )
    return _st_model


def _get_oa_client() -> Any:
    global _oa_client
    if _oa_client is None:
        try:
            import openai  # type: ignore[import]

            _oa_client = openai.AsyncOpenAI(
                api_key=getattr(settings, "openai_api_key", None)
            )
        except ImportError:
            raise EmbeddingProviderError(
                "openai package is not installed. Install with: pip install openai"
            )
    return _oa_client


# ── Public interface ─────────────────────────────────────────────────────────


async def embed_text(
    texts: list[str],
    *,
    provider: str,
    model: str,
    normalize: bool = True,
    batch_size: int = 32,
) -> list[list[float]]:
    """Return a list of embedding vectors, one per input text.

    Args:
        texts:      Input strings to embed.
        provider:   One of VALID_PROVIDERS from validators.
        model:      Provider-specific model identifier.
        normalize:  L2-normalise output vectors.
        batch_size: Number of texts to embed per model call.

    Returns:
        List of float vectors, shape (len(texts), embedding_dimensions).
    """
    start = time.perf_counter()
    breaker = get_embed_breaker(provider)
    try:
        vectors = await breaker.call(
            _dispatch_text,
            texts, provider=provider, model=model,
            normalize=normalize, batch_size=batch_size,
        )
    except EmbeddingProviderError:
        raise
    except Exception as exc:
        metrics.embed_errors_total.labels(provider=provider).inc()
        raise EmbeddingProviderError(
            f"Embedding failed ({provider}/{model}): {exc}"
        ) from exc

    elapsed = time.perf_counter() - start
    metrics.embed_latency_seconds.labels(provider=provider).observe(elapsed)
    metrics.embed_requests_total.labels(provider=provider).inc()
    return vectors


async def _dispatch_text(
    texts: list[str],
    *,
    provider: str,
    model: str,
    normalize: bool,
    batch_size: int,
) -> list[list[float]]:
    if provider == "sentence_transformers":
        return await _embed_sentence_transformers(
            texts, model=model, normalize=normalize, batch_size=batch_size
        )
    if provider == "openai":
        return await _embed_openai(texts, model=model)
    if provider in ("clip", "whisper", "video_frame"):
        raise EmbeddingProviderError(
            f"Provider '{provider}' is not yet implemented in this runtime. "
            "Planned for Phase 8."
        )
    raise EmbeddingProviderError(f"Unknown embedding provider: '{provider}'")


# ── sentence_transformers ─────────────────────────────────────────────────────


async def _embed_sentence_transformers(
    texts: list[str],
    *,
    model: str,
    normalize: bool,
    batch_size: int,
) -> list[list[float]]:
    """Runs ST encode() in a thread pool to avoid blocking the event loop."""

    def _encode() -> list[list[float]]:
        st = _get_st_model(model)
        vecs = st.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return vecs.tolist()

    return await asyncio.to_thread(_encode)


# ── OpenAI ────────────────────────────────────────────────────────────────────


async def _embed_openai(
    texts: list[str],
    *,
    model: str,
) -> list[list[float]]:
    client = _get_oa_client()
    # OpenAI recommends batches ≤ 2048 inputs; we pass through as-is.
    response = await client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]
