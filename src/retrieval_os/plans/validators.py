"""Semantic validation and config hashing for retrieval plan configs.

Validation runs at write time (plan creation and new version), so the serving
path can trust that any PlanVersion in the database is contractually valid.
"""

import hashlib
import json
from typing import Any

from retrieval_os.core.exceptions import AppValidationError

# ── Static registries ──────────────────────────────────────────────────────────

VALID_PROVIDERS = {"sentence_transformers", "openai", "clip", "whisper", "video_frame"}
VALID_BACKENDS = {"qdrant", "pgvector"}
VALID_METRICS = {"cosine", "dot", "euclidean"}
VALID_MODALITIES = {"text", "image", "audio", "video"}
VALID_QUANTIZATIONS = {"scalar", "product"}

# Which modalities each provider can embed
PROVIDER_MODALITIES: dict[str, set[str]] = {
    "sentence_transformers": {"text"},
    "openai": {"text"},
    "clip": {"text", "image"},
    "whisper": {"audio"},
    "video_frame": {"video"},
}


def validate_plan_config(config: dict[str, Any]) -> None:
    """
    Validates plan config semantics. Raises AppValidationError with all
    failures collected rather than stopping at the first one.
    """
    errors: list[str] = []

    provider = config.get("embedding_provider", "")
    backend = config.get("index_backend", "")
    metric = config.get("distance_metric", "")
    modalities: list[str] = config.get("modalities", [])
    top_k: int = config.get("top_k", 0)
    rerank_top_k: int | None = config.get("rerank_top_k")
    hybrid_alpha: float | None = config.get("hybrid_alpha")
    quantization: str | None = config.get("quantization")
    cache_ttl: int = config.get("cache_ttl_seconds", 0)

    # Embedding provider
    if provider not in VALID_PROVIDERS:
        errors.append(
            f"embedding_provider '{provider}' is not registered; "
            f"valid values: {sorted(VALID_PROVIDERS)}"
        )

    # Index backend
    if backend not in VALID_BACKENDS:
        errors.append(
            f"index_backend '{backend}' is not registered; valid values: {sorted(VALID_BACKENDS)}"
        )

    # Distance metric
    if metric not in VALID_METRICS:
        errors.append(
            f"distance_metric '{metric}' is not valid; valid values: {sorted(VALID_METRICS)}"
        )

    # Modalities
    if not modalities:
        errors.append("modalities must be non-empty")
    else:
        invalid = set(modalities) - VALID_MODALITIES
        if invalid:
            errors.append(f"unknown modalities: {sorted(invalid)}")
        elif provider in PROVIDER_MODALITIES:
            supported = PROVIDER_MODALITIES[provider]
            unsupported = set(modalities) - supported
            if unsupported:
                errors.append(
                    f"embedding_provider '{provider}' does not support "
                    f"{sorted(unsupported)}; it supports {sorted(supported)}"
                )

    # top_k
    if top_k < 1:
        errors.append("top_k must be >= 1")

    # rerank_top_k
    if rerank_top_k is not None:
        if rerank_top_k < 1:
            errors.append("rerank_top_k must be >= 1")
        elif rerank_top_k > top_k:
            errors.append(f"rerank_top_k ({rerank_top_k}) must be <= top_k ({top_k})")

    # hybrid_alpha
    if hybrid_alpha is not None and not (0.0 <= hybrid_alpha <= 1.0):
        errors.append("hybrid_alpha must be between 0.0 and 1.0")

    # quantization
    if quantization is not None and quantization not in VALID_QUANTIZATIONS:
        errors.append(
            f"quantization '{quantization}' is not valid; "
            f"valid values: {sorted(VALID_QUANTIZATIONS)}"
        )

    # cache_ttl_seconds
    if cache_ttl < 0:
        errors.append("cache_ttl_seconds must be >= 0")

    if errors:
        raise AppValidationError(
            "Plan configuration validation failed",
            detail={"errors": errors},
        )


# ── Config hash ────────────────────────────────────────────────────────────────

_HASH_FIELDS = (
    "embedding_provider",
    "embedding_model",
    "embedding_dimensions",
    "modalities",
    "index_backend",
    "index_collection",
    "distance_metric",
    "quantization",
    "top_k",
    "rerank_top_k",
    "reranker",
    "hybrid_alpha",
    "metadata_filters",
)


def compute_config_hash(config: dict[str, Any]) -> str:
    """
    SHA-256 of a canonical JSON subset of the config fields that define
    retrieval behaviour. Two PlanVersions with the same hash are functionally
    identical (cost config, caching, and comments are excluded).
    """
    canonical: dict[str, Any] = {}
    for field in _HASH_FIELDS:
        value = config.get(field)
        # Normalise lists so order doesn't affect the hash
        if isinstance(value, list):
            value = sorted(value)
        canonical[field] = value

    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
