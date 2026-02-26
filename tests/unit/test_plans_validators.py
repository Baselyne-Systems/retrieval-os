"""Unit tests for index config validation and config hash computation."""

import pytest

from retrieval_os.core.exceptions import AppValidationError
from retrieval_os.plans.validators import (
    VALID_BACKENDS,
    VALID_METRICS,
    VALID_PROVIDERS,
    compute_config_hash,
    validate_plan_config,
)


def _base_config(**overrides: object) -> dict:
    cfg = {
        "embedding_provider": "sentence_transformers",
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimensions": 768,
        "modalities": ["text"],
        "embedding_batch_size": 32,
        "embedding_normalize": True,
        "index_backend": "qdrant",
        "index_collection": "docs",
        "distance_metric": "cosine",
        "quantization": None,
        "change_comment": "",
    }
    cfg.update(overrides)
    return cfg


# ── validate_plan_config ───────────────────────────────────────────────────────


class TestValidatePlanConfig:
    def test_valid_config_passes(self) -> None:
        validate_plan_config(_base_config())  # should not raise

    # Provider
    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(AppValidationError) as exc_info:
            validate_plan_config(_base_config(embedding_provider="gpt99"))
        assert "embedding_provider" in exc_info.value.detail["errors"][0]

    @pytest.mark.parametrize("provider", sorted(VALID_PROVIDERS))
    def test_all_valid_providers_pass(self, provider: str) -> None:
        modality_map = {
            "sentence_transformers": ["text"],
            "openai": ["text"],
            "clip": ["image"],
            "whisper": ["audio"],
            "video_frame": ["video"],
        }
        validate_plan_config(
            _base_config(
                embedding_provider=provider,
                modalities=modality_map[provider],
            )
        )

    # Backend
    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(AppValidationError) as exc_info:
            validate_plan_config(_base_config(index_backend="pinecone"))
        assert "index_backend" in exc_info.value.detail["errors"][0]

    @pytest.mark.parametrize("backend", sorted(VALID_BACKENDS))
    def test_all_valid_backends_pass(self, backend: str) -> None:
        validate_plan_config(_base_config(index_backend=backend))

    # Distance metric
    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(AppValidationError) as exc_info:
            validate_plan_config(_base_config(distance_metric="manhattan"))
        assert "distance_metric" in exc_info.value.detail["errors"][0]

    @pytest.mark.parametrize("metric", sorted(VALID_METRICS))
    def test_all_valid_metrics_pass(self, metric: str) -> None:
        validate_plan_config(_base_config(distance_metric=metric))

    # Modalities
    def test_empty_modalities_raises(self) -> None:
        with pytest.raises(AppValidationError) as exc_info:
            validate_plan_config(_base_config(modalities=[]))
        assert "modalities" in exc_info.value.detail["errors"][0]

    def test_unknown_modality_raises(self) -> None:
        with pytest.raises(AppValidationError) as exc_info:
            validate_plan_config(_base_config(modalities=["text", "smell"]))
        assert "modalities" in exc_info.value.detail["errors"][0]

    def test_provider_modality_mismatch_raises(self) -> None:
        # openai only supports text; asking for image should fail
        with pytest.raises(AppValidationError) as exc_info:
            validate_plan_config(_base_config(embedding_provider="openai", modalities=["image"]))
        errors = exc_info.value.detail["errors"]
        assert any("openai" in e for e in errors)

    def test_clip_supports_text_and_image(self) -> None:
        validate_plan_config(_base_config(embedding_provider="clip", modalities=["text", "image"]))

    # quantization
    def test_invalid_quantization_raises(self) -> None:
        with pytest.raises(AppValidationError) as exc_info:
            validate_plan_config(_base_config(quantization="binary"))
        assert "quantization" in exc_info.value.detail["errors"][0]

    def test_valid_quantizations_pass(self) -> None:
        validate_plan_config(_base_config(quantization="scalar"))
        validate_plan_config(_base_config(quantization="product"))

    # Multiple errors collected
    def test_multiple_errors_collected(self) -> None:
        with pytest.raises(AppValidationError) as exc_info:
            validate_plan_config(
                _base_config(
                    embedding_provider="bad",
                    index_backend="bad",
                )
            )
        assert len(exc_info.value.detail["errors"]) >= 2


# ── compute_config_hash ────────────────────────────────────────────────────────


class TestComputeConfigHash:
    def test_same_config_same_hash(self) -> None:
        cfg = _base_config()
        assert compute_config_hash(cfg) == compute_config_hash(cfg)

    def test_different_config_different_hash(self) -> None:
        h1 = compute_config_hash(_base_config(index_collection="docs_v1"))
        h2 = compute_config_hash(_base_config(index_collection="docs_v2"))
        assert h1 != h2

    def test_hash_is_64_hex_chars(self) -> None:
        h = compute_config_hash(_base_config())
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_modality_order_does_not_affect_hash(self) -> None:
        h1 = compute_config_hash(
            _base_config(
                embedding_provider="clip",
                modalities=["text", "image"],
            )
        )
        h2 = compute_config_hash(
            _base_config(
                embedding_provider="clip",
                modalities=["image", "text"],
            )
        )
        assert h1 == h2

    def test_change_comment_excluded_from_hash(self) -> None:
        h1 = compute_config_hash(_base_config(change_comment="first"))
        h2 = compute_config_hash(_base_config(change_comment="second"))
        assert h1 == h2

    def test_model_change_changes_hash(self) -> None:
        h1 = compute_config_hash(_base_config(embedding_model="BAAI/bge-m3"))
        h2 = compute_config_hash(_base_config(embedding_model="text-embedding-3-large"))
        assert h1 != h2

    def test_collection_change_changes_hash(self) -> None:
        h1 = compute_config_hash(_base_config(index_collection="col_v1"))
        h2 = compute_config_hash(_base_config(index_collection="col_v2"))
        assert h1 != h2
