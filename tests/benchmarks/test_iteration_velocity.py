"""Benchmark: Iteration Velocity — search-config tuning never causes re-indexing.

Customer claim
--------------
Teams can change top_k, reranker, cache settings, and hybrid_alpha as many times
as they want without triggering a re-embed / re-index cycle.  Only changes to the
embedding model, index collection, or distance metric should require rebuilding the
index.

These tests prove the invariant at two levels:
1. Pure-function: the config hash excludes all nine search-config fields.
2. Service-level: submitting an IndexConfig that is hash-identical to an existing
   one raises DuplicateConfigError *before* any embedding work is attempted.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval_os.core.exceptions import DuplicateConfigError
from retrieval_os.plans.schemas import CreateIndexConfigRequest, IndexConfigInput
from retrieval_os.plans.service import create_index_config
from retrieval_os.plans.validators import compute_config_hash

# ── Base config helpers ───────────────────────────────────────────────────────


def _base_index_config(**overrides: object) -> dict:
    cfg = {
        "embedding_provider": "sentence_transformers",
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimensions": 768,
        "modalities": ["text"],
        "embedding_batch_size": 32,
        "embedding_normalize": True,
        "index_backend": "qdrant",
        "index_collection": "docs_v1",
        "distance_metric": "cosine",
        "quantization": None,
        "change_comment": "",
        # search-config fields (should be ignored by hash)
        "top_k": 10,
        "reranker": None,
        "rerank_top_k": None,
        "hybrid_alpha": None,
        "cache_enabled": True,
        "cache_ttl_seconds": 3600,
        "max_tokens_per_query": None,
        "metadata_filters": None,
        "tenant_isolation_field": None,
    }
    cfg.update(overrides)
    return cfg


# ── Hash stability for search-config changes ──────────────────────────────────


class TestSearchTuningIsHashNeutral:
    """Changing any search-config field must not alter the config hash."""

    def _hash(self, **overrides: object) -> str:
        return compute_config_hash(_base_index_config(**overrides))

    def _base_hash(self) -> str:
        return compute_config_hash(_base_index_config())

    def test_top_k_change_does_not_affect_hash(self) -> None:
        assert self._hash(top_k=50) == self._base_hash()

    def test_reranker_change_does_not_affect_hash(self) -> None:
        assert self._hash(reranker="cross-encoder/ms-marco-MiniLM-L-6-v2") == self._base_hash()

    def test_rerank_top_k_change_does_not_affect_hash(self) -> None:
        assert self._hash(rerank_top_k=5) == self._base_hash()

    def test_hybrid_alpha_change_does_not_affect_hash(self) -> None:
        assert self._hash(hybrid_alpha=0.7) == self._base_hash()

    def test_cache_enabled_change_does_not_affect_hash(self) -> None:
        assert self._hash(cache_enabled=False) == self._base_hash()

    def test_cache_ttl_change_does_not_affect_hash(self) -> None:
        assert self._hash(cache_ttl_seconds=900) == self._base_hash()

    def test_max_tokens_change_does_not_affect_hash(self) -> None:
        assert self._hash(max_tokens_per_query=512) == self._base_hash()

    def test_metadata_filters_change_does_not_affect_hash(self) -> None:
        assert self._hash(metadata_filters={"lang": "en"}) == self._base_hash()

    def test_tenant_isolation_field_change_does_not_affect_hash(self) -> None:
        assert self._hash(tenant_isolation_field="org_id") == self._base_hash()

    def test_all_search_config_fields_combined_do_not_affect_hash(self) -> None:
        """Changing every search-config field at once still leaves the hash unchanged."""
        assert (
            self._hash(
                top_k=100,
                reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
                rerank_top_k=10,
                hybrid_alpha=0.5,
                cache_enabled=False,
                cache_ttl_seconds=0,
                max_tokens_per_query=256,
                metadata_filters={"team": "eng"},
                tenant_isolation_field="tenant_id",
            )
            == self._base_hash()
        )


class TestIndexRebuildFieldsDoChangeHash:
    """Changing an index-build field MUST produce a different hash — sanity check."""

    def _hash(self, **overrides: object) -> str:
        return compute_config_hash(_base_index_config(**overrides))

    def _base_hash(self) -> str:
        return compute_config_hash(_base_index_config())

    def test_embedding_model_change_changes_hash(self) -> None:
        assert self._hash(embedding_model="text-embedding-3-large") != self._base_hash()

    def test_index_collection_change_changes_hash(self) -> None:
        assert self._hash(index_collection="docs_v2") != self._base_hash()

    def test_distance_metric_change_changes_hash(self) -> None:
        assert self._hash(distance_metric="dot") != self._base_hash()

    def test_embedding_dimensions_change_changes_hash(self) -> None:
        assert self._hash(embedding_dimensions=1536) != self._base_hash()


# ── Service-level dedup ───────────────────────────────────────────────────────


class TestConfigDedup:
    """create_index_config raises DuplicateConfigError when a hash-identical config
    already exists — no embedding work is done."""

    @pytest.mark.asyncio
    async def test_duplicate_config_rejected_before_any_embedding(self) -> None:
        """Submitting the same index config a second time must raise immediately."""
        request = CreateIndexConfigRequest(
            config=IndexConfigInput(
                embedding_provider="sentence_transformers",
                embedding_model="BAAI/bge-m3",
                index_collection="docs_v1",
            ),
            created_by="alice",
        )

        mock_project = MagicMock()
        mock_project.is_archived = False
        mock_project.id = uuid.uuid4()

        mock_existing = MagicMock()
        mock_existing.version = 1

        embed_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.plans.service.project_repo.get_by_name",
                new=AsyncMock(return_value=mock_project),
            ),
            patch(
                "retrieval_os.plans.service.project_repo.get_index_config_by_config_hash",
                new=AsyncMock(return_value=mock_existing),
            ),
            patch(
                "retrieval_os.plans.service.project_repo.get_next_version_number", new=embed_mock
            ),
        ):
            with pytest.raises(DuplicateConfigError):
                await create_index_config(MagicMock(), "my-docs", request)

        # Embedding/indexing work (represented by get_next_version_number) was never reached
        embed_mock.assert_not_awaited()
