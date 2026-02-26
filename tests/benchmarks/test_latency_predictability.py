"""Benchmark: Latency Predictability — hot path is Redis-only, no Postgres.

Customer claims
---------------
1. The Redis key format is stable and documented: ros:project:{name}:active.
2. On a cache hit (serving config in Redis), the query is served without touching
   Postgres — no connection pool pressure, no join latency.
3. On a cache miss, _load_project_config falls back to Postgres exactly once,
   then warms the Redis cache so subsequent requests are fast.
4. The merged serving config carries all 16 fields needed to serve a query so the
   hot path never needs to go back to the DB mid-request.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval_os.serving.query_router import _load_project_config, _project_redis_key

# ── Key format ────────────────────────────────────────────────────────────────


class TestRedisKeyFormat:
    def test_key_is_stable_and_documented(self) -> None:
        assert _project_redis_key("my-docs") == "ros:project:my-docs:active"

    def test_key_embeds_project_name(self) -> None:
        assert "wiki-search" in _project_redis_key("wiki-search")

    def test_key_prefix_is_namespace_safe(self) -> None:
        key = _project_redis_key("proj")
        assert key.startswith("ros:project:"), "Key must be in the ros: namespace"
        assert key.endswith(":active")


# ── Redis hit → no Postgres ───────────────────────────────────────────────────


_VALID_CONFIG = {
    "project_name": "my-docs",
    "index_config_version": 2,
    "embedding_provider": "sentence_transformers",
    "embedding_model": "BAAI/bge-m3",
    "embedding_normalize": True,
    "embedding_batch_size": 32,
    "index_backend": "qdrant",
    "index_collection": "docs_v2",
    "distance_metric": "cosine",
    "top_k": 10,
    "reranker": None,
    "rerank_top_k": None,
    "metadata_filters": None,
    "cache_enabled": True,
    "cache_ttl_seconds": 3600,
    "hybrid_alpha": None,
}


class TestRedisHitSkipsPostgres:
    @pytest.mark.asyncio
    async def test_redis_hit_never_reads_postgres(self) -> None:
        """When serving config is in Redis, Postgres is not touched."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=json.dumps(_VALID_CONFIG).encode())

        project_repo_mock = MagicMock()
        project_repo_mock.get_by_name = AsyncMock()
        db_session = MagicMock()

        with (
            patch(
                "retrieval_os.serving.query_router.get_redis",
                new=AsyncMock(return_value=redis_mock),
            ),
            patch("retrieval_os.serving.query_router.project_repo", project_repo_mock),
        ):
            config = await _load_project_config("my-docs", db_session)

        assert config == _VALID_CONFIG
        project_repo_mock.get_by_name.assert_not_called()
        db_session.execute.assert_not_called() if hasattr(db_session, "execute") else None

    @pytest.mark.asyncio
    async def test_redis_hit_returns_all_16_config_fields(self) -> None:
        """The config dict has all 16 fields needed to execute a retrieval query."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=json.dumps(_VALID_CONFIG).encode())

        expected_fields = {
            "project_name",
            "index_config_version",
            "embedding_provider",
            "embedding_model",
            "embedding_normalize",
            "embedding_batch_size",
            "index_backend",
            "index_collection",
            "distance_metric",
            "top_k",
            "reranker",
            "rerank_top_k",
            "metadata_filters",
            "cache_enabled",
            "cache_ttl_seconds",
            "hybrid_alpha",
        }

        with (
            patch(
                "retrieval_os.serving.query_router.get_redis",
                new=AsyncMock(return_value=redis_mock),
            ),
            patch("retrieval_os.serving.query_router.project_repo", MagicMock()),
        ):
            config = await _load_project_config("my-docs", MagicMock())

        missing = expected_fields - set(config.keys())
        assert not missing, f"Config is missing fields required for serving: {missing}"


# ── Redis miss → Postgres fallback + cache warm ───────────────────────────────


class TestRedisMissFallback:
    @pytest.mark.asyncio
    async def test_redis_miss_warms_cache_for_next_request(self) -> None:
        """On a cache miss, Postgres is read once and the result is stored in Redis."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)  # miss
        redis_mock.set = AsyncMock()

        index_config_id = uuid.uuid4()

        mock_project = SimpleNamespace(
            name="my-docs",
            is_archived=False,
        )
        mock_deployment = SimpleNamespace(
            index_config_id=index_config_id,
            index_config_version=1,
            top_k=10,
            reranker=None,
            rerank_top_k=None,
            metadata_filters=None,
            cache_enabled=True,
            cache_ttl_seconds=3600,
            hybrid_alpha=None,
        )
        mock_index_config = SimpleNamespace(
            embedding_provider="sentence_transformers",
            embedding_model="BAAI/bge-m3",
            embedding_normalize=True,
            embedding_batch_size=32,
            index_backend="qdrant",
            index_collection="docs_v1",
            distance_metric="cosine",
        )

        project_repo_mock = MagicMock()
        project_repo_mock.get_by_name = AsyncMock(return_value=mock_project)
        project_repo_mock.get_index_config_by_id = AsyncMock(return_value=mock_index_config)

        deployment_repo_mock = MagicMock()
        deployment_repo_mock.get_active_for_project = AsyncMock(return_value=mock_deployment)

        with (
            patch(
                "retrieval_os.serving.query_router.get_redis",
                new=AsyncMock(return_value=redis_mock),
            ),
            patch("retrieval_os.serving.query_router.project_repo", project_repo_mock),
            patch("retrieval_os.serving.query_router.deployment_repo", deployment_repo_mock),
        ):
            config = await _load_project_config("my-docs", MagicMock())

        # Postgres was queried exactly once
        project_repo_mock.get_by_name.assert_called_once()
        # Redis was warmed so the next request will be a hit
        redis_mock.set.assert_called_once()
        set_key = redis_mock.set.call_args[0][0]
        assert set_key == "ros:project:my-docs:active"

        # Returned config is complete
        assert config["project_name"] == "my-docs"
        assert config["embedding_model"] == "BAAI/bge-m3"
        assert config["top_k"] == 10

    @pytest.mark.asyncio
    async def test_redis_miss_merged_config_carries_deployment_search_fields(self) -> None:
        """Deployment search fields (top_k, reranker, cache settings) are merged
        into the serving config returned on a Postgres fallback."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.set = AsyncMock()

        index_config_id = uuid.uuid4()

        mock_project = SimpleNamespace(name="my-docs", is_archived=False)
        mock_deployment = SimpleNamespace(
            index_config_id=index_config_id,
            index_config_version=3,
            top_k=25,
            reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_top_k=5,
            metadata_filters={"lang": "en"},
            cache_enabled=False,
            cache_ttl_seconds=0,
            hybrid_alpha=0.6,
        )
        mock_index_config = SimpleNamespace(
            embedding_provider="openai",
            embedding_model="text-embedding-3-large",
            embedding_normalize=False,
            embedding_batch_size=64,
            index_backend="qdrant",
            index_collection="my_v3",
            distance_metric="dot",
        )

        project_repo_mock = MagicMock()
        project_repo_mock.get_by_name = AsyncMock(return_value=mock_project)
        project_repo_mock.get_index_config_by_id = AsyncMock(return_value=mock_index_config)

        deployment_repo_mock = MagicMock()
        deployment_repo_mock.get_active_for_project = AsyncMock(return_value=mock_deployment)

        with (
            patch(
                "retrieval_os.serving.query_router.get_redis",
                new=AsyncMock(return_value=redis_mock),
            ),
            patch("retrieval_os.serving.query_router.project_repo", project_repo_mock),
            patch("retrieval_os.serving.query_router.deployment_repo", deployment_repo_mock),
        ):
            config = await _load_project_config("my-docs", MagicMock())

        # Deployment search fields
        assert config["top_k"] == 25
        assert config["reranker"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config["rerank_top_k"] == 5
        assert config["metadata_filters"] == {"lang": "en"}
        assert config["cache_enabled"] is False
        assert config["cache_ttl_seconds"] == 0
        assert config["hybrid_alpha"] == pytest.approx(0.6)

        # IndexConfig embed fields
        assert config["embedding_provider"] == "openai"
        assert config["embedding_model"] == "text-embedding-3-large"
        assert config["index_collection"] == "my_v3"
        assert config["distance_metric"] == "dot"

    @pytest.mark.asyncio
    async def test_redis_miss_project_not_found_raises(self) -> None:
        """If the project does not exist, ProjectNotFoundError is raised — no hang."""
        from retrieval_os.core.exceptions import ProjectNotFoundError

        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)

        project_repo_mock = MagicMock()
        project_repo_mock.get_by_name = AsyncMock(return_value=None)

        with (
            patch(
                "retrieval_os.serving.query_router.get_redis",
                new=AsyncMock(return_value=redis_mock),
            ),
            patch("retrieval_os.serving.query_router.project_repo", project_repo_mock),
        ):
            with pytest.raises(ProjectNotFoundError):
                await _load_project_config("ghost-project", MagicMock())
