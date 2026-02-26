"""Integration tests: serving / query path HTTP and config-loading flows.

Two test classes:

- TestQueryEndpoint      — exercises the full HTTP stack via AsyncClient,
                           patching ``route_query`` at the serving_router import site.
- TestQueryRouterConfig  — calls ``_load_project_config()`` directly, patching
                           Redis and the repo singletons in query_router.

Value over unit tests
---------------------
- Verifies the ``/v1/query/{plan_name}`` route is wired and serialises correctly.
- Verifies domain exceptions (ProjectNotFoundError) map to the right HTTP codes.
- Verifies the Redis key format and the 16-field merged config blob.
- Verifies that a Redis hit never touches Postgres.
- Verifies that a Redis miss warms the cache with the correct key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval_os.core.exceptions import ProjectNotFoundError
from retrieval_os.serving.query_router import _load_project_config, _project_redis_key

# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_chunk() -> SimpleNamespace:
    return SimpleNamespace(
        id="chunk-1",
        score=0.95,
        text="RAG is a retrieval-augmented generation technique.",
        metadata={"source": "doc.pdf"},
    )


def _make_query_info(*, project_name: str = "my-docs", cache_hit: bool = False) -> dict:
    return {
        "project_name": project_name,
        "version": 1,
        "cache_hit": cache_hit,
        "result_count": 1,
    }


# ── TestQueryEndpoint ─────────────────────────────────────────────────────────


class TestQueryEndpoint:
    @pytest.mark.asyncio
    async def test_query_returns_200_with_results(self, int_client) -> None:
        client, _ = int_client
        chunk = _make_chunk()
        info = _make_query_info()
        with (
            patch(
                "retrieval_os.api.serving_router.route_query",
                new=AsyncMock(return_value=([chunk], info)),
            ),
            patch("retrieval_os.api.serving_router.fire_usage_record"),
        ):
            resp = await client.post("/v1/query/my-docs", json={"query": "what is RAG?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["project_name"] == "my-docs"
        assert body["version"] == 1
        assert body["cache_hit"] is False
        assert body["result_count"] == 1
        assert len(body["results"]) == 1
        r = body["results"][0]
        assert r["id"] == "chunk-1"
        assert r["score"] == 0.95
        assert r["text"] == "RAG is a retrieval-augmented generation technique."
        assert r["metadata"] == {"source": "doc.pdf"}

    @pytest.mark.asyncio
    async def test_query_cache_hit_true_reflected_in_response(self, int_client) -> None:
        client, _ = int_client
        chunk = _make_chunk()
        info = _make_query_info(cache_hit=True)
        with (
            patch(
                "retrieval_os.api.serving_router.route_query",
                new=AsyncMock(return_value=([chunk], info)),
            ),
            patch("retrieval_os.api.serving_router.fire_usage_record"),
        ):
            resp = await client.post("/v1/query/my-docs", json={"query": "what is RAG?"})
        assert resp.status_code == 200
        assert resp.json()["cache_hit"] is True

    @pytest.mark.asyncio
    async def test_query_project_not_found_returns_404(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.api.serving_router.route_query",
            new=AsyncMock(side_effect=ProjectNotFoundError("project missing")),
        ):
            resp = await client.post("/v1/query/missing-project", json={"query": "test"})
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"] == "PROJECT_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_query_missing_query_field_returns_422(self, int_client) -> None:
        """Pydantic rejects an empty body before the handler is reached."""
        client, _ = int_client
        resp = await client.post("/v1/query/my-docs", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_query_metadata_filter_override_forwarded(self, int_client) -> None:
        """metadata_filters from the request body are passed to route_query."""
        client, _ = int_client
        chunk = _make_chunk()
        info = _make_query_info()
        mock_route = AsyncMock(return_value=([chunk], info))
        with (
            patch("retrieval_os.api.serving_router.route_query", mock_route),
            patch("retrieval_os.api.serving_router.fire_usage_record"),
        ):
            await client.post(
                "/v1/query/my-docs",
                json={"query": "foo", "metadata_filters": {"lang": "en"}},
            )
        call_kwargs = mock_route.call_args.kwargs
        assert call_kwargs["metadata_filter_override"] == {"lang": "en"}


# ── TestQueryRouterConfig ─────────────────────────────────────────────────────


class TestQueryRouterConfig:
    def test_redis_key_format(self) -> None:
        assert _project_redis_key("my-docs") == "ros:project:my-docs:active"

    @pytest.mark.asyncio
    async def test_load_project_config_redis_hit_skips_postgres(self) -> None:
        """When Redis returns a cached blob, no Postgres calls are made."""
        config_blob = {
            "project_name": "my-docs",
            "index_config_version": 1,
            "embedding_provider": "sentence_transformers",
            "embedding_model": "BAAI/bge-m3",
            "embedding_normalize": True,
            "embedding_batch_size": 32,
            "index_backend": "qdrant",
            "index_collection": "my_docs_v1",
            "distance_metric": "cosine",
            "top_k": 10,
            "reranker": None,
            "rerank_top_k": None,
            "metadata_filters": None,
            "cache_enabled": True,
            "cache_ttl_seconds": 3600,
            "hybrid_alpha": None,
        }
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(config_blob))
        mock_project_repo = MagicMock()

        with (
            patch(
                "retrieval_os.serving.query_router.get_redis",
                new=AsyncMock(return_value=mock_redis),
            ),
            patch("retrieval_os.serving.query_router.project_repo", mock_project_repo),
        ):
            mock_db = AsyncMock()
            result = await _load_project_config("my-docs", mock_db)

        assert result == config_blob
        mock_redis.get.assert_called_once_with("ros:project:my-docs:active")
        mock_project_repo.get_by_name.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_project_config_redis_miss_falls_back_and_warms_cache(self) -> None:
        """On a Redis miss, config is loaded from Postgres and the cache is warmed."""
        fake_project = SimpleNamespace(name="my-docs", is_archived=False)
        fake_config_id = "018e7a2b-4000-7000-b234-000000000001"
        fake_deployment = SimpleNamespace(
            index_config_id=fake_config_id,
            index_config_version=1,
            top_k=10,
            reranker=None,
            rerank_top_k=None,
            metadata_filters=None,
            cache_enabled=True,
            cache_ttl_seconds=3600,
            hybrid_alpha=None,
        )
        fake_index_config = SimpleNamespace(
            embedding_provider="sentence_transformers",
            embedding_model="BAAI/bge-m3",
            embedding_normalize=True,
            embedding_batch_size=32,
            index_backend="qdrant",
            index_collection="my_docs_v1",
            distance_metric="cosine",
        )

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()

        mock_project_repo = MagicMock()
        mock_project_repo.get_by_name = AsyncMock(return_value=fake_project)
        mock_project_repo.get_index_config_by_id = AsyncMock(return_value=fake_index_config)

        mock_deployment_repo = MagicMock()
        mock_deployment_repo.get_active_for_project = AsyncMock(return_value=fake_deployment)

        with (
            patch(
                "retrieval_os.serving.query_router.get_redis",
                new=AsyncMock(return_value=mock_redis),
            ),
            patch("retrieval_os.serving.query_router.project_repo", mock_project_repo),
            patch("retrieval_os.serving.query_router.deployment_repo", mock_deployment_repo),
        ):
            mock_db = AsyncMock()
            result = await _load_project_config("my-docs", mock_db)

        expected_keys = {
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
        assert set(result.keys()) == expected_keys
        mock_redis.set.assert_called_once()
        assert mock_redis.set.call_args.args[0] == "ros:project:my-docs:active"

    @pytest.mark.asyncio
    async def test_load_project_config_merged_config_carries_deployment_search_fields(
        self,
    ) -> None:
        """Deployment search config fields AND IndexConfig embed fields both appear in the merged blob."""
        fake_project = SimpleNamespace(name="my-docs", is_archived=False)
        fake_config_id = "018e7a2b-4000-7000-b234-000000000001"
        fake_deployment = SimpleNamespace(
            index_config_id=fake_config_id,
            index_config_version=2,
            top_k=25,
            reranker="cross_encoder:cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_top_k=5,
            metadata_filters={"lang": "en"},
            cache_enabled=False,
            cache_ttl_seconds=900,
            hybrid_alpha=0.6,
        )
        fake_index_config = SimpleNamespace(
            embedding_provider="sentence_transformers",
            embedding_model="BAAI/bge-m3",
            embedding_normalize=True,
            embedding_batch_size=32,
            index_backend="qdrant",
            index_collection="my_v2",
            distance_metric="cosine",
        )

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()

        mock_project_repo = MagicMock()
        mock_project_repo.get_by_name = AsyncMock(return_value=fake_project)
        mock_project_repo.get_index_config_by_id = AsyncMock(return_value=fake_index_config)

        mock_deployment_repo = MagicMock()
        mock_deployment_repo.get_active_for_project = AsyncMock(return_value=fake_deployment)

        with (
            patch(
                "retrieval_os.serving.query_router.get_redis",
                new=AsyncMock(return_value=mock_redis),
            ),
            patch("retrieval_os.serving.query_router.project_repo", mock_project_repo),
            patch("retrieval_os.serving.query_router.deployment_repo", mock_deployment_repo),
        ):
            mock_db = AsyncMock()
            result = await _load_project_config("my-docs", mock_db)

        # Deployment search fields
        assert result["top_k"] == 25
        assert result["reranker"] == "cross_encoder:cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert result["cache_ttl_seconds"] == 900
        assert result["cache_enabled"] is False
        # IndexConfig embed fields
        assert result["embedding_model"] == "BAAI/bge-m3"
        assert result["index_collection"] == "my_v2"

    @pytest.mark.asyncio
    async def test_load_project_config_project_not_found_raises(self) -> None:
        """ProjectNotFoundError is raised when the project row is missing."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        mock_project_repo = MagicMock()
        mock_project_repo.get_by_name = AsyncMock(return_value=None)

        with (
            patch(
                "retrieval_os.serving.query_router.get_redis",
                new=AsyncMock(return_value=mock_redis),
            ),
            patch("retrieval_os.serving.query_router.project_repo", mock_project_repo),
        ):
            mock_db = AsyncMock()
            with pytest.raises(ProjectNotFoundError):
                await _load_project_config("ghost-project", mock_db)
