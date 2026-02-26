"""Integration tests: project + deployment HTTP API flows.

These tests exercise the full HTTP stack — routing, request validation,
response serialisation, and exception-to-status-code mapping — by calling
the FastAPI app through an AsyncClient while patching the service functions.

Value over unit tests
---------------------
- Verifies URL routing is wired correctly (right handler for each endpoint).
- Verifies Pydantic validates request bodies before service functions are called.
- Verifies domain exceptions are converted to the correct HTTP status codes and
  the standard error envelope ``{"error": ..., "message": ..., "detail": ...}``.
- Verifies response models serialise service return values to JSON correctly.
- Tests are independent of the DB/Redis/Qdrant implementation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from retrieval_os.core.exceptions import (
    ConflictError,
    DeploymentNotFoundError,
    DeploymentStateError,
    ProjectNotFoundError,
)

from .conftest import make_deployment_response, make_index_config_response, make_project_response

# ── Fixtures ──────────────────────────────────────────────────────────────────

PROJECT_BODY = {
    "name": "my-docs",
    "description": "test",
    "created_by": "alice",
    "config": {
        "embedding_provider": "sentence_transformers",
        "embedding_model": "BAAI/bge-m3",
        "index_collection": "my_docs_v1",
    },
}

DEPLOY_BODY = {
    "index_config_version": 1,
    "created_by": "alice",
}

ROLLBACK_BODY = {
    "reason": "latency spike",
    "created_by": "oncall",
}


# ── Project endpoints ─────────────────────────────────────────────────────────


class TestProjectEndpoints:
    @pytest.mark.asyncio
    async def test_create_project_returns_201_with_id(self, int_client) -> None:
        client, _ = int_client
        project = make_project_response()
        with patch(
            "retrieval_os.plans.service.create_project", new=AsyncMock(return_value=project)
        ):
            resp = await client.post("/v1/projects", json=PROJECT_BODY)
        assert resp.status_code == 201
        body = resp.json()
        assert body["id"] == str(project.id)
        assert body["name"] == "my-docs"
        assert "current_index_config" in body

    @pytest.mark.asyncio
    async def test_create_project_invalid_name_returns_422(self, int_client) -> None:
        """Slug validation rejects uppercase names before the service is called."""
        client, _ = int_client
        bad_body = {**PROJECT_BODY, "name": "My-Docs"}  # uppercase not allowed
        resp = await client.post("/v1/projects", json=bad_body)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_project_missing_required_field_returns_422(self, int_client) -> None:
        """Request without 'config' must be rejected by Pydantic."""
        client, _ = int_client
        resp = await client.post("/v1/projects", json={"name": "my-docs", "created_by": "alice"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_project_conflict_returns_409_with_error_envelope(
        self, int_client
    ) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.plans.service.create_project",
            new=AsyncMock(side_effect=ConflictError("Project 'my-docs' already exists")),
        ):
            resp = await client.post("/v1/projects", json=PROJECT_BODY)
        assert resp.status_code == 409
        body = resp.json()
        assert body["error"] == "CONFLICT"
        assert "message" in body

    @pytest.mark.asyncio
    async def test_get_project_returns_200(self, int_client) -> None:
        client, _ = int_client
        project = make_project_response()
        with patch("retrieval_os.plans.service.get_project", new=AsyncMock(return_value=project)):
            resp = await client.get("/v1/projects/my-docs")
        assert resp.status_code == 200
        assert resp.json()["name"] == "my-docs"

    @pytest.mark.asyncio
    async def test_get_project_not_found_returns_404(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.plans.service.get_project",
            new=AsyncMock(side_effect=ProjectNotFoundError("not found")),
        ):
            resp = await client.get("/v1/projects/nonexistent")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"] == "PROJECT_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_create_index_config_returns_201(self, int_client) -> None:
        client, _ = int_client
        config = make_index_config_response(version=2)
        with patch(
            "retrieval_os.plans.service.create_index_config",
            new=AsyncMock(return_value=config),
        ):
            resp = await client.post(
                "/v1/projects/my-docs/index-configs",
                json={
                    "created_by": "alice",
                    "config": {
                        "embedding_provider": "sentence_transformers",
                        "embedding_model": "BAAI/bge-m3",
                        "index_collection": "my_docs_v2",
                    },
                },
            )
        assert resp.status_code == 201
        assert resp.json()["version"] == 2

    @pytest.mark.asyncio
    async def test_list_index_configs_returns_200(self, int_client) -> None:
        client, _ = int_client
        configs = [make_index_config_response(version=1), make_index_config_response(version=2)]
        with patch(
            "retrieval_os.plans.service.list_index_configs",
            new=AsyncMock(return_value=configs),
        ):
            resp = await client.get("/v1/projects/my-docs/index-configs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    @pytest.mark.asyncio
    async def test_archive_project_returns_204(self, int_client) -> None:
        client, _ = int_client
        with patch("retrieval_os.plans.service.archive_project", new=AsyncMock(return_value=None)):
            resp = await client.delete("/v1/projects/my-docs")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_archive_project_not_found_returns_404(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.plans.service.archive_project",
            new=AsyncMock(side_effect=ProjectNotFoundError("not found")),
        ):
            resp = await client.delete("/v1/projects/ghost-project")
        assert resp.status_code == 404


# ── Deployment endpoints ──────────────────────────────────────────────────────


class TestDeploymentEndpoints:
    @pytest.mark.asyncio
    async def test_create_deployment_returns_201_active(self, int_client) -> None:
        client, _ = int_client
        dep = make_deployment_response(status="ACTIVE")
        with patch(
            "retrieval_os.deployments.service.create_deployment",
            new=AsyncMock(return_value=dep),
        ):
            resp = await client.post("/v1/projects/my-docs/deployments", json=DEPLOY_BODY)
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "ACTIVE"
        assert body["traffic_weight"] == 1.0
        assert body["project_name"] == "my-docs"

    @pytest.mark.asyncio
    async def test_create_deployment_project_not_found_returns_404(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.deployments.service.create_deployment",
            new=AsyncMock(side_effect=ProjectNotFoundError("project missing")),
        ):
            resp = await client.post("/v1/projects/missing-project/deployments", json=DEPLOY_BODY)
        assert resp.status_code == 404
        assert resp.json()["error"] == "PROJECT_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_create_deployment_already_live_returns_409(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.deployments.service.create_deployment",
            new=AsyncMock(side_effect=ConflictError("already live")),
        ):
            resp = await client.post("/v1/projects/my-docs/deployments", json=DEPLOY_BODY)
        assert resp.status_code == 409
        assert resp.json()["error"] == "CONFLICT"

    @pytest.mark.asyncio
    async def test_create_deployment_missing_required_field_returns_422(self, int_client) -> None:
        client, _ = int_client
        # 'created_by' is required; omitting it should fail Pydantic validation
        resp = await client.post(
            "/v1/projects/my-docs/deployments", json={"index_config_version": 1}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_gradual_deployment_both_rollout_fields_required(self, int_client) -> None:
        """Providing only one of the paired rollout fields should return 422."""
        from retrieval_os.core.exceptions import AppValidationError

        client, _ = int_client
        with patch(
            "retrieval_os.deployments.service.create_deployment",
            new=AsyncMock(side_effect=AppValidationError("both rollout fields required")),
        ):
            resp = await client.post(
                "/v1/projects/my-docs/deployments",
                json={**DEPLOY_BODY, "rollout_step_percent": 10.0},
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_get_deployment_returns_200(self, int_client) -> None:
        client, _ = int_client
        dep = make_deployment_response(dep_id="dep-abc")
        with patch(
            "retrieval_os.deployments.service.get_deployment",
            new=AsyncMock(return_value=dep),
        ):
            resp = await client.get("/v1/projects/my-docs/deployments/dep-abc")
        assert resp.status_code == 200
        assert resp.json()["id"] == "dep-abc"

    @pytest.mark.asyncio
    async def test_get_deployment_not_found_returns_404(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.deployments.service.get_deployment",
            new=AsyncMock(side_effect=DeploymentNotFoundError("not found")),
        ):
            resp = await client.get("/v1/projects/my-docs/deployments/ghost")
        assert resp.status_code == 404
        assert resp.json()["error"] == "DEPLOYMENT_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_list_deployments_returns_200(self, int_client) -> None:
        client, _ = int_client
        deps = [make_deployment_response(dep_id="dep-a"), make_deployment_response(dep_id="dep-b")]
        with patch(
            "retrieval_os.deployments.service.list_deployments",
            new=AsyncMock(return_value=deps),
        ):
            resp = await client.get("/v1/projects/my-docs/deployments")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 2
        assert {d["id"] for d in body} == {"dep-a", "dep-b"}

    @pytest.mark.asyncio
    async def test_rollback_returns_200_rolled_back_status(self, int_client) -> None:
        client, _ = int_client
        dep = make_deployment_response(status="ROLLED_BACK", rollback_reason="latency spike")
        with patch(
            "retrieval_os.deployments.service.rollback_deployment",
            new=AsyncMock(return_value=dep),
        ):
            resp = await client.post(
                "/v1/projects/my-docs/deployments/dep-001/rollback",
                json=ROLLBACK_BODY,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ROLLED_BACK"
        assert body["rollback_reason"] == "latency spike"

    @pytest.mark.asyncio
    async def test_rollback_already_rolled_back_returns_409(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.deployments.service.rollback_deployment",
            new=AsyncMock(side_effect=DeploymentStateError("not live")),
        ):
            resp = await client.post(
                "/v1/projects/my-docs/deployments/dep-001/rollback",
                json=ROLLBACK_BODY,
            )
        assert resp.status_code == 409
        assert resp.json()["error"] == "DEPLOYMENT_STATE_ERROR"

    @pytest.mark.asyncio
    async def test_rollback_missing_reason_returns_422(self, int_client) -> None:
        """'reason' is required on the rollback request."""
        client, _ = int_client
        resp = await client.post(
            "/v1/projects/my-docs/deployments/dep-001/rollback",
            json={"created_by": "alice"},  # reason missing
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_deployment_all_search_config_fields_accepted_and_returned(
        self, int_client
    ) -> None:
        """All 9 search config fields round-trip through create → response."""
        client, _ = int_client
        search_config = {
            "top_k": 25,
            "reranker": "cross_encoder:cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_top_k": 5,
            "hybrid_alpha": 0.6,
            "metadata_filters": {"lang": "en"},
            "tenant_isolation_field": "org_id",
            "cache_enabled": False,
            "cache_ttl_seconds": 900,
            "max_tokens_per_query": 500,
        }
        dep = make_deployment_response(**search_config)
        with patch(
            "retrieval_os.deployments.service.create_deployment",
            new=AsyncMock(return_value=dep),
        ):
            resp = await client.post(
                "/v1/projects/my-docs/deployments",
                json={**DEPLOY_BODY, **search_config},
            )
        assert resp.status_code == 201
        body = resp.json()
        assert body["top_k"] == 25
        assert body["reranker"] == "cross_encoder:cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert body["rerank_top_k"] == 5
        assert body["hybrid_alpha"] == 0.6
        assert body["metadata_filters"] == {"lang": "en"}
        assert body["tenant_isolation_field"] == "org_id"
        assert body["cache_enabled"] is False
        assert body["cache_ttl_seconds"] == 900
        assert body["max_tokens_per_query"] == 500

    @pytest.mark.asyncio
    async def test_create_deployment_search_config_defaults_applied(self, int_client) -> None:
        """Omitting all search config fields yields schema defaults in the response."""
        client, _ = int_client
        dep = make_deployment_response()
        with patch(
            "retrieval_os.deployments.service.create_deployment",
            new=AsyncMock(return_value=dep),
        ):
            resp = await client.post("/v1/projects/my-docs/deployments", json=DEPLOY_BODY)
        assert resp.status_code == 201
        body = resp.json()
        assert body["top_k"] == 10
        assert body["cache_enabled"] is True
        assert body["cache_ttl_seconds"] == 3600
        assert body["reranker"] is None

    @pytest.mark.asyncio
    async def test_create_deployment_invalid_hybrid_alpha_returns_422(self, int_client) -> None:
        """hybrid_alpha must be in [0.0, 1.0]; 1.5 should be rejected by Pydantic."""
        client, _ = int_client
        resp = await client.post(
            "/v1/projects/my-docs/deployments",
            json={**DEPLOY_BODY, "hybrid_alpha": 1.5},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_deployment_invalid_top_k_returns_422(self, int_client) -> None:
        """top_k has a gt=0 constraint; sending 0 must be rejected by Pydantic."""
        client, _ = int_client
        resp = await client.post(
            "/v1/projects/my-docs/deployments",
            json={**DEPLOY_BODY, "top_k": 0},
        )
        assert resp.status_code == 422


# ── Multi-step flows ──────────────────────────────────────────────────────────


class TestMultiStepFlows:
    @pytest.mark.asyncio
    async def test_create_project_then_deploy_then_rollback(self, int_client) -> None:
        """Three-step flow: create project → instant deploy → rollback."""
        client, _ = int_client
        project = make_project_response()
        dep_active = make_deployment_response(dep_id="dep-flow-001", status="ACTIVE")
        dep_rolled = make_deployment_response(
            dep_id="dep-flow-001",
            status="ROLLED_BACK",
            rollback_reason="performance degradation",
        )

        with patch(
            "retrieval_os.plans.service.create_project", new=AsyncMock(return_value=project)
        ):
            r1 = await client.post("/v1/projects", json=PROJECT_BODY)
        assert r1.status_code == 201

        with patch(
            "retrieval_os.deployments.service.create_deployment",
            new=AsyncMock(return_value=dep_active),
        ):
            r2 = await client.post(f"/v1/projects/{project.name}/deployments", json=DEPLOY_BODY)
        assert r2.status_code == 201
        dep_id = r2.json()["id"]
        assert r2.json()["status"] == "ACTIVE"

        with patch(
            "retrieval_os.deployments.service.rollback_deployment",
            new=AsyncMock(return_value=dep_rolled),
        ):
            r3 = await client.post(
                f"/v1/projects/{project.name}/deployments/{dep_id}/rollback",
                json=ROLLBACK_BODY,
            )
        assert r3.status_code == 200
        assert r3.json()["status"] == "ROLLED_BACK"
        assert r3.json()["id"] == dep_id

    @pytest.mark.asyncio
    async def test_deploy_to_archived_project_returns_404(self, int_client) -> None:
        """Deploying to an archived project should surface a 404 from the service."""
        client, _ = int_client
        with patch(
            "retrieval_os.deployments.service.create_deployment",
            new=AsyncMock(side_effect=ProjectNotFoundError("Project is archived")),
        ):
            resp = await client.post("/v1/projects/archived-project/deployments", json=DEPLOY_BODY)
        assert resp.status_code == 404
