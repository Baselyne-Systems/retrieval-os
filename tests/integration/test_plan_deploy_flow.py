"""Integration tests: plan + deployment HTTP API flows.

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
    PlanNotFoundError,
)

from .conftest import make_deployment_response, make_plan_response, make_version_response

# ── Fixtures ──────────────────────────────────────────────────────────────────

PLAN_BODY = {
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
    "plan_version": 1,
    "created_by": "alice",
}

ROLLBACK_BODY = {
    "reason": "latency spike",
    "created_by": "oncall",
}


# ── Plan endpoints ────────────────────────────────────────────────────────────


class TestPlanEndpoints:
    @pytest.mark.asyncio
    async def test_create_plan_returns_201_with_id(self, int_client) -> None:
        client, _ = int_client
        plan = make_plan_response()
        with patch("retrieval_os.plans.service.create_plan", new=AsyncMock(return_value=plan)):
            resp = await client.post("/v1/plans", json=PLAN_BODY)
        assert resp.status_code == 201
        body = resp.json()
        assert body["id"] == str(plan.id)
        assert body["name"] == "my-docs"
        assert "current_version" in body

    @pytest.mark.asyncio
    async def test_create_plan_invalid_name_returns_422(self, int_client) -> None:
        """Slug validation rejects uppercase names before the service is called."""
        client, _ = int_client
        bad_body = {**PLAN_BODY, "name": "My-Docs"}  # uppercase not allowed
        resp = await client.post("/v1/plans", json=bad_body)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_plan_missing_required_field_returns_422(self, int_client) -> None:
        """Request without 'config' must be rejected by Pydantic."""
        client, _ = int_client
        resp = await client.post("/v1/plans", json={"name": "my-docs", "created_by": "alice"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_plan_conflict_returns_409_with_error_envelope(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.plans.service.create_plan",
            new=AsyncMock(side_effect=ConflictError("Plan 'my-docs' already exists")),
        ):
            resp = await client.post("/v1/plans", json=PLAN_BODY)
        assert resp.status_code == 409
        body = resp.json()
        assert body["error"] == "CONFLICT"
        assert "message" in body

    @pytest.mark.asyncio
    async def test_get_plan_returns_200(self, int_client) -> None:
        client, _ = int_client
        plan = make_plan_response()
        with patch("retrieval_os.plans.service.get_plan", new=AsyncMock(return_value=plan)):
            resp = await client.get("/v1/plans/my-docs")
        assert resp.status_code == 200
        assert resp.json()["name"] == "my-docs"

    @pytest.mark.asyncio
    async def test_get_plan_not_found_returns_404(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.plans.service.get_plan",
            new=AsyncMock(side_effect=PlanNotFoundError("not found")),
        ):
            resp = await client.get("/v1/plans/nonexistent")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"] == "PLAN_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_create_version_returns_201(self, int_client) -> None:
        client, _ = int_client
        version = make_version_response(version=2)
        with patch(
            "retrieval_os.plans.service.create_version", new=AsyncMock(return_value=version)
        ):
            resp = await client.post(
                "/v1/plans/my-docs/versions",
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
    async def test_list_versions_returns_200(self, int_client) -> None:
        client, _ = int_client
        versions = [make_version_response(version=1), make_version_response(version=2)]
        with patch(
            "retrieval_os.plans.service.list_versions", new=AsyncMock(return_value=versions)
        ):
            resp = await client.get("/v1/plans/my-docs/versions")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    @pytest.mark.asyncio
    async def test_archive_plan_returns_204(self, int_client) -> None:
        client, _ = int_client
        with patch("retrieval_os.plans.service.archive_plan", new=AsyncMock(return_value=None)):
            resp = await client.delete("/v1/plans/my-docs")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_archive_plan_not_found_returns_404(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.plans.service.archive_plan",
            new=AsyncMock(side_effect=PlanNotFoundError("not found")),
        ):
            resp = await client.delete("/v1/plans/ghost-plan")
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
            resp = await client.post("/v1/plans/my-docs/deployments", json=DEPLOY_BODY)
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "ACTIVE"
        assert body["traffic_weight"] == 1.0
        assert body["plan_name"] == "my-docs"

    @pytest.mark.asyncio
    async def test_create_deployment_plan_not_found_returns_404(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.deployments.service.create_deployment",
            new=AsyncMock(side_effect=PlanNotFoundError("plan missing")),
        ):
            resp = await client.post("/v1/plans/missing-plan/deployments", json=DEPLOY_BODY)
        assert resp.status_code == 404
        assert resp.json()["error"] == "PLAN_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_create_deployment_already_live_returns_409(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.deployments.service.create_deployment",
            new=AsyncMock(side_effect=ConflictError("already live")),
        ):
            resp = await client.post("/v1/plans/my-docs/deployments", json=DEPLOY_BODY)
        assert resp.status_code == 409
        assert resp.json()["error"] == "CONFLICT"

    @pytest.mark.asyncio
    async def test_create_deployment_missing_required_field_returns_422(self, int_client) -> None:
        client, _ = int_client
        # 'created_by' is required; omitting it should fail Pydantic validation
        resp = await client.post("/v1/plans/my-docs/deployments", json={"plan_version": 1})
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
                "/v1/plans/my-docs/deployments",
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
            resp = await client.get("/v1/plans/my-docs/deployments/dep-abc")
        assert resp.status_code == 200
        assert resp.json()["id"] == "dep-abc"

    @pytest.mark.asyncio
    async def test_get_deployment_not_found_returns_404(self, int_client) -> None:
        client, _ = int_client
        with patch(
            "retrieval_os.deployments.service.get_deployment",
            new=AsyncMock(side_effect=DeploymentNotFoundError("not found")),
        ):
            resp = await client.get("/v1/plans/my-docs/deployments/ghost")
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
            resp = await client.get("/v1/plans/my-docs/deployments")
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
                "/v1/plans/my-docs/deployments/dep-001/rollback",
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
                "/v1/plans/my-docs/deployments/dep-001/rollback",
                json=ROLLBACK_BODY,
            )
        assert resp.status_code == 409
        assert resp.json()["error"] == "DEPLOYMENT_STATE_ERROR"

    @pytest.mark.asyncio
    async def test_rollback_missing_reason_returns_422(self, int_client) -> None:
        """'reason' is required on the rollback request."""
        client, _ = int_client
        resp = await client.post(
            "/v1/plans/my-docs/deployments/dep-001/rollback",
            json={"created_by": "alice"},  # reason missing
        )
        assert resp.status_code == 422


# ── Multi-step flows ──────────────────────────────────────────────────────────


class TestMultiStepFlows:
    @pytest.mark.asyncio
    async def test_create_plan_then_deploy_then_rollback(self, int_client) -> None:
        """Three-step flow: create plan → instant deploy → rollback.

        Verifies the response shapes at each step and that the deployment ID
        returned from the deploy step can be used in the rollback URL.
        """
        client, _ = int_client
        plan = make_plan_response()
        dep_active = make_deployment_response(dep_id="dep-flow-001", status="ACTIVE")
        dep_rolled = make_deployment_response(
            dep_id="dep-flow-001",
            status="ROLLED_BACK",
            rollback_reason="performance degradation",
        )

        with patch("retrieval_os.plans.service.create_plan", new=AsyncMock(return_value=plan)):
            r1 = await client.post("/v1/plans", json=PLAN_BODY)
        assert r1.status_code == 201

        with patch(
            "retrieval_os.deployments.service.create_deployment",
            new=AsyncMock(return_value=dep_active),
        ):
            r2 = await client.post(f"/v1/plans/{plan.name}/deployments", json=DEPLOY_BODY)
        assert r2.status_code == 201
        dep_id = r2.json()["id"]
        assert r2.json()["status"] == "ACTIVE"

        with patch(
            "retrieval_os.deployments.service.rollback_deployment",
            new=AsyncMock(return_value=dep_rolled),
        ):
            r3 = await client.post(
                f"/v1/plans/{plan.name}/deployments/{dep_id}/rollback",
                json=ROLLBACK_BODY,
            )
        assert r3.status_code == 200
        assert r3.json()["status"] == "ROLLED_BACK"
        assert r3.json()["id"] == dep_id

    @pytest.mark.asyncio
    async def test_deploy_to_archived_plan_returns_404(self, int_client) -> None:
        """Deploying to an archived plan should surface a 404 from the service."""
        client, _ = int_client
        with patch(
            "retrieval_os.deployments.service.create_deployment",
            new=AsyncMock(side_effect=PlanNotFoundError("Plan is archived")),
        ):
            resp = await client.post("/v1/plans/archived-plan/deployments", json=DEPLOY_BODY)
        assert resp.status_code == 404
