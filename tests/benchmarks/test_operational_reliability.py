"""Benchmark: Operational Reliability — zero-downtime deployments.

Customer claims
---------------
1. Only one deployment is live at a time — a second concurrent deployment is
   rejected immediately with a clear error, not silently queued.
2. Rollback is atomic: Redis is cleared in the same operation that marks the
   deployment as ROLLED_BACK, so queries can never be served from a rolled-back
   deployment.
3. Gradual rollouts advance from 0% to 100% in discrete, predictable steps and
   promote to ACTIVE automatically at full weight.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval_os.core.exceptions import ConflictError
from retrieval_os.deployments.models import Deployment, DeploymentStatus
from retrieval_os.deployments.schemas import RollbackRequest
from retrieval_os.deployments.service import (
    rollback_deployment,
    step_rolling_deployments,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _dep(
    *,
    dep_id: str = "dep-001",
    project_name: str = "my-docs",
    status: str = DeploymentStatus.ACTIVE.value,
    traffic_weight: float = 1.0,
    rollout_step_percent: float | None = None,
    index_config_version: int = 1,
) -> Deployment:
    now = datetime.now(UTC)
    return Deployment(
        id=dep_id,
        project_name=project_name,
        project_id=uuid.uuid4(),
        index_config_id=uuid.uuid4(),
        index_config_version=index_config_version,
        status=status,
        traffic_weight=traffic_weight,
        rollout_step_percent=rollout_step_percent,
        top_k=10,
        cache_enabled=True,
        cache_ttl_seconds=3600,
        change_note="",
        created_at=now,
        updated_at=now,
        created_by="test",
    )


# ── Single live deployment enforcement ───────────────────────────────────────


class TestOneLiveDeploymentAtATime:
    @pytest.mark.asyncio
    async def test_second_deployment_rejected_when_one_is_active(self) -> None:
        """Attempting to create a second deployment while one is live raises ConflictError."""
        from retrieval_os.deployments.schemas import CreateDeploymentRequest

        existing = _dep()
        request = CreateDeploymentRequest(index_config_version=2, created_by="alice")

        mock_project = MagicMock()
        mock_project.is_archived = False
        mock_project.id = uuid.uuid4()
        mock_index_config = MagicMock()

        with (
            patch(
                "retrieval_os.deployments.service.project_repo.get_by_name",
                new=AsyncMock(return_value=mock_project),
            ),
            patch(
                "retrieval_os.deployments.service.project_repo.get_index_config",
                new=AsyncMock(return_value=mock_index_config),
            ),
            patch(
                "retrieval_os.deployments.service.deployment_repo.get_active_for_project",
                new=AsyncMock(return_value=existing),
            ),
        ):
            with pytest.raises(ConflictError):
                from retrieval_os.deployments.service import create_deployment

                await create_deployment(MagicMock(), "my-docs", request)

    @pytest.mark.asyncio
    async def test_deployment_allowed_when_no_active_deployment_exists(self) -> None:
        """A new deployment proceeds when the slot is empty."""
        from retrieval_os.deployments.schemas import CreateDeploymentRequest
        from retrieval_os.deployments.service import create_deployment

        request = CreateDeploymentRequest(index_config_version=1, created_by="alice")

        mock_project = MagicMock()
        mock_project.is_archived = False
        mock_project.id = uuid.uuid4()
        mock_project.name = "my-docs"

        mock_index_config = MagicMock()
        mock_index_config.id = uuid.uuid4()
        mock_index_config.embedding_provider = "sentence_transformers"
        mock_index_config.embedding_model = "BAAI/bge-m3"
        mock_index_config.embedding_normalize = True
        mock_index_config.embedding_batch_size = 32
        mock_index_config.index_backend = "qdrant"
        mock_index_config.index_collection = "docs_v1"
        mock_index_config.distance_metric = "cosine"

        mock_deployment = _dep(dep_id="new-dep")

        with (
            patch(
                "retrieval_os.deployments.service.project_repo.get_by_name",
                new=AsyncMock(return_value=mock_project),
            ),
            patch(
                "retrieval_os.deployments.service.project_repo.get_index_config",
                new=AsyncMock(return_value=mock_index_config),
            ),
            patch(
                "retrieval_os.deployments.service.deployment_repo.get_active_for_project",
                new=AsyncMock(return_value=None),  # slot is empty
            ),
            patch(
                "retrieval_os.deployments.service.deployment_repo.create",
                new=AsyncMock(return_value=mock_deployment),
            ),
            patch(
                "retrieval_os.deployments.service._activate",
                new=AsyncMock(return_value=mock_deployment),
            ),
            patch("retrieval_os.deployments.service.set_active_deployment", new=AsyncMock()),
            patch("retrieval_os.deployments.service.metrics.deployment_status"),
            patch("retrieval_os.deployments.service.metrics.deployment_traffic_weight"),
        ):
            response = await create_deployment(MagicMock(), "my-docs", request)

        assert response is not None


# ── Atomic rollback ───────────────────────────────────────────────────────────


class TestAtomicRollback:
    @pytest.mark.asyncio
    async def test_rollback_clears_redis_in_same_operation(self) -> None:
        """clear_active_deployment is called immediately after the DB status update —
        no window where a rolled-back deployment can serve traffic."""
        dep = _dep()
        request = RollbackRequest(reason="recall dropped to 0.45", created_by="ops")

        call_order: list[str] = []

        async def _update(*args: object, **kwargs: object) -> None:
            call_order.append("db_update")

        async def _clear(*args: object) -> None:
            call_order.append("redis_clear")

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.get_by_id",
                new=AsyncMock(return_value=dep),
            ),
            patch(
                "retrieval_os.deployments.service.deployment_repo.update_status",
                side_effect=_update,
            ),
            patch(
                "retrieval_os.deployments.service.clear_active_deployment",
                side_effect=_clear,
            ),
            patch("retrieval_os.deployments.service.fire_webhook_event", new=AsyncMock()),
            patch("retrieval_os.deployments.service.metrics.rollback_events_total"),
        ):
            session = MagicMock()
            session.refresh = AsyncMock()
            await rollback_deployment(session, "my-docs", "dep-001", request)

        assert call_order[0] == "db_update", "DB must be updated before Redis is cleared"
        assert call_order[1] == "redis_clear", "Redis must be cleared immediately after DB update"
        assert len(call_order) == 2

    @pytest.mark.asyncio
    async def test_rollback_fires_webhook_event(self) -> None:
        """A webhook event is emitted so downstream systems know the deployment changed."""
        dep = _dep()
        request = RollbackRequest(reason="error rate spiked", created_by="ops")
        webhook_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.get_by_id",
                new=AsyncMock(return_value=dep),
            ),
            patch(
                "retrieval_os.deployments.service.deployment_repo.update_status", new=AsyncMock()
            ),
            patch("retrieval_os.deployments.service.clear_active_deployment", new=AsyncMock()),
            patch(
                "retrieval_os.deployments.service.fire_webhook_event",
                new=webhook_mock,
            ),
            patch("retrieval_os.deployments.service.metrics.rollback_events_total"),
        ):
            session = MagicMock()
            session.refresh = AsyncMock()
            await rollback_deployment(session, "my-docs", "dep-001", request)

        webhook_mock.assert_awaited_once()
        event_payload = webhook_mock.call_args[0][1]
        assert event_payload["status"] == DeploymentStatus.ROLLED_BACK.value


# ── Gradual rollout progression ───────────────────────────────────────────────


class TestGradualRolloutProgression:
    @pytest.mark.asyncio
    async def test_each_step_advances_weight_by_step_percent(self) -> None:
        """Traffic weight increases by step_percent/100 each time the stepper runs."""
        dep = _dep(
            status=DeploymentStatus.ROLLING_OUT.value,
            traffic_weight=0.2,
            rollout_step_percent=20.0,
        )
        update_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_rolling_out",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.deployment_repo.update_status",
                new=update_mock,
            ),
            patch("retrieval_os.deployments.service.fire_webhook_event", new=AsyncMock()),
            patch("retrieval_os.deployments.service.set_active_deployment", new=AsyncMock()),
        ):
            count = await step_rolling_deployments(MagicMock())

        assert count == 1
        new_weight = update_mock.call_args[1]["traffic_weight"]
        assert abs(new_weight - 0.4) < 1e-6, f"Expected 0.4 (0.2 + 20%), got {new_weight}"
        assert update_mock.call_args[0][2] == DeploymentStatus.ROLLING_OUT.value

    @pytest.mark.asyncio
    async def test_rollout_promotes_to_active_at_full_weight(self) -> None:
        """When weight reaches 1.0, the deployment is promoted to ACTIVE automatically."""
        dep = _dep(
            status=DeploymentStatus.ROLLING_OUT.value,
            traffic_weight=0.8,
            rollout_step_percent=25.0,
        )
        update_mock = AsyncMock()
        fire_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_rolling_out",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.deployment_repo.update_status",
                new=update_mock,
            ),
            patch(
                "retrieval_os.deployments.service.fire_webhook_event",
                new=fire_mock,
            ),
            patch("retrieval_os.deployments.service.metrics.rollout_duration_seconds"),
        ):
            count = await step_rolling_deployments(MagicMock())

        assert count == 1
        assert update_mock.call_args[0][2] == DeploymentStatus.ACTIVE.value
        assert update_mock.call_args[1]["traffic_weight"] == 1.0
        fire_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_weight_never_exceeds_1_0(self) -> None:
        """Traffic weight is capped at exactly 1.0 regardless of step size."""
        dep = _dep(
            status=DeploymentStatus.ROLLING_OUT.value,
            traffic_weight=0.95,
            rollout_step_percent=50.0,  # would overshoot to 1.45 without cap
        )
        update_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_rolling_out",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.deployment_repo.update_status",
                new=update_mock,
            ),
            patch("retrieval_os.deployments.service.fire_webhook_event", new=AsyncMock()),
            patch("retrieval_os.deployments.service.metrics.rollout_duration_seconds"),
        ):
            await step_rolling_deployments(MagicMock())

        final_weight = update_mock.call_args[1]["traffic_weight"]
        assert final_weight == 1.0, f"Weight must not exceed 1.0; got {final_weight}"
