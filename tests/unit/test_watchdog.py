"""Unit tests for the rollback watchdog (check_rollback_thresholds)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval_os.deployments.models import Deployment, DeploymentStatus
from retrieval_os.deployments.service import check_rollback_thresholds

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_deployment(
    *,
    recall_threshold: float | None = None,
    error_threshold: float | None = None,
    status: str = DeploymentStatus.ACTIVE.value,
    plan_name: str = "acme",
    plan_version: int = 1,
) -> Deployment:
    now = datetime.now(UTC)
    return Deployment(
        id="dep-001",
        plan_name=plan_name,
        plan_version=plan_version,
        status=status,
        traffic_weight=1.0,
        rollback_recall_threshold=recall_threshold,
        rollback_error_rate_threshold=error_threshold,
        change_note="",
        created_at=now,
        updated_at=now,
        created_by="test",
    )


def _make_eval(
    *,
    recall_at_5: float = 0.8,
    total_queries: int = 100,
    failed_queries: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="eval-001",
        plan_name="acme",
        plan_version=1,
        recall_at_5=recall_at_5,
        total_queries=total_queries,
        failed_queries=failed_queries,
        created_at=datetime.now(UTC),
    )


# ── No-op cases ───────────────────────────────────────────────────────────────


class TestWatchdogNoOp:
    @pytest.mark.asyncio
    async def test_no_live_deployments_returns_zero(self) -> None:
        mock_session = MagicMock()
        with patch(
            "retrieval_os.deployments.service.deployment_repo.list_live",
            new=AsyncMock(return_value=[]),
        ):
            result = await check_rollback_thresholds(mock_session)
        assert result == 0

    @pytest.mark.asyncio
    async def test_deployment_without_thresholds_skipped(self) -> None:
        dep = _make_deployment(recall_threshold=None, error_threshold=None)
        mock_session = MagicMock()
        with patch(
            "retrieval_os.deployments.service.deployment_repo.list_live",
            new=AsyncMock(return_value=[dep]),
        ):
            result = await check_rollback_thresholds(mock_session)
        assert result == 0

    @pytest.mark.asyncio
    async def test_no_eval_data_skips_rollback(self) -> None:
        dep = _make_deployment(recall_threshold=0.7)
        mock_session = MagicMock()
        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_plan",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await check_rollback_thresholds(mock_session)
        assert result == 0

    @pytest.mark.asyncio
    async def test_recall_above_threshold_no_rollback(self) -> None:
        dep = _make_deployment(recall_threshold=0.6)
        eval_job = _make_eval(recall_at_5=0.85)  # above threshold
        mock_session = MagicMock()
        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_plan",
                new=AsyncMock(return_value=eval_job),
            ),
        ):
            result = await check_rollback_thresholds(mock_session)
        assert result == 0

    @pytest.mark.asyncio
    async def test_error_rate_below_threshold_no_rollback(self) -> None:
        dep = _make_deployment(error_threshold=0.1)
        eval_job = _make_eval(total_queries=100, failed_queries=5)  # 5% < 10%
        mock_session = MagicMock()
        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_plan",
                new=AsyncMock(return_value=eval_job),
            ),
        ):
            result = await check_rollback_thresholds(mock_session)
        assert result == 0

    @pytest.mark.asyncio
    async def test_zero_total_queries_skips_error_rate_check(self) -> None:
        dep = _make_deployment(error_threshold=0.05)
        eval_job = _make_eval(total_queries=0, failed_queries=0)
        mock_session = MagicMock()
        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_plan",
                new=AsyncMock(return_value=eval_job),
            ),
        ):
            result = await check_rollback_thresholds(mock_session)
        assert result == 0


# ── Rollback triggered ────────────────────────────────────────────────────────


class TestWatchdogTriggersRollback:
    @pytest.mark.asyncio
    async def test_recall_below_threshold_triggers_rollback(self) -> None:
        dep = _make_deployment(recall_threshold=0.75)
        eval_job = _make_eval(recall_at_5=0.60)  # below threshold
        rollback_mock = AsyncMock()
        mock_session = MagicMock()
        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_plan",
                new=AsyncMock(return_value=eval_job),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            result = await check_rollback_thresholds(mock_session)

        assert result == 1
        rollback_mock.assert_awaited_once()
        _, dep_arg, reason_arg = rollback_mock.call_args[0]
        assert dep_arg.id == "dep-001"
        assert "recall@5" in reason_arg

    @pytest.mark.asyncio
    async def test_error_rate_above_threshold_triggers_rollback(self) -> None:
        dep = _make_deployment(error_threshold=0.05)
        eval_job = _make_eval(total_queries=100, failed_queries=20)  # 20% > 5%
        rollback_mock = AsyncMock()
        mock_session = MagicMock()
        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_plan",
                new=AsyncMock(return_value=eval_job),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            result = await check_rollback_thresholds(mock_session)

        assert result == 1
        _, _, reason_arg = rollback_mock.call_args[0]
        assert "error_rate" in reason_arg

    @pytest.mark.asyncio
    async def test_recall_checked_before_error_rate(self) -> None:
        """When both thresholds breach, reason should mention recall@5 (checked first)."""
        dep = _make_deployment(recall_threshold=0.80, error_threshold=0.05)
        eval_job = _make_eval(
            recall_at_5=0.50,  # below recall threshold
            total_queries=100,
            failed_queries=30,  # also above error threshold
        )
        rollback_mock = AsyncMock()
        mock_session = MagicMock()
        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_plan",
                new=AsyncMock(return_value=eval_job),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            await check_rollback_thresholds(mock_session)

        _, _, reason_arg = rollback_mock.call_args[0]
        assert "recall@5" in reason_arg

    @pytest.mark.asyncio
    async def test_multiple_deployments_each_checked(self) -> None:
        deps = [
            _make_deployment(recall_threshold=0.75, plan_name="plan-a"),
            _make_deployment(recall_threshold=0.75, plan_name="plan-b"),
        ]
        deps[0].id = "dep-a"
        deps[1].id = "dep-b"
        eval_job = _make_eval(recall_at_5=0.50)  # both breach
        rollback_mock = AsyncMock()
        mock_session = MagicMock()
        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=deps),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_plan",
                new=AsyncMock(return_value=eval_job),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            result = await check_rollback_thresholds(mock_session)

        assert result == 2
        assert rollback_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_rolling_out_deployment_also_checked(self) -> None:
        dep = _make_deployment(recall_threshold=0.75, status=DeploymentStatus.ROLLING_OUT.value)
        eval_job = _make_eval(recall_at_5=0.50)
        rollback_mock = AsyncMock()
        mock_session = MagicMock()
        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_plan",
                new=AsyncMock(return_value=eval_job),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            result = await check_rollback_thresholds(mock_session)

        assert result == 1

    @pytest.mark.asyncio
    async def test_exact_threshold_boundary_does_not_trigger(self) -> None:
        """recall@5 exactly equal to threshold should NOT trigger (strict less-than)."""
        dep = _make_deployment(recall_threshold=0.75)
        eval_job = _make_eval(recall_at_5=0.75)  # exactly at threshold
        rollback_mock = AsyncMock()
        mock_session = MagicMock()
        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_plan",
                new=AsyncMock(return_value=eval_job),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            result = await check_rollback_thresholds(mock_session)

        assert result == 0
        rollback_mock.assert_not_awaited()
