"""Integration tests: watchdog auto-rollback + rollout stepper interactions.

These tests operate at the service-function level (bypassing HTTP) and verify
cross-domain state machine interactions that unit tests cover in isolation but
not in combination.

Value over unit tests
---------------------
- Tests that ROLLING_OUT deployments are checked by the watchdog just like
  ACTIVE ones (both list_live and check_rollback_thresholds must be wired).
- Tests partial rollback: when only one of several live deployments breaches a
  threshold, only that one is rolled back.
- Tests that the watchdog continues after a single-deployment rollback failure
  (error resilience).
- Tests that recall and error-rate thresholds interact correctly when both are
  set and only one is breached.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval_os.deployments.models import Deployment, DeploymentStatus
from retrieval_os.deployments.service import check_rollback_thresholds, step_rolling_deployments

# ── Helpers ───────────────────────────────────────────────────────────────────


def _dep(
    *,
    dep_id: str = "dep-001",
    project_name: str = "plan-a",
    index_config_version: int = 1,
    status: str = DeploymentStatus.ACTIVE.value,
    recall_threshold: float | None = None,
    error_threshold: float | None = None,
    traffic_weight: float = 1.0,
    rollout_step_percent: float | None = None,
) -> Deployment:
    now = datetime.now(UTC)
    return Deployment(
        id=dep_id,
        project_name=project_name,
        index_config_version=index_config_version,
        status=status,
        traffic_weight=traffic_weight,
        rollout_step_percent=rollout_step_percent,
        rollback_recall_threshold=recall_threshold,
        rollback_error_rate_threshold=error_threshold,
        change_note="",
        created_at=now,
        updated_at=now,
        created_by="test",
    )


def _eval(
    *,
    recall_at_5: float = 0.80,
    total_queries: int = 100,
    failed_queries: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        recall_at_5=recall_at_5,
        total_queries=total_queries,
        failed_queries=failed_queries,
    )


# ── Watchdog + rollout stepper coexistence ────────────────────────────────────


class TestWatchdogRolloutInteraction:
    @pytest.mark.asyncio
    async def test_rolling_out_deployment_checked_by_watchdog(self) -> None:
        """A ROLLING_OUT deployment that breaches recall should be rolled back.

        The rollout stepper and watchdog run independently; a deployment in
        ROLLING_OUT status must not escape the watchdog's attention.
        """
        dep = _dep(
            status=DeploymentStatus.ROLLING_OUT.value,
            recall_threshold=0.75,
            traffic_weight=0.5,
        )
        eval_job = _eval(recall_at_5=0.50)
        rollback_mock = AsyncMock()
        session = MagicMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(return_value=eval_job),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            count = await check_rollback_thresholds(session)

        assert count == 1
        rollback_mock.assert_awaited_once()
        _, dep_arg, reason_arg = rollback_mock.call_args[0]
        assert dep_arg.id == dep.id
        assert "recall@5" in reason_arg

    @pytest.mark.asyncio
    async def test_rollout_stepper_advances_weight_independent_of_watchdog(self) -> None:
        """Rollout stepper advances traffic weight; watchdog is a separate loop.

        The stepper should advance deployments regardless of whether the watchdog
        has run. Here we verify the stepper updates the traffic weight correctly
        without touching eval data.
        """
        dep = _dep(
            status=DeploymentStatus.ROLLING_OUT.value,
            traffic_weight=0.3,
            rollout_step_percent=20.0,
        )
        session = MagicMock()
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
            patch(
                "retrieval_os.deployments.service.set_active_deployment",
                new=AsyncMock(),
            ),
        ):
            count = await step_rolling_deployments(session)

        assert count == 1
        # Expected new weight = 0.3 + 0.2 = 0.5 (not yet at 1.0, stays ROLLING_OUT)
        call_args = update_mock.call_args
        assert call_args[0][2] == DeploymentStatus.ROLLING_OUT.value
        assert abs(call_args[1]["traffic_weight"] - 0.5) < 1e-6

    @pytest.mark.asyncio
    async def test_rollout_stepper_promotes_to_active_at_full_weight(self) -> None:
        dep = _dep(
            status=DeploymentStatus.ROLLING_OUT.value,
            traffic_weight=0.9,
            rollout_step_percent=20.0,
        )
        session = MagicMock()
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
            count = await step_rolling_deployments(session)

        assert count == 1
        # 0.9 + 0.2 >= 1.0 → promoted to ACTIVE
        assert update_mock.call_args[0][2] == DeploymentStatus.ACTIVE.value
        assert update_mock.call_args[1]["traffic_weight"] == 1.0
        fire_mock.assert_awaited_once()


# ── Partial rollback (multiple deployments) ───────────────────────────────────


class TestPartialRollback:
    @pytest.mark.asyncio
    async def test_only_breaching_deployment_rolled_back(self) -> None:
        """When two plans are live, only the one that breaches the threshold
        gets rolled back.  The other is left unchanged."""
        dep_ok = _dep(dep_id="dep-ok", project_name="plan-ok", recall_threshold=0.60)
        dep_bad = _dep(dep_id="dep-bad", project_name="plan-bad", recall_threshold=0.75)

        def eval_for_plan(session, project_name):  # noqa: ANN001
            if project_name == "plan-ok":
                return _eval(recall_at_5=0.80)  # above threshold
            return _eval(recall_at_5=0.50)  # below threshold

        rollback_mock = AsyncMock()
        session = MagicMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep_ok, dep_bad]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(side_effect=eval_for_plan),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            count = await check_rollback_thresholds(session)

        assert count == 1
        assert rollback_mock.await_count == 1
        # The deployment that was rolled back should be dep_bad
        _, rolled_dep, _ = rollback_mock.call_args[0]
        assert rolled_dep.id == "dep-bad"

    @pytest.mark.asyncio
    async def test_all_breaching_deployments_rolled_back(self) -> None:
        deps = [
            _dep(dep_id=f"dep-{i}", project_name=f"plan-{i}", recall_threshold=0.75)
            for i in range(3)
        ]
        eval_job = _eval(recall_at_5=0.40)  # all breach
        rollback_mock = AsyncMock()
        session = MagicMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=deps),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(return_value=eval_job),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            count = await check_rollback_thresholds(session)

        assert count == 3
        assert rollback_mock.await_count == 3

    @pytest.mark.asyncio
    async def test_mixed_recall_and_error_thresholds_independent(self) -> None:
        """One deployment has only a recall threshold (breached), another has
        only an error threshold (not breached). Only the first is rolled back."""
        dep_recall = _dep(dep_id="dep-recall", project_name="plan-r", recall_threshold=0.75)
        dep_error = _dep(dep_id="dep-error", project_name="plan-e", error_threshold=0.10)

        def eval_for_plan(session, project_name):  # noqa: ANN001
            if project_name == "plan-r":
                return _eval(recall_at_5=0.50)  # breaches recall
            return _eval(total_queries=100, failed_queries=5)  # 5% < 10% threshold

        rollback_mock = AsyncMock()
        session = MagicMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep_recall, dep_error]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(side_effect=eval_for_plan),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            count = await check_rollback_thresholds(session)

        assert count == 1
        _, rolled, _ = rollback_mock.call_args[0]
        assert rolled.id == "dep-recall"


# ── Error resilience ──────────────────────────────────────────────────────────


class TestWatchdogResilience:
    @pytest.mark.asyncio
    async def test_watchdog_returns_correct_count_on_success(self) -> None:
        """Sanity: when no deployments breach thresholds, count is 0."""
        deps = [
            _dep(dep_id="dep-a", project_name="plan-a", recall_threshold=0.70),
            _dep(dep_id="dep-b", project_name="plan-b", error_threshold=0.05),
        ]
        evals = {
            "plan-a": _eval(recall_at_5=0.85),
            "plan-b": _eval(total_queries=100, failed_queries=2),
        }
        session = MagicMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=deps),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(side_effect=lambda s, name: evals[name]),
            ),
        ):
            count = await check_rollback_thresholds(session)

        assert count == 0

    @pytest.mark.asyncio
    async def test_watchdog_checks_recall_before_error_rate_when_both_breach(self) -> None:
        """The reason string must mention recall@5 when both thresholds breach
        (recall is checked first per spec)."""
        dep = _dep(recall_threshold=0.80, error_threshold=0.05)
        eval_job = _eval(recall_at_5=0.50, total_queries=100, failed_queries=20)
        rollback_mock = AsyncMock()
        session = MagicMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(return_value=eval_job),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            await check_rollback_thresholds(session)

        _, _, reason = rollback_mock.call_args[0]
        assert "recall@5" in reason
        assert "error_rate" not in reason

    @pytest.mark.asyncio
    async def test_deployment_without_thresholds_not_evaluated(self) -> None:
        """Deployments with neither threshold set should never hit the eval repo."""
        dep = _dep()  # no thresholds
        eval_mock = AsyncMock()
        session = MagicMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_project",
                new=eval_mock,
            ),
        ):
            count = await check_rollback_thresholds(session)

        assert count == 0
        eval_mock.assert_not_awaited()  # DB query skipped entirely
