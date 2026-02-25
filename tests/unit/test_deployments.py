"""Unit tests for the Deployments domain (no live DB or Redis)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from retrieval_os.deployments.models import Deployment, DeploymentStatus

# ── DeploymentStatus enum ──────────────────────────────────────────────────────


class TestDeploymentStatus:
    def test_all_expected_statuses_exist(self) -> None:
        statuses = {s.value for s in DeploymentStatus}
        assert statuses == {
            "PENDING",
            "ROLLING_OUT",
            "ACTIVE",
            "ROLLING_BACK",
            "ROLLED_BACK",
            "FAILED",
        }

    def test_status_values_are_strings(self) -> None:
        for status in DeploymentStatus:
            assert isinstance(status.value, str)


# ── Deployment.is_live ────────────────────────────────────────────────────────


class TestDeploymentIsLive:
    def _make(self, status: str) -> Deployment:
        now = datetime.now(UTC)
        return Deployment(
            plan_name="docs",
            plan_version=1,
            status=status,
            traffic_weight=0.0,
            change_note="",
            created_at=now,
            updated_at=now,
            created_by="test",
        )

    def test_active_is_live(self) -> None:
        assert self._make(DeploymentStatus.ACTIVE.value).is_live

    def test_rolling_out_is_live(self) -> None:
        assert self._make(DeploymentStatus.ROLLING_OUT.value).is_live

    def test_pending_not_live(self) -> None:
        assert not self._make(DeploymentStatus.PENDING.value).is_live

    def test_rolled_back_not_live(self) -> None:
        assert not self._make(DeploymentStatus.ROLLED_BACK.value).is_live

    def test_failed_not_live(self) -> None:
        assert not self._make(DeploymentStatus.FAILED.value).is_live


# ── Deployment schemas ────────────────────────────────────────────────────────


class TestDeploymentSchemas:
    def test_create_request_valid(self) -> None:
        from retrieval_os.deployments.schemas import CreateDeploymentRequest

        req = CreateDeploymentRequest(
            plan_version=1,
            created_by="alice",
        )
        assert req.plan_version == 1
        assert req.rollout_step_percent is None

    def test_create_request_gradual(self) -> None:
        from retrieval_os.deployments.schemas import CreateDeploymentRequest

        req = CreateDeploymentRequest(
            plan_version=2,
            rollout_step_percent=10.0,
            rollout_step_interval_seconds=60,
            created_by="alice",
        )
        assert req.rollout_step_percent == 10.0
        assert req.rollout_step_interval_seconds == 60

    def test_create_request_plan_version_must_be_positive(self) -> None:
        from retrieval_os.deployments.schemas import CreateDeploymentRequest

        with pytest.raises(Exception):
            CreateDeploymentRequest(plan_version=0, created_by="alice")

    def test_rollback_request_valid(self) -> None:
        from retrieval_os.deployments.schemas import RollbackRequest

        req = RollbackRequest(reason="recall dropped below threshold", created_by="ops")
        assert req.reason == "recall dropped below threshold"

    def test_rollback_request_empty_reason_invalid(self) -> None:
        from retrieval_os.deployments.schemas import RollbackRequest

        with pytest.raises(Exception):
            RollbackRequest(reason="", created_by="ops")


# ── Traffic helpers ───────────────────────────────────────────────────────────


class TestTrafficKeys:
    def test_active_key_format(self) -> None:
        from retrieval_os.deployments.traffic import _active_key

        assert _active_key("my-plan") == "ros:deployment:my-plan:active"

    def test_plan_config_key_format(self) -> None:
        from retrieval_os.deployments.traffic import _plan_config_key

        assert _plan_config_key("my-plan") == "ros:plan:my-plan:current"
