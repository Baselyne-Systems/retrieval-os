"""Unit tests for the Deployments domain (no live DB or Redis)."""

from __future__ import annotations

import uuid
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
            project_name="docs",
            project_id=uuid.uuid4(),
            index_config_id=uuid.uuid4(),
            index_config_version=1,
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
            index_config_version=1,
            created_by="alice",
        )
        assert req.index_config_version == 1
        assert req.rollout_step_percent is None
        assert req.top_k == 10
        assert req.cache_enabled is True

    def test_create_request_gradual(self) -> None:
        from retrieval_os.deployments.schemas import CreateDeploymentRequest

        req = CreateDeploymentRequest(
            index_config_version=2,
            rollout_step_percent=10.0,
            rollout_step_interval_seconds=60,
            created_by="alice",
        )
        assert req.rollout_step_percent == 10.0
        assert req.rollout_step_interval_seconds == 60

    def test_create_request_with_eval_dataset_uri(self) -> None:
        from retrieval_os.deployments.schemas import CreateDeploymentRequest

        req = CreateDeploymentRequest(
            index_config_version=1,
            eval_dataset_uri="s3://bucket/eval.jsonl",
            created_by="alice",
        )
        assert req.eval_dataset_uri == "s3://bucket/eval.jsonl"

    def test_create_request_eval_dataset_uri_defaults_to_none(self) -> None:
        from retrieval_os.deployments.schemas import CreateDeploymentRequest

        req = CreateDeploymentRequest(index_config_version=1, created_by="alice")
        assert req.eval_dataset_uri is None

    def test_create_request_with_search_config(self) -> None:
        from retrieval_os.deployments.schemas import CreateDeploymentRequest

        req = CreateDeploymentRequest(
            index_config_version=1,
            top_k=20,
            reranker="cross-encoder",
            rerank_top_k=5,
            cache_enabled=False,
            created_by="alice",
        )
        assert req.top_k == 20
        assert req.reranker == "cross-encoder"
        assert req.rerank_top_k == 5
        assert req.cache_enabled is False

    def test_create_request_index_config_version_must_be_positive(self) -> None:
        from retrieval_os.deployments.schemas import CreateDeploymentRequest

        with pytest.raises(Exception):
            CreateDeploymentRequest(index_config_version=0, created_by="alice")

    def test_deployment_response_includes_eval_dataset_uri(self) -> None:
        import uuid

        from retrieval_os.deployments.schemas import DeploymentResponse

        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        resp = DeploymentResponse(
            id="d1",
            project_name="p",
            project_id=uuid.uuid4(),
            index_config_id=uuid.uuid4(),
            index_config_version=1,
            top_k=10,
            rerank_top_k=None,
            reranker=None,
            hybrid_alpha=None,
            metadata_filters=None,
            tenant_isolation_field=None,
            cache_enabled=True,
            cache_ttl_seconds=3600,
            max_tokens_per_query=None,
            status="ACTIVE",
            traffic_weight=1.0,
            rollout_step_percent=None,
            rollout_step_interval_seconds=None,
            rollback_recall_threshold=None,
            rollback_error_rate_threshold=None,
            eval_dataset_uri="s3://bucket/eval.jsonl",
            change_note="",
            created_at=now,
            updated_at=now,
            created_by="alice",
            activated_at=now,
            rolled_back_at=None,
            rollback_reason=None,
        )
        assert resp.eval_dataset_uri == "s3://bucket/eval.jsonl"

    def test_rollback_request_valid(self) -> None:
        from retrieval_os.deployments.schemas import RollbackRequest

        req = RollbackRequest(reason="recall dropped below threshold", created_by="ops")
        assert req.reason == "recall dropped below threshold"

    def test_rollback_request_empty_reason_invalid(self) -> None:
        from retrieval_os.deployments.schemas import RollbackRequest

        with pytest.raises(Exception):
            RollbackRequest(reason="", created_by="ops")


# ── Traffic helpers ───────────────────────────────────────────────────────────


# ── Auto-eval trigger ─────────────────────────────────────────────────────────


class TestAutoEvalTrigger:
    @pytest.mark.asyncio
    async def test_auto_queue_eval_called_with_uri(self) -> None:
        """auto_queue_eval is called when eval_dataset_uri is set on deployment."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from retrieval_os.evaluations.service import auto_queue_eval

        deployment = MagicMock()
        deployment.eval_dataset_uri = "s3://bucket/eval.jsonl"
        deployment.index_config_version = 1
        deployment.top_k = 10

        mock_session = MagicMock()

        with patch(
            "retrieval_os.evaluations.service.queue_eval_job",
            new=AsyncMock(return_value=MagicMock(id="job-1")),
        ) as mock_queue:
            await auto_queue_eval(mock_session, "my-project", deployment)

        mock_queue.assert_awaited_once()
        req = mock_queue.call_args[0][1]
        assert req.project_name == "my-project"
        assert req.dataset_uri == "s3://bucket/eval.jsonl"
        assert req.created_by == "system:auto-eval"

    @pytest.mark.asyncio
    async def test_auto_queue_eval_skips_when_uri_none(self) -> None:
        """auto_queue_eval returns None immediately when eval_dataset_uri is None."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from retrieval_os.evaluations.service import auto_queue_eval

        deployment = MagicMock()
        deployment.eval_dataset_uri = None

        with patch(
            "retrieval_os.evaluations.service.queue_eval_job", new=AsyncMock()
        ) as mock_queue:
            await auto_queue_eval(MagicMock(), "p", deployment)

        # Return value is None when eval_dataset_uri is None
        mock_queue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auto_queue_eval_never_raises(self) -> None:
        """auto_queue_eval swallows exceptions so activation is never blocked."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from retrieval_os.evaluations.service import auto_queue_eval

        deployment = MagicMock()
        deployment.eval_dataset_uri = "s3://bucket/eval.jsonl"
        deployment.index_config_version = 1
        deployment.top_k = 10

        with patch(
            "retrieval_os.evaluations.service.queue_eval_job",
            new=AsyncMock(side_effect=RuntimeError("DB down")),
        ):
            # Must not raise
            await auto_queue_eval(MagicMock(), "p", deployment)

        # Return value is None on exception (no raise)


class TestTrafficKeys:
    def test_active_key_format(self) -> None:
        from retrieval_os.deployments.traffic import _active_key

        assert _active_key("my-plan") == "ros:deployment:my-plan:active"

    def test_project_config_key_format(self) -> None:
        from retrieval_os.deployments.traffic import _project_config_key

        assert _project_config_key("my-plan") == "ros:project:my-plan:active"
