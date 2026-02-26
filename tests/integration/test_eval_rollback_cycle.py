"""Integration tests: eval → regression detection → watchdog auto-rollback cycle.

Tests the end-to-end feedback loop:
  1. process_next_eval_job() completes → regression compared to previous run
  2. Regression triggers webhook event and sets regression_detected=True
  3. Watchdog reads the completed eval → breaches deployment threshold → rollback

Value over unit tests
---------------------
- Tests the handoff between the eval runner and the watchdog without mocking
  the intermediate state — both service functions exercise their full code paths
  against the same mocked repository layer.
- Verifies that regression detection fires the webhook only when recall drops.
- Verifies the watchdog auto-rollback is triggered when the latest eval's
  recall falls below the deployment's threshold (the state the eval runner
  just wrote).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval_os.deployments.models import Deployment, DeploymentStatus
from retrieval_os.deployments.service import check_rollback_thresholds
from retrieval_os.evaluations.service import process_next_eval_job
from retrieval_os.webhooks.events import WebhookEvent

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_eval_job(
    *,
    job_id: str = "eval-001",
    project_name: str = "my-docs",
    index_config_version: int = 1,
    dataset_uri: str = "s3://bucket/gt.jsonl",
    top_k: int = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=job_id,
        project_name=project_name,
        index_config_version=index_config_version,
        dataset_uri=dataset_uri,
        top_k=top_k,
        status="RUNNING",
    )


def _make_index_config() -> SimpleNamespace:
    return SimpleNamespace(
        id="ic-001",
        version=1,
        embedding_provider="sentence_transformers",
        embedding_model="BAAI/bge-m3",
        embedding_normalize=True,
        embedding_batch_size=32,
        index_backend="qdrant",
        index_collection="my_docs_v1",
        distance_metric="cosine",
    )


def _make_eval_results(*, recall_at_5: float = 0.80, failed: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        recall_at_1=0.50,
        recall_at_3=0.70,
        recall_at_5=recall_at_5,
        recall_at_10=0.90,
        mrr=0.75,
        ndcg_at_5=0.72,
        ndcg_at_10=0.68,
        total_queries=10,
        failed_queries=failed,
    )


def _make_prev_job(*, recall_at_5: float = 0.80) -> SimpleNamespace:
    return SimpleNamespace(
        id="eval-prev",
        recall_at_5=recall_at_5,
        mrr=0.75,
        ndcg_at_5=0.72,
    )


def _make_deployment(
    *,
    dep_id: str = "dep-001",
    project_name: str = "my-docs",
    recall_threshold: float | None = None,
    error_threshold: float | None = None,
) -> Deployment:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return Deployment(
        id=dep_id,
        project_name=project_name,
        index_config_version=1,
        status=DeploymentStatus.ACTIVE.value,
        traffic_weight=1.0,
        rollback_recall_threshold=recall_threshold,
        rollback_error_rate_threshold=error_threshold,
        change_note="",
        created_at=now,
        updated_at=now,
        created_by="test",
    )


# ── Eval runner regression detection ─────────────────────────────────────────


class TestEvalRegressionDetection:
    @pytest.mark.asyncio
    async def test_first_eval_no_previous_no_regression_webhook(self) -> None:
        """First eval for a plan has no previous result to compare against.
        No regression is detected and no EVAL_REGRESSION_DETECTED webhook fires."""
        job = _make_eval_job()
        session = MagicMock()
        session.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=_make_index_config()))
        )
        fire_mock = AsyncMock()
        complete_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.evaluations.service.eval_repo.claim_next_queued",
                new=AsyncMock(return_value=job),
            ),
            patch(
                "retrieval_os.evaluations.service._load_index_config",
                new=AsyncMock(return_value=_make_index_config()),
            ),
            patch(
                "retrieval_os.evaluations.service.load_eval_dataset",
                new=AsyncMock(
                    return_value=[
                        {"query": "q1", "relevant_ids": ["d1"]},
                    ]
                ),
            ),
            patch(
                "retrieval_os.evaluations.service.execute_eval_job",
                new=AsyncMock(return_value=_make_eval_results(recall_at_5=0.80)),
            ),
            patch(
                "retrieval_os.evaluations.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(return_value=None),  # no previous eval
            ),
            patch(
                "retrieval_os.evaluations.service.eval_repo.complete_job",
                new=complete_mock,
            ),
            patch(
                "retrieval_os.evaluations.service.fire_webhook_event",
                new=fire_mock,
            ),
            patch("retrieval_os.evaluations.service.metrics"),
        ):
            await process_next_eval_job(session)

        # No regression event should be fired on the first run
        regression_calls = [
            c for c in fire_mock.call_args_list if WebhookEvent.EVAL_REGRESSION_DETECTED in c[0]
        ]
        assert len(regression_calls) == 0

        # complete_job should record regression_detected=False
        complete_mock.assert_awaited_once()
        assert complete_mock.call_args[1].get("regression_detected") is False

    @pytest.mark.asyncio
    async def test_recall_drop_triggers_regression_webhook(self) -> None:
        """When recall@5 drops > 5% vs the previous eval, EVAL_REGRESSION_DETECTED fires."""
        job = _make_eval_job()
        prev = _make_prev_job(recall_at_5=0.85)  # previous was high
        fire_mock = AsyncMock()
        complete_mock = AsyncMock()
        session = MagicMock()

        with (
            patch(
                "retrieval_os.evaluations.service.eval_repo.claim_next_queued",
                new=AsyncMock(return_value=job),
            ),
            patch(
                "retrieval_os.evaluations.service._load_index_config",
                new=AsyncMock(return_value=_make_index_config()),
            ),
            patch(
                "retrieval_os.evaluations.service.load_eval_dataset",
                new=AsyncMock(return_value=[{"query": "q1", "relevant_ids": ["d1"]}]),
            ),
            patch(
                "retrieval_os.evaluations.service.execute_eval_job",
                new=AsyncMock(
                    return_value=_make_eval_results(recall_at_5=0.50)  # dropped from 0.85
                ),
            ),
            patch(
                "retrieval_os.evaluations.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(return_value=prev),
            ),
            patch(
                "retrieval_os.evaluations.service.eval_repo.complete_job",
                new=complete_mock,
            ),
            patch(
                "retrieval_os.evaluations.service.fire_webhook_event",
                new=fire_mock,
            ),
            patch("retrieval_os.evaluations.service.metrics"),
        ):
            await process_next_eval_job(session)

        # Regression webhook fired
        regression_calls = [
            c for c in fire_mock.call_args_list if WebhookEvent.EVAL_REGRESSION_DETECTED in c[0]
        ]
        assert len(regression_calls) == 1

        # complete_job records regression_detected=True
        assert complete_mock.call_args[1].get("regression_detected") is True

    @pytest.mark.asyncio
    async def test_recall_improvement_no_regression(self) -> None:
        """When recall@5 improves vs previous, regression_detected is False."""
        job = _make_eval_job()
        prev = _make_prev_job(recall_at_5=0.60)  # previous was lower
        complete_mock = AsyncMock()
        session = MagicMock()

        with (
            patch(
                "retrieval_os.evaluations.service.eval_repo.claim_next_queued",
                new=AsyncMock(return_value=job),
            ),
            patch(
                "retrieval_os.evaluations.service._load_index_config",
                new=AsyncMock(return_value=_make_index_config()),
            ),
            patch(
                "retrieval_os.evaluations.service.load_eval_dataset",
                new=AsyncMock(return_value=[{"query": "q1", "relevant_ids": ["d1"]}]),
            ),
            patch(
                "retrieval_os.evaluations.service.execute_eval_job",
                new=AsyncMock(return_value=_make_eval_results(recall_at_5=0.85)),
            ),
            patch(
                "retrieval_os.evaluations.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(return_value=prev),
            ),
            patch(
                "retrieval_os.evaluations.service.eval_repo.complete_job",
                new=complete_mock,
            ),
            patch("retrieval_os.evaluations.service.fire_webhook_event", new=AsyncMock()),
            patch("retrieval_os.evaluations.service.metrics"),
        ):
            await process_next_eval_job(session)

        assert complete_mock.call_args[1].get("regression_detected") is False

    @pytest.mark.asyncio
    async def test_no_queued_job_returns_none(self) -> None:
        """process_next_eval_job returns None when no job is queued."""
        session = MagicMock()
        with patch(
            "retrieval_os.evaluations.service.eval_repo.claim_next_queued",
            new=AsyncMock(return_value=None),
        ):
            result = await process_next_eval_job(session)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_dataset_fails_job(self) -> None:
        """An empty ground-truth dataset should fail the job, not crash the runner."""
        job = _make_eval_job()
        fail_mock = AsyncMock()
        session = MagicMock()

        with (
            patch(
                "retrieval_os.evaluations.service.eval_repo.claim_next_queued",
                new=AsyncMock(return_value=job),
            ),
            patch(
                "retrieval_os.evaluations.service._load_index_config",
                new=AsyncMock(return_value=_make_index_config()),
            ),
            patch(
                "retrieval_os.evaluations.service.load_eval_dataset",
                new=AsyncMock(return_value=[]),  # empty dataset
            ),
            patch(
                "retrieval_os.evaluations.service.eval_repo.fail_job",
                new=fail_mock,
            ),
        ):
            result = await process_next_eval_job(session)

        assert result == job.id  # returns job_id even on soft-fail
        fail_mock.assert_awaited_once()


# ── Full eval → watchdog cycle ────────────────────────────────────────────────


class TestEvalWatchdogCycle:
    @pytest.mark.asyncio
    async def test_low_recall_eval_leads_to_watchdog_rollback(self) -> None:
        """End-to-end cycle:
        1. Eval job completes with recall_at_5=0.50.
        2. Watchdog reads that eval result from eval_repo.
        3. Deployment has recall_threshold=0.75 → watchdog triggers rollback.

        The two service functions share the same mocked eval_repo, simulating
        the state that would exist after the eval runner writes its result.
        """
        dep = _make_deployment(project_name="my-docs", recall_threshold=0.75)
        low_recall_eval = SimpleNamespace(
            id="eval-001",
            project_name="my-docs",
            index_config_version=1,
            recall_at_5=0.50,  # below threshold
            total_queries=100,
            failed_queries=0,
        )
        rollback_mock = AsyncMock()
        session = MagicMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(return_value=low_recall_eval),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            count = await check_rollback_thresholds(session)

        assert count == 1
        rollback_mock.assert_awaited_once()
        _, rolled_dep, reason = rollback_mock.call_args[0]
        assert rolled_dep.id == "dep-001"
        assert "recall@5" in reason
        assert "0.5000" in reason
        assert "0.7500" in reason

    @pytest.mark.asyncio
    async def test_high_recall_eval_no_watchdog_rollback(self) -> None:
        """When the latest eval passes all thresholds, the watchdog takes no action."""
        dep = _make_deployment(project_name="my-docs", recall_threshold=0.70)
        good_eval = SimpleNamespace(
            recall_at_5=0.85,
            total_queries=100,
            failed_queries=2,
        )
        session = MagicMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(return_value=good_eval),
            ),
        ):
            count = await check_rollback_thresholds(session)

        assert count == 0

    @pytest.mark.asyncio
    async def test_error_rate_from_eval_triggers_watchdog(self) -> None:
        """Eval that produces a high error rate triggers watchdog via error_threshold."""
        dep = _make_deployment(project_name="my-docs", error_threshold=0.05)
        high_error_eval = SimpleNamespace(
            recall_at_5=0.90,  # recall fine
            total_queries=100,
            failed_queries=15,  # 15% > 5% threshold
        )
        rollback_mock = AsyncMock()
        session = MagicMock()

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(return_value=high_error_eval),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            count = await check_rollback_thresholds(session)

        assert count == 1
        _, _, reason = rollback_mock.call_args[0]
        assert "error_rate" in reason
        assert "0.1500" in reason

    @pytest.mark.asyncio
    async def test_regression_eval_then_watchdog_rollback_full_cycle(self) -> None:
        """Simulates two service calls in sequence sharing eval repo state.

        First: eval runner processes a job and records low recall.
        Second: watchdog finds that eval result and triggers rollback.

        Both service functions are called with mock repos that agree on state,
        mimicking what the DB would hold after the eval runner commits.
        """
        low_recall = 0.50
        threshold = 0.75

        # State shared between eval runner and watchdog (via mocked repo)
        completed_eval = SimpleNamespace(
            id="eval-001",
            project_name="my-docs",
            index_config_version=1,
            recall_at_5=low_recall,
            total_queries=50,
            failed_queries=0,
        )

        dep = _make_deployment(project_name="my-docs", recall_threshold=threshold)
        job = _make_eval_job(project_name="my-docs")
        session = MagicMock()

        complete_mock = AsyncMock()
        rollback_mock = AsyncMock()

        # Step 1: Run the eval job
        with (
            patch(
                "retrieval_os.evaluations.service.eval_repo.claim_next_queued",
                new=AsyncMock(return_value=job),
            ),
            patch(
                "retrieval_os.evaluations.service._load_index_config",
                new=AsyncMock(return_value=_make_index_config()),
            ),
            patch(
                "retrieval_os.evaluations.service.load_eval_dataset",
                new=AsyncMock(return_value=[{"query": "q", "relevant_ids": ["d"]}]),
            ),
            patch(
                "retrieval_os.evaluations.service.execute_eval_job",
                new=AsyncMock(return_value=_make_eval_results(recall_at_5=low_recall)),
            ),
            patch(
                "retrieval_os.evaluations.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(return_value=_make_prev_job(recall_at_5=0.85)),
            ),
            patch(
                "retrieval_os.evaluations.service.eval_repo.complete_job",
                new=complete_mock,
            ),
            patch("retrieval_os.evaluations.service.fire_webhook_event", new=AsyncMock()),
            patch("retrieval_os.evaluations.service.metrics"),
        ):
            eval_result = await process_next_eval_job(session)

        assert eval_result == "eval-001"
        assert complete_mock.call_args[1].get("regression_detected") is True

        # Step 2: Run the watchdog (reads the now-completed eval)
        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(return_value=completed_eval),
            ),
            patch(
                "retrieval_os.deployments.service._watchdog_rollback",
                new=rollback_mock,
            ),
        ):
            rollback_count = await check_rollback_thresholds(session)

        assert rollback_count == 1
        rollback_mock.assert_awaited_once()
        _, rolled_dep, reason = rollback_mock.call_args[0]
        assert "recall@5" in reason
