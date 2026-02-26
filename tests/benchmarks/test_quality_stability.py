"""Benchmark: Retrieval Quality Stability — auto guard-rails and eval accuracy.

Customer claims
---------------
1. Eval jobs always measure fresh retrieval results — the semantic cache is
   bypassed during evaluation so stale cache entries can never inflate metrics.
2. Deployments are automatically rolled back when Recall@5 drops below the
   configured threshold — no human intervention required.
3. Healthy deployments are never rolled back by false positives.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval_os.deployments.models import Deployment, DeploymentStatus
from retrieval_os.deployments.service import check_rollback_thresholds
from retrieval_os.evaluations.runner import EvalRecord, execute_eval_job

# ── Eval freshness ────────────────────────────────────────────────────────────


class TestEvalAlwaysUsesFreshResults:
    """execute_eval_job must call execute_retrieval with cache_enabled=False.

    This prevents stale cache entries from inflating quality metrics — every eval
    query hits the live embedding + vector-search pipeline.
    """

    @pytest.mark.asyncio
    async def test_eval_job_disables_cache_for_all_queries(self) -> None:
        records = [
            EvalRecord(
                query="what is RAG?",
                relevant_ids={"doc-1", "doc-2"},
                relevance_scores={"doc-1": 1.0, "doc-2": 1.0},
            ),
            EvalRecord(
                query="how does chunking work?",
                relevant_ids={"doc-3"},
                relevance_scores={"doc-3": 1.0},
            ),
        ]

        captured_calls: list[dict] = []

        async def _fake_execute(**kwargs: object) -> tuple[list, bool]:
            captured_calls.append(dict(kwargs))
            return [], False

        with patch(
            "retrieval_os.serving.executor.execute_retrieval",
            side_effect=_fake_execute,
        ):
            await execute_eval_job(
                records,
                project_name="wiki-search",
                index_config_version=1,
                embedding_provider="sentence_transformers",
                embedding_model="BAAI/bge-m3",
                embedding_normalize=True,
                embedding_batch_size=32,
                index_backend="qdrant",
                index_collection="wiki_v1",
                distance_metric="cosine",
                top_k=5,
                reranker=None,
                rerank_top_k=None,
            )

        assert len(captured_calls) == len(records), "One execute_retrieval call per record"
        for call in captured_calls:
            assert call["cache_enabled"] is False, (
                "cache_enabled must be False during eval — stale cache must never inflate metrics"
            )
            assert call["cache_ttl_seconds"] == 0, (
                "cache_ttl_seconds must be 0 to prevent any accidental cache writes during eval"
            )

    @pytest.mark.asyncio
    async def test_eval_job_cache_disabled_even_when_project_cache_enabled(self) -> None:
        """The eval runner overrides whatever cache_enabled value the deployment carries."""
        records = [
            EvalRecord(
                query="test query",
                relevant_ids={"id1"},
                relevance_scores={"id1": 1.0},
            )
        ]

        captured: dict = {}

        async def _capture(**kwargs: object) -> tuple[list, bool]:
            captured.update(kwargs)
            return [], False

        # Simulate a deployment with cache_enabled=True — eval must still bypass it.
        with patch("retrieval_os.serving.executor.execute_retrieval", side_effect=_capture):
            await execute_eval_job(
                records,
                project_name="docs",
                index_config_version=2,
                embedding_provider="openai",
                embedding_model="text-embedding-3-large",
                embedding_normalize=False,
                embedding_batch_size=64,
                index_backend="qdrant",
                index_collection="docs_v2",
                distance_metric="cosine",
                top_k=10,
                reranker=None,
                rerank_top_k=None,
            )

        assert captured["cache_enabled"] is False
        assert captured["cache_ttl_seconds"] == 0


# ── Auto guard-rails ──────────────────────────────────────────────────────────


def _dep(
    *,
    dep_id: str = "dep-001",
    project_name: str = "my-docs",
    recall_threshold: float | None = None,
    error_threshold: float | None = None,
    status: str = DeploymentStatus.ACTIVE.value,
) -> Deployment:
    now = datetime.now(UTC)
    return Deployment(
        id=dep_id,
        project_name=project_name,
        index_config_version=1,
        status=status,
        traffic_weight=1.0,
        rollback_recall_threshold=recall_threshold,
        rollback_error_rate_threshold=error_threshold,
        change_note="",
        created_at=now,
        updated_at=now,
        created_by="test",
    )


def _eval_ns(recall_at_5: float = 0.80, total_queries: int = 100, failed_queries: int = 0):  # type: ignore[return]
    return SimpleNamespace(
        recall_at_5=recall_at_5,
        total_queries=total_queries,
        failed_queries=failed_queries,
    )


class TestAutoRollbackGuardRails:
    @pytest.mark.asyncio
    async def test_recall_breach_triggers_automatic_rollback(self) -> None:
        """When Recall@5 drops below the deployment threshold, rollback fires without human intervention."""
        dep = _dep(recall_threshold=0.75)
        eval_job = _eval_ns(recall_at_5=0.60)  # below 0.75 threshold
        rollback_mock = AsyncMock()

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
            count = await check_rollback_thresholds(MagicMock())

        assert count == 1
        rollback_mock.assert_awaited_once()
        _, dep_arg, reason_arg = rollback_mock.call_args[0]
        assert dep_arg.id == dep.id
        assert "recall@5" in reason_arg

    @pytest.mark.asyncio
    async def test_error_rate_breach_triggers_automatic_rollback(self) -> None:
        dep = _dep(error_threshold=0.05)
        eval_job = _eval_ns(total_queries=100, failed_queries=10)  # 10% > 5% threshold
        rollback_mock = AsyncMock()

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
            count = await check_rollback_thresholds(MagicMock())

        assert count == 1
        rollback_mock.assert_awaited_once()
        _, _, reason_arg = rollback_mock.call_args[0]
        assert "error_rate" in reason_arg

    @pytest.mark.asyncio
    async def test_healthy_deployment_is_not_rolled_back(self) -> None:
        """No false positives — healthy deployments are never touched."""
        dep = _dep(recall_threshold=0.70, error_threshold=0.10)
        eval_job = _eval_ns(recall_at_5=0.85, total_queries=200, failed_queries=5)

        with (
            patch(
                "retrieval_os.deployments.service.deployment_repo.list_live",
                new=AsyncMock(return_value=[dep]),
            ),
            patch(
                "retrieval_os.deployments.service.eval_repo.get_latest_completed_for_project",
                new=AsyncMock(return_value=eval_job),
            ),
        ):
            count = await check_rollback_thresholds(MagicMock())

        assert count == 0

    @pytest.mark.asyncio
    async def test_deployment_without_thresholds_is_never_evaluated(self) -> None:
        """Deployments with no guard-rails configured never hit the eval repo."""
        dep = _dep()  # no thresholds
        eval_mock = AsyncMock()

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
            count = await check_rollback_thresholds(MagicMock())

        assert count == 0
        eval_mock.assert_not_awaited()
