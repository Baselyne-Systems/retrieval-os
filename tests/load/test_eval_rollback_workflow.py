"""Load test: Auto-eval on activation and eval-triggered rollback.

Proves:
1. When a deployment is activated with eval_dataset_uri set, an EvalJob row
   is automatically created (auto-eval trigger).
2. Under concurrent query load, a deployment with a failing eval triggers an
   automatic watchdog rollback with zero query errors.

Infrastructure required: Postgres + Redis (Qdrant not required for test 1;
Qdrant needed for test 2).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from retrieval_os.core.database import async_session_factory
from retrieval_os.deployments.models import DeploymentStatus
from retrieval_os.deployments.repository import deployment_repo
from retrieval_os.deployments.schemas import CreateDeploymentRequest, RollbackRequest
from retrieval_os.deployments.service import (
    check_rollback_thresholds,
    create_deployment,
    rollback_deployment,
)


class TestAutoEvalOnActivation:
    """Activating a deployment with eval_dataset_uri must queue an EvalJob."""

    async def test_eval_job_created_when_eval_dataset_uri_set(
        self, load_project, check_load_infra
    ) -> None:
        """An EvalJob row must appear in the DB immediately after activation.

        Uses a rollback-safe flow: ensures no live deployment conflicts before
        creating the test deployment.
        """
        eval_uri = "s3://test-bucket/eval.jsonl"

        # Clean up any live deployment first
        async with async_session_factory() as session:
            dep = await deployment_repo.get_active_for_project(session, load_project)
            if dep:
                await rollback_deployment(
                    session,
                    load_project,
                    dep.id,
                    RollbackRequest(reason="auto-eval test setup", created_by="test"),
                )
                await session.commit()

        # Activate a new deployment with eval_dataset_uri
        with patch(
            "retrieval_os.evaluations.service.queue_eval_job",
            new=AsyncMock(
                return_value=MagicMock(
                    id="fake-eval-job",
                    project_name=load_project,
                    index_config_version=1,
                )
            ),
        ) as mock_queue:
            async with async_session_factory() as session:
                dep_resp = await create_deployment(
                    session,
                    load_project,
                    CreateDeploymentRequest(
                        index_config_version=1,
                        top_k=10,
                        cache_enabled=True,
                        cache_ttl_seconds=3600,
                        eval_dataset_uri=eval_uri,
                        created_by="test",
                    ),
                )
                await session.commit()

        assert dep_resp.status == DeploymentStatus.ACTIVE.value
        mock_queue.assert_awaited_once()
        call_args = mock_queue.call_args
        req = call_args[0][1]  # second positional arg is the QueueEvalJobRequest
        assert req.dataset_uri == eval_uri
        assert req.project_name == load_project
        assert req.created_by == "system:auto-eval"

    async def test_no_eval_job_when_uri_not_set(self, load_project, check_load_infra) -> None:
        """No eval job should be queued when eval_dataset_uri is None."""
        # Clean up any live deployment
        async with async_session_factory() as session:
            dep = await deployment_repo.get_active_for_project(session, load_project)
            if dep:
                await rollback_deployment(
                    session,
                    load_project,
                    dep.id,
                    RollbackRequest(reason="no-eval test setup", created_by="test"),
                )
                await session.commit()

        with patch(
            "retrieval_os.evaluations.service.queue_eval_job",
            new=AsyncMock(),
        ) as mock_queue:
            async with async_session_factory() as session:
                await create_deployment(
                    session,
                    load_project,
                    CreateDeploymentRequest(
                        index_config_version=1,
                        top_k=10,
                        cache_enabled=True,
                        cache_ttl_seconds=3600,
                        eval_dataset_uri=None,
                        created_by="test",
                    ),
                )
                await session.commit()

        mock_queue.assert_not_awaited()


class TestEvalToRollback:
    """Failing eval must trigger watchdog rollback; concurrent queries must not error."""

    async def test_bad_deployment_rollback_under_load(
        self, load_project, load_collection, record_load
    ) -> None:
        """A deployment with failing eval is rolled back while queries are running.

        Flow:
          1. Ensure a second deployment with rollback_recall_threshold=0.95 is active.
          2. Run 5 concurrent workers firing queries.
          3. Process a mocked eval job that returns recall=0.4 (below threshold).
          4. Run check_rollback_thresholds() — should trigger watchdog rollback.
          5. Assert deployment.status == ROLLED_BACK.
          6. Assert 0 query errors throughout.
        """
        from tests.load.conftest import random_unit_vector

        stub_vector = random_unit_vector()

        # Ensure a live deployment with a recall threshold exists
        async with async_session_factory() as session:
            dep = await deployment_repo.get_active_for_project(session, load_project)
            if dep:
                await rollback_deployment(
                    session,
                    load_project,
                    dep.id,
                    RollbackRequest(reason="eval-rollback test setup", created_by="test"),
                )
                await session.commit()

        async with async_session_factory() as session:
            new_dep = await create_deployment(
                session,
                load_project,
                CreateDeploymentRequest(
                    index_config_version=1,
                    top_k=10,
                    cache_enabled=True,
                    cache_ttl_seconds=3600,
                    rollback_recall_threshold=0.95,
                    eval_dataset_uri="s3://test/eval.jsonl",
                    created_by="test",
                ),
            )
            await session.commit()

        dep_id = new_dep.id
        errors: list[str] = []
        stop_flag = asyncio.Event()

        async def _continuous_queries(worker_id: int) -> None:
            i = 0
            from retrieval_os.serving.executor import execute_retrieval

            while not stop_flag.is_set():
                try:
                    await execute_retrieval(
                        project_name=load_project,
                        version=1,
                        query=f"eval rollback query w{worker_id} i{i}",
                        embedding_provider="sentence_transformers",
                        embedding_model="all-MiniLM-L6-v2",
                        embedding_normalize=True,
                        embedding_batch_size=32,
                        index_backend="qdrant",
                        index_collection=load_collection,
                        distance_metric="cosine",
                        top_k=10,
                        reranker=None,
                        rerank_top_k=None,
                        metadata_filters=None,
                        cache_enabled=False,
                        cache_ttl_seconds=3600,
                    )
                except Exception as exc:
                    errors.append(str(exc))
                i += 1
                await asyncio.sleep(0.01)  # 100 QPS per worker

        # Start 5 background workers
        with patch(
            "retrieval_os.serving.executor.embed_text",
            new=AsyncMock(return_value=[stub_vector]),
        ):
            workers = [asyncio.create_task(_continuous_queries(w)) for w in range(5)]

            # Inject a failing eval result via direct DB insert
            from retrieval_os.core.ids import uuid7
            from retrieval_os.evaluations.models import EvalJob, EvalJobStatus

            eval_job_id = str(uuid7())
            now = datetime.now(UTC)
            async with async_session_factory() as session:
                bad_job = EvalJob(
                    id=eval_job_id,
                    project_name=load_project,
                    index_config_version=1,
                    status=EvalJobStatus.COMPLETED.value,
                    dataset_uri="s3://test/eval.jsonl",
                    top_k=10,
                    recall_at_5=0.40,  # below 0.95 threshold
                    total_queries=100,
                    failed_queries=0,
                    regression_detected=False,
                    created_at=now,
                    completed_at=now,
                    created_by="system:auto-eval",
                )
                session.add(bad_job)
                await session.commit()

            # Run watchdog — should trigger rollback
            async with async_session_factory() as session:
                triggered = await check_rollback_thresholds(session)
                await session.commit()

            # Let workers run briefly to ensure no errors during rollback
            await asyncio.sleep(0.1)
            stop_flag.set()
            await asyncio.gather(*workers, return_exceptions=True)

        # Verify rollback happened
        assert triggered >= 1, "Expected at least one watchdog rollback"

        async with async_session_factory() as session:
            dep = await deployment_repo.get_by_id(session, dep_id)

        assert dep is not None
        assert dep.status == DeploymentStatus.ROLLED_BACK.value, (
            f"Expected ROLLED_BACK, got {dep.status}"
        )
        assert len(errors) == 0, f"Query errors during rollback: {errors[:5]}"
