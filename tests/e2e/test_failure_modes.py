"""E2E: Production failure modes and graceful degradation

Tests proving the system handles adverse conditions correctly:

1. Unknown project — _load_project_config raises ProjectNotFoundError instead
   of returning a misconfigured dict that would crash the executor.

2. No active deployment — a project exists but nothing is deployed. Queries
   must fail fast with a descriptive error rather than returning empty results.

3. Watchdog no-op when no eval data — a deployment with recall guard rails but
   zero completed eval jobs must not be rolled back. Triggering rollback without
   data would take down the deployment every 30 s.

4. Watchdog triggered when recall is below threshold — a COMPLETED eval job
   with recall below the guard rail must trigger auto-rollback and clear the
   Redis serving config. This is the core quality protection mechanism.

5. Watchdog skips healthy deployments — a deployment whose eval metrics are
   ABOVE the threshold must stay ACTIVE.
"""

from __future__ import annotations

import pytest

from retrieval_os.core.database import async_session_factory
from retrieval_os.core.exceptions import ProjectNotFoundError
from retrieval_os.core.redis_client import get_redis
from retrieval_os.deployments.models import DeploymentStatus
from retrieval_os.deployments.repository import deployment_repo
from retrieval_os.deployments.service import check_rollback_thresholds
from retrieval_os.serving.query_router import _load_project_config, _project_redis_key
from tests.e2e.conftest import (
    insert_completed_eval_job,
    setup_deployment,
    setup_project_with_config,
)

# ── ProjectNotFound failure modes ─────────────────────────────────────────────


class TestProjectNotFound:
    async def test_missing_project_raises_not_found(self, project_name: str) -> None:
        """Querying a project name that does not exist in Postgres (and therefore
        has no Redis key) must raise ProjectNotFoundError immediately.

        The executor must never receive a None config — a KeyError on the hot
        path would produce a 500 instead of a 404.
        """
        redis = await get_redis()
        # Ensure no stale Redis key exists either
        await redis.delete(_project_redis_key(project_name))

        with pytest.raises(ProjectNotFoundError, match=project_name):
            async with async_session_factory() as session:
                await _load_project_config(project_name, session)

    async def test_project_without_active_deployment_raises_not_found(
        self, project_name: str
    ) -> None:
        """A project that exists in Postgres but has no ACTIVE/ROLLING_OUT
        deployment must raise ProjectNotFoundError — not return empty results.

        This prevents the query router from attempting to search an index that
        hasn't been built yet and potentially returning zero results silently.
        """
        await setup_project_with_config(project_name)  # project + index config, no deployment

        # No Redis key (never deployed)
        redis = await get_redis()
        await redis.delete(_project_redis_key(project_name))

        with pytest.raises(ProjectNotFoundError, match="no active deployment"):
            async with async_session_factory() as session:
                await _load_project_config(project_name, session)


# ── Rollback watchdog ──────────────────────────────────────────────────────────


class TestRollbackWatchdog:
    async def test_watchdog_does_not_rollback_when_no_eval_data(self, project_name: str) -> None:
        """A live deployment with a recall guard rail but zero completed eval
        jobs must not be rolled back.

        The watchdog skips deployments with no eval data because rolling back
        without evidence would unnecessarily take down every new deployment
        before its first eval cycle completes.
        """
        await setup_project_with_config(project_name)
        dep_id = await setup_deployment(
            project_name,
            rollback_recall_threshold=0.80,
        )

        async with async_session_factory() as session:
            triggered = await check_rollback_thresholds(session)
            await session.commit()

        assert triggered == 0, (
            f"Watchdog triggered {triggered} rollback(s) without any eval data — "
            "it should wait for the first eval cycle to complete"
        )

        async with async_session_factory() as session:
            dep = await deployment_repo.get_by_id(session, dep_id)
        assert dep is not None
        assert dep.status == DeploymentStatus.ACTIVE.value

    async def test_watchdog_triggers_rollback_when_recall_below_threshold(
        self, project_name: str
    ) -> None:
        """When a COMPLETED eval job reports recall@5 below the deployment's
        guard rail, the watchdog must roll back the deployment and clear the
        Redis serving config.

        This end-to-end test exercises the full rollback path:
        DB status transition → Redis key deletion → rollback metrics increment.
        """
        await setup_project_with_config(project_name)
        dep_id = await setup_deployment(
            project_name,
            rollback_recall_threshold=0.80,  # threshold: 80%
        )
        await insert_completed_eval_job(
            project_name,
            recall_at_5=0.60,  # below threshold → must trigger rollback
        )

        redis = await get_redis()
        key = _project_redis_key(project_name)
        assert await redis.exists(key), "Serving config must be in Redis before watchdog runs"

        async with async_session_factory() as session:
            triggered = await check_rollback_thresholds(session)
            await session.commit()

        assert triggered == 1, (
            f"Watchdog triggered {triggered} rollback(s); expected 1 (recall 0.60 < threshold 0.80)"
        )

        # Deployment must be ROLLED_BACK in the DB
        async with async_session_factory() as session:
            dep = await deployment_repo.get_by_id(session, dep_id)
        assert dep is not None
        assert dep.status == DeploymentStatus.ROLLED_BACK.value, (
            f"Expected ROLLED_BACK, got {dep.status}"
        )
        assert dep.traffic_weight == 0.0

        # Redis serving config must be cleared (fix for the stale-cache bug)
        assert not await redis.exists(key), (
            "Watchdog rolled back the deployment but left the serving config in Redis — "
            "queries will keep using the rolled-back config until TTL expires"
        )

    async def test_watchdog_does_not_rollback_when_recall_above_threshold(
        self, project_name: str
    ) -> None:
        """A deployment whose recall is above the guard rail must stay ACTIVE.

        This guards against false-positive rollbacks that would unnecessarily
        disrupt production traffic on a healthy deployment.
        """
        await setup_project_with_config(project_name)
        dep_id = await setup_deployment(
            project_name,
            rollback_recall_threshold=0.70,  # threshold: 70%
        )
        await insert_completed_eval_job(
            project_name,
            recall_at_5=0.85,  # above threshold → no rollback
        )

        async with async_session_factory() as session:
            triggered = await check_rollback_thresholds(session)
            await session.commit()

        assert triggered == 0, (
            f"Watchdog triggered {triggered} rollback(s) for a healthy deployment "
            f"(recall 0.85 > threshold 0.70)"
        )

        async with async_session_factory() as session:
            dep = await deployment_repo.get_by_id(session, dep_id)
        assert dep is not None
        assert dep.status == DeploymentStatus.ACTIVE.value
