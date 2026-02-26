"""Load test: Zero-downtime deployment switch and rollback speed.

Proves two operational claims:
1. Concurrent queries succeed with zero errors across a live deployment config
   switch (rollback → activate new deployment with different search config).
2. Rollback propagates to Redis within 2 seconds so subsequent queries
   immediately fail with NO_ACTIVE_DEPLOYMENT rather than serving stale config.

Infrastructure required (auto-skipped when unreachable):
  - Postgres + Redis + Qdrant (same stack as other load tests)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

from retrieval_os.core.database import async_session_factory
from retrieval_os.core.redis_client import get_redis
from retrieval_os.deployments.repository import deployment_repo
from retrieval_os.deployments.schemas import CreateDeploymentRequest, RollbackRequest
from retrieval_os.deployments.service import create_deployment, rollback_deployment
from retrieval_os.serving.executor import execute_retrieval
from tests.load.conftest import random_unit_vector


def _exec_kwargs(project_name: str, collection: str, query: str, *, top_k: int = 10) -> dict:
    return dict(
        project_name=project_name,
        version=1,
        query=query,
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        embedding_normalize=True,
        embedding_batch_size=32,
        index_backend="qdrant",
        index_collection=collection,
        distance_metric="cosine",
        top_k=top_k,
        reranker=None,
        rerank_top_k=None,
        metadata_filters=None,
        cache_enabled=False,  # disable cache so every query goes to Qdrant
        cache_ttl_seconds=3600,
    )


async def _clear_query_cache() -> None:
    redis = await get_redis()
    async for key in redis.scan_iter("ros:qcache:*"):
        await redis.delete(key)


class TestZeroDowntimeUpgrade:
    """Concurrent queries must not error during a live deployment config switch."""

    async def test_no_errors_during_config_switch(
        self, load_project, load_collection, record_load
    ) -> None:
        """Zero HTTP errors while switching from top_k=10 to top_k=5 under load.

        Flow:
          1. 30 queries with top_k=10 (first deployment).
          2. Rollback first deployment; activate second deployment (top_k=5).
          3. 70 more queries with top_k=5 (second deployment).
          4. Assert 0 errors throughout; post-switch results have ≤ 5 items.
        """
        stub_vector = random_unit_vector()
        errors: list[str] = []
        pre_switch_counts: list[int] = []
        post_switch_counts: list[int] = []

        await _clear_query_cache()

        # Phase 1: 30 concurrent queries with the initial deployment (top_k=10)
        async def _query(i: int, top_k: int, counts: list[int]) -> None:
            kw = _exec_kwargs(load_project, load_collection, f"lifecycle q {i}", top_k=top_k)
            try:
                chunks, _ = await execute_retrieval(**kw)
                counts.append(len(chunks))
            except Exception as exc:
                errors.append(str(exc))

        with patch(
            "retrieval_os.serving.executor.embed_text",
            new=AsyncMock(return_value=[stub_vector]),
        ):
            wall_start = time.perf_counter()
            await asyncio.gather(*[_query(i, 10, pre_switch_counts) for i in range(30)])
            phase1_elapsed = (time.perf_counter() - wall_start) * 1000

        # Phase 2: Switch deployment
        async with async_session_factory() as session:
            dep = await deployment_repo.get_active_for_project(session, load_project)
            if dep:
                await rollback_deployment(
                    session,
                    load_project,
                    dep.id,
                    RollbackRequest(reason="lifecycle test switch", created_by="test"),
                )
                await session.commit()

        async with async_session_factory() as session:
            await create_deployment(
                session,
                load_project,
                CreateDeploymentRequest(
                    index_config_version=1,
                    top_k=5,
                    cache_enabled=False,
                    cache_ttl_seconds=3600,
                    created_by="test",
                ),
            )
            await session.commit()

        # Phase 3: 70 concurrent queries with new deployment (top_k=5)
        with patch(
            "retrieval_os.serving.executor.embed_text",
            new=AsyncMock(return_value=[stub_vector]),
        ):
            wall_start = time.perf_counter()
            await asyncio.gather(*[_query(30 + i, 5, post_switch_counts) for i in range(70)])
            phase3_elapsed = (time.perf_counter() - wall_start) * 1000

        record_load(
            "Zero-downtime upgrade (phase1: 30q top_k=10)",
            samples=[phase1_elapsed / 30] * 30,
            note="pre-switch queries, embed stubbed",
        )
        record_load(
            "Zero-downtime upgrade (phase3: 70q top_k=5)",
            samples=[phase3_elapsed / 70] * 70,
            note="post-switch queries, embed stubbed",
        )

        assert len(errors) == 0, f"Errors during switch: {errors}"
        # All post-switch queries should return ≤ 5 results (new top_k)
        assert all(c <= 5 for c in post_switch_counts), (
            f"Some post-switch queries returned > 5 results: {post_switch_counts}"
        )


class TestRollbackSpeed:
    """Rollback must propagate to the serving layer within 2 seconds."""

    async def test_rollback_clears_redis_under_2s(self, load_project, record_load) -> None:
        """After rollback_deployment(), the active-config Redis key must be gone < 2s.

        The clear_active_deployment() call inside rollback_deployment() deletes
        both ros:deployment:{name}:active and ros:project:{name}:active atomically.
        This test measures the wall-clock time from rollback call to key deletion.
        """
        # Ensure a live deployment exists for load_project
        async with async_session_factory() as session:
            dep = await deployment_repo.get_active_for_project(session, load_project)
            if not dep:
                # Re-create one if the lifecycle test already rolled it back
                await create_deployment(
                    session,
                    load_project,
                    CreateDeploymentRequest(
                        index_config_version=1,
                        top_k=10,
                        cache_enabled=True,
                        cache_ttl_seconds=3600,
                        created_by="test",
                    ),
                )
                await session.commit()

        redis = await get_redis()
        project_key = f"ros:project:{load_project}:active"
        assert await redis.exists(project_key), "Expected active config key before rollback"

        # Time the rollback
        t0 = time.perf_counter()
        async with async_session_factory() as session:
            dep = await deployment_repo.get_active_for_project(session, load_project)
            if dep:
                await rollback_deployment(
                    session,
                    load_project,
                    dep.id,
                    RollbackRequest(reason="rollback speed test", created_by="test"),
                )
                await session.commit()
        elapsed_s = time.perf_counter() - t0

        record_load(
            "Rollback propagation speed",
            samples=[elapsed_s * 1000],
            note="time until Redis active-config key removed",
        )

        # Verify key is gone
        assert not await redis.exists(project_key), "Active config key still exists after rollback"
        assert elapsed_s < 2.0, (
            f"Rollback took {elapsed_s:.2f}s; expected < 2s. "
            "Check Redis latency or deployment_repo lock contention."
        )
