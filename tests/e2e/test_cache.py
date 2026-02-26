"""E2E: Serving config Redis cache behaviour

Production failure modes proven here:

1. Deployment warms Redis — when a deployment goes ACTIVE, the 16-field serving
   config must be written to Redis immediately so the first query never hits
   Postgres on the hot path.

2. Redis miss falls back and rewarms — when the Redis key is absent (cold start,
   eviction, manual flush), _load_project_config must load the config from
   Postgres and rewrite the Redis key so subsequent queries are fast.

3. Rollback clears Redis — rolling back a deployment must delete
   ros:project:{name}:active so queries immediately stop serving the rolled-back
   config instead of silently serving stale data for up to 5 minutes.
   (This test caught the bug in clear_active_deployment that only deleted
   the deployment marker key, not the serving config key.)

4. Redis hot-path latency — with the serving config in Redis, p99 latency of
   _load_project_config must be < 5 ms. At 10 000 QPS, this step contributes
   < 50 ms of total per-second CPU overhead.

5. Cold-start stampede resilience — 20 concurrent queries all hitting a cold
   Redis (key absent) must all resolve without error. The system provides
   availability over consistency: every query falls back to Postgres, each
   potentially rewarming the cache. No stampede-induced failures.
"""

from __future__ import annotations

import json
import statistics
import time

from retrieval_os.core.database import async_session_factory
from retrieval_os.core.redis_client import get_redis
from retrieval_os.deployments.schemas import RollbackRequest
from retrieval_os.deployments.service import rollback_deployment
from retrieval_os.serving.query_router import _load_project_config, _project_redis_key
from tests.e2e.conftest import setup_deployment, setup_project_with_config

_REQUIRED_CONFIG_FIELDS = {
    "project_name",
    "index_config_version",
    "embedding_provider",
    "embedding_model",
    "embedding_normalize",
    "embedding_batch_size",
    "index_backend",
    "index_collection",
    "distance_metric",
    "top_k",
    "reranker",
    "rerank_top_k",
    "metadata_filters",
    "cache_enabled",
    "cache_ttl_seconds",
    "hybrid_alpha",
}


# ── Redis serving config correctness ──────────────────────────────────────────


class TestServingConfigRedis:
    async def test_deployment_warms_redis_serving_config(self, project_name: str) -> None:
        """The moment a deployment goes ACTIVE, the 16-field serving config must
        be present in Redis at ros:project:{name}:active.

        If this key is absent, the first query after every deployment hits
        Postgres — negating the Redis-first design.
        """
        await setup_project_with_config(project_name)
        await setup_deployment(project_name)

        redis = await get_redis()
        raw = await redis.get(_project_redis_key(project_name))
        assert raw is not None, (
            f"ros:project:{project_name}:active not found in Redis after deployment"
        )

        config = json.loads(raw)
        missing = _REQUIRED_CONFIG_FIELDS - set(config.keys())
        assert not missing, f"Serving config is missing fields after deployment: {missing}"
        assert config["project_name"] == project_name
        assert config["top_k"] == 10
        assert config["cache_enabled"] is True

    async def test_redis_miss_falls_back_to_postgres_and_rewarms_cache(
        self, project_name: str
    ) -> None:
        """When ros:project:{name}:active is absent (cold start / TTL expiry),
        _load_project_config must load the config from Postgres and rewrite
        the Redis key so the next query hits Redis.

        This proves the Redis-first + Postgres-fallback invariant end-to-end.
        """
        await setup_project_with_config(project_name)
        await setup_deployment(project_name)

        redis = await get_redis()
        key = _project_redis_key(project_name)

        # Simulate cold Redis
        await redis.delete(key)
        assert not await redis.exists(key)

        # Load config — must hit Postgres and rewarm Redis
        async with async_session_factory() as session:
            config = await _load_project_config(project_name, session)

        assert config["project_name"] == project_name
        missing = _REQUIRED_CONFIG_FIELDS - set(config.keys())
        assert not missing, f"Serving config missing fields after cold-start load: {missing}"

        # Verify cache was rewarmed
        rewarmed = await redis.exists(key)
        assert rewarmed, (
            "_load_project_config did not rewarm Redis after a cache miss — "
            "subsequent queries will keep hitting Postgres"
        )

    async def test_rollback_clears_redis_serving_config(self, project_name: str) -> None:
        """Rolling back a deployment must delete ros:project:{name}:active.

        Before this fix, clear_active_deployment only deleted the deployment
        marker key (ros:deployment:{name}:active) but left the serving config key
        intact. Queries after rollback would silently serve the rolled-back
        deployment's config for up to 5 minutes (the Redis TTL).
        """
        await setup_project_with_config(project_name)
        dep_id = await setup_deployment(project_name)

        redis = await get_redis()
        key = _project_redis_key(project_name)

        assert await redis.exists(key), "Serving config should exist in Redis after deployment"

        async with async_session_factory() as session:
            await rollback_deployment(
                session,
                project_name,
                dep_id,
                RollbackRequest(reason="e2e rollback test", created_by="e2e"),
            )
            await session.commit()

        assert not await redis.exists(key), (
            "ros:project:{name}:active was NOT cleared on rollback. "
            "Stale serving config will be served until TTL expires "
            "(up to 5 min of queries using rolled-back config)."
        )


# ── Performance characteristics ────────────────────────────────────────────────


class TestServingConfigLatency:
    async def test_redis_hot_path_p99_latency_under_5ms(self, project_name: str) -> None:
        """With the serving config in Redis, p99 latency of _load_project_config
        must be < 5 ms.

        At 10 000 QPS, the config-read step can contribute at most
        50 ms of total CPU time per second (0.5% overhead budget).
        Network round-trip to a local Redis is typically 0.1–0.5 ms;
        5 ms p99 leaves a comfortable margin for JSON parse overhead.
        """
        await setup_project_with_config(project_name)
        await setup_deployment(project_name)

        # Verify Redis is warm before measuring
        redis = await get_redis()
        assert await redis.exists(_project_redis_key(project_name))

        n = 100
        latencies_ms: list[float] = []
        async with async_session_factory() as session:
            for _ in range(n):
                t0 = time.perf_counter()
                await _load_project_config(project_name, session)
                latencies_ms.append((time.perf_counter() - t0) * 1000)

        p99 = statistics.quantiles(latencies_ms, n=100)[98]  # 99th percentile
        assert p99 < 5.0, (
            f"Redis hot-path p99 latency is {p99:.2f} ms; must be < 5 ms. "
            f"Median: {statistics.median(latencies_ms):.2f} ms, "
            f"Max: {max(latencies_ms):.2f} ms"
        )

    async def test_cold_start_stampede_20_concurrent_queries_all_resolve(
        self, project_name: str
    ) -> None:
        """20 queries hitting a cold Redis (no serving config cached) must all
        resolve without raising exceptions.

        Without a stampede-prevention mechanism (e.g., Redis locks or singleflight),
        every coroutine falls back to Postgres independently. The critical property
        is that all 20 queries succeed — if Postgres falls over under this load,
        the system violates its availability guarantee.
        """
        import asyncio

        await setup_project_with_config(project_name)
        await setup_deployment(project_name)

        # Force cold Redis
        redis = await get_redis()
        await redis.delete(_project_redis_key(project_name))

        errors: list[Exception] = []

        async def single_query() -> None:
            try:
                async with async_session_factory() as session:
                    await _load_project_config(project_name, session)
            except Exception as exc:
                errors.append(exc)

        await asyncio.gather(*[single_query() for _ in range(20)])

        assert not errors, (
            f"{len(errors)}/20 concurrent cold-start queries failed: "
            f"{[type(e).__name__ for e in errors]}"
        )
