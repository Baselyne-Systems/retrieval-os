"""Shared fixtures and helpers for e2e tests.

E2e tests exercise real infrastructure:
  - PostgreSQL  (default localhost:5432) — run `make infra && make migrate`
  - Redis       (default localhost:6379)

All tests in this directory are automatically skipped when infrastructure
is unreachable so that CI can gate the e2e suite separately from unit/integration.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from retrieval_os.core.database import async_session_factory, check_db_connection
from retrieval_os.core.ids import uuid7
from retrieval_os.core.redis_client import check_redis_connection, get_redis
from retrieval_os.deployments.schemas import CreateDeploymentRequest
from retrieval_os.deployments.service import create_deployment
from retrieval_os.plans.schemas import CreateProjectRequest, IndexConfigInput
from retrieval_os.plans.service import create_project

# ── Infrastructure availability guard ─────────────────────────────────────────


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring live infrastructure")


@pytest.fixture(autouse=True)
async def require_infra() -> None:
    """Skip every test in this directory if Postgres or Redis is unreachable.

    Keeps the test suite green in environments where infrastructure isn't
    available (e.g., lightweight CI steps that only run unit tests).
    Run `make infra && make migrate` to enable e2e tests.
    """
    if not await check_db_connection():
        pytest.skip("Postgres not reachable — run `make infra && make migrate`")
    if not await check_redis_connection():
        pytest.skip("Redis not reachable — run `make infra`")


# ── Per-test project name + cleanup ───────────────────────────────────────────


@pytest.fixture
async def project_name() -> str:
    """Yields a unique project name and cleans up all related DB rows and Redis
    keys after the test completes (pass or fail).
    """
    name = f"e2e-{uuid.uuid4().hex[:8]}"
    yield name

    # Best-effort cleanup so test runs don't accumulate stale data.
    try:
        redis = await get_redis()
        await redis.delete(
            f"ros:project:{name}:active",
            f"ros:deployment:{name}:active",
        )
    except Exception:
        pass

    try:
        async with async_session_factory() as session:
            await session.execute(
                text("DELETE FROM ingestion_jobs WHERE project_name = :n"), {"n": name}
            )
            await session.execute(
                text("DELETE FROM eval_jobs WHERE project_name = :n"), {"n": name}
            )
            await session.execute(
                text("DELETE FROM deployments WHERE project_name = :n"), {"n": name}
            )
            await session.execute(
                text(
                    "DELETE FROM index_configs "
                    "WHERE project_id = (SELECT id FROM projects WHERE name = :n)"
                ),
                {"n": name},
            )
            await session.execute(text("DELETE FROM projects WHERE name = :n"), {"n": name})
            await session.commit()
    except Exception:
        pass


# ── Real DB session ────────────────────────────────────────────────────────────


@pytest.fixture
async def db():
    """Real async SQLAlchemy session.

    Does NOT auto-commit — tests must call ``await db.commit()`` when they
    need changes visible to other sessions (e.g., Redis cache warming).
    Rolls back any uncommitted changes on teardown.
    """
    async with async_session_factory() as session:
        yield session
        await session.rollback()


# ── Data-setup helpers (used by test files) ────────────────────────────────────


def _base_index_config_input(*, collection: str) -> IndexConfigInput:
    return IndexConfigInput(
        embedding_provider="sentence_transformers",
        embedding_model="BAAI/bge-m3",
        embedding_dimensions=1024,
        index_collection=collection,
        index_backend="qdrant",
        distance_metric="cosine",
        change_comment="e2e test",
    )


async def setup_project_with_config(name: str) -> None:
    """Create a project + index config v1 and commit.

    Uses a fresh session so callers can continue with their own sessions
    without session-sharing conflicts.
    """
    async with async_session_factory() as session:
        await create_project(
            session,
            CreateProjectRequest(
                name=name,
                description="e2e test project",
                config=_base_index_config_input(collection=f"{name}-v1"),
                created_by="e2e",
            ),
        )
        await session.commit()


async def setup_deployment(name: str, **kwargs: object) -> str:
    """Create an instant deployment for *name* and commit. Returns the deployment ID."""
    async with async_session_factory() as session:
        dep = await create_deployment(
            session,
            name,
            CreateDeploymentRequest(index_config_version=1, created_by="e2e", **kwargs),
        )
        await session.commit()
        return dep.id


async def insert_completed_eval_job(
    project_name: str,
    *,
    recall_at_5: float,
    total_queries: int = 100,
    failed_queries: int = 0,
) -> str:
    """Insert a COMPLETED eval job directly into the DB and commit. Returns job ID."""
    from retrieval_os.evaluations.models import EvalJob, EvalJobStatus

    job_id = str(uuid7())
    now = datetime.now(UTC)
    async with async_session_factory() as session:
        job = EvalJob(
            id=job_id,
            project_name=project_name,
            index_config_version=1,
            status=EvalJobStatus.COMPLETED.value,
            dataset_uri="inline://e2e-test",
            top_k=10,
            recall_at_5=recall_at_5,
            total_queries=total_queries,
            failed_queries=failed_queries,
            created_at=now,
            completed_at=now,
            created_by="e2e",
        )
        session.add(job)
        await session.flush()
        await session.commit()
    return job_id
