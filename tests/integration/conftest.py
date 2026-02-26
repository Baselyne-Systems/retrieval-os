"""Shared fixtures for integration tests.

Integration tests differ from unit tests in two ways:
- They exercise the full HTTP stack (routing → middleware → service) via AsyncClient.
- They patch only external I/O boundaries (DB session, Redis, Qdrant, S3),
  leaving service-layer logic intact.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from retrieval_os.api.main import app
from retrieval_os.core.database import get_db
from retrieval_os.deployments.models import DeploymentStatus
from retrieval_os.deployments.schemas import DeploymentResponse
from retrieval_os.plans.schemas import PlanResponse, PlanVersionResponse

# ── Mock session ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_session() -> AsyncMock:
    """Minimal async DB session mock.  Service functions receive this instead
    of a real SQLAlchemy session — all awaited calls are no-ops by default."""
    session = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
async def int_client(mock_session: AsyncMock):
    """AsyncClient wired to the FastAPI app with get_db dependency overridden.

    Yields ``(client, mock_session)`` so tests can inspect what the session
    received without standing up a real database.
    """

    async def _override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, mock_session
    app.dependency_overrides.clear()


# ── Response object factories ─────────────────────────────────────────────────


def _ts() -> datetime:
    return datetime.now(UTC)


def make_version_response(
    *,
    version: int = 1,
    plan_name: str = "my-docs",
) -> PlanVersionResponse:
    return PlanVersionResponse(
        id="018e7a2b-4000-7000-b234-000000000001",
        plan_id="018e7a2b-3f4c-7000-a123-000000000001",
        version=version,
        is_current=True,
        embedding_provider="sentence_transformers",
        embedding_model="BAAI/bge-m3",
        embedding_dimensions=1024,
        modalities=["text"],
        embedding_batch_size=32,
        embedding_normalize=True,
        index_backend="qdrant",
        index_collection=f"{plan_name}_v{version}",
        distance_metric="cosine",
        quantization=None,
        top_k=10,
        rerank_top_k=None,
        reranker=None,
        hybrid_alpha=None,
        metadata_filters=None,
        tenant_isolation_field=None,
        cache_enabled=True,
        cache_ttl_seconds=3600,
        max_tokens_per_query=None,
        change_comment="initial",
        config_hash="a" * 64,
        created_at=_ts(),
        created_by="alice",
    )


def make_plan_response(
    *,
    name: str = "my-docs",
    is_archived: bool = False,
) -> PlanResponse:
    return PlanResponse(
        id="018e7a2b-3f4c-7000-a123-000000000001",
        name=name,
        description="Test plan",
        is_archived=is_archived,
        created_at=_ts(),
        updated_at=_ts(),
        created_by="alice",
        current_version=make_version_response(plan_name=name),
    )


def make_deployment_response(
    *,
    plan_name: str = "my-docs",
    plan_version: int = 1,
    status: str = DeploymentStatus.ACTIVE.value,
    dep_id: str = "dep-001",
    rollback_recall_threshold: float | None = None,
    rollback_error_rate_threshold: float | None = None,
    rollback_reason: str | None = None,
) -> DeploymentResponse:
    return DeploymentResponse(
        id=dep_id,
        plan_name=plan_name,
        plan_version=plan_version,
        status=status,
        traffic_weight=1.0 if status == DeploymentStatus.ACTIVE.value else 0.0,
        rollout_step_percent=None,
        rollout_step_interval_seconds=None,
        rollback_recall_threshold=rollback_recall_threshold,
        rollback_error_rate_threshold=rollback_error_rate_threshold,
        change_note="test deploy",
        created_at=_ts(),
        updated_at=_ts(),
        created_by="alice",
        activated_at=_ts() if status == DeploymentStatus.ACTIVE.value else None,
        rolled_back_at=_ts() if status == DeploymentStatus.ROLLED_BACK.value else None,
        rollback_reason=rollback_reason,
    )


def make_eval_ns(
    *,
    recall_at_5: float = 0.80,
    total_queries: int = 100,
    failed_queries: int = 0,
) -> SimpleNamespace:
    """Lightweight eval-job-like namespace for watchdog / eval cycle tests."""
    return SimpleNamespace(
        id="eval-001",
        plan_name="my-docs",
        plan_version=1,
        recall_at_5=recall_at_5,
        total_queries=total_queries,
        failed_queries=failed_queries,
        created_at=datetime.now(UTC),
    )
