"""Business logic for the Deployments domain.

State machine:
    PENDING  →  ROLLING_OUT (gradual)  →  ACTIVE
             ↘  ACTIVE (instant)
    ACTIVE   →  ROLLING_BACK  →  ROLLED_BACK
    *        →  FAILED (on unrecoverable error)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core import metrics
from retrieval_os.core.exceptions import (
    AppValidationError,
    ConflictError,
    DeploymentNotFoundError,
    DeploymentStateError,
    PlanNotFoundError,
    PlanVersionNotFoundError,
)
from retrieval_os.core.ids import uuid7
from retrieval_os.deployments.models import Deployment, DeploymentStatus
from retrieval_os.deployments.repository import deployment_repo
from retrieval_os.deployments.schemas import (
    CreateDeploymentRequest,
    DeploymentResponse,
    RollbackRequest,
)
from retrieval_os.deployments.traffic import clear_active_deployment, set_active_deployment
from retrieval_os.plans.repository import plan_repo

log = logging.getLogger(__name__)


def _plan_config_from_version(plan_name: str, version: object) -> dict:
    """Build the Redis plan config dict from a PlanVersion ORM object."""
    return {
        "plan_name": plan_name,
        "version": version.version,
        "embedding_provider": version.embedding_provider,
        "embedding_model": version.embedding_model,
        "embedding_normalize": version.embedding_normalize,
        "embedding_batch_size": version.embedding_batch_size,
        "index_backend": version.index_backend,
        "index_collection": version.index_collection,
        "distance_metric": version.distance_metric,
        "top_k": version.top_k,
        "reranker": version.reranker,
        "rerank_top_k": version.rerank_top_k,
        "metadata_filters": version.metadata_filters,
        "cache_enabled": version.cache_enabled,
        "cache_ttl_seconds": version.cache_ttl_seconds,
    }


async def create_deployment(
    session: AsyncSession,
    plan_name: str,
    request: CreateDeploymentRequest,
) -> DeploymentResponse:
    # 1. Plan must exist and not be archived
    plan = await plan_repo.get_by_name(session, plan_name)
    if not plan:
        raise PlanNotFoundError(f"Plan '{plan_name}' not found")
    if plan.is_archived:
        raise ConflictError(f"Plan '{plan_name}' is archived")

    # 2. Requested version must exist
    version = await plan_repo.get_version(session, plan.id, request.plan_version)
    if not version:
        raise PlanVersionNotFoundError(
            f"Version {request.plan_version} of plan '{plan_name}' not found"
        )

    # 3. No other live deployment allowed
    existing = await deployment_repo.get_active_for_plan(session, plan_name)
    if existing:
        raise ConflictError(
            f"Plan '{plan_name}' already has an active deployment "
            f"(id={existing.id}, status={existing.status}). "
            "Rollback the current deployment before creating a new one."
        )

    # 4. Validate gradual rollout params
    if (request.rollout_step_percent is None) != (
        request.rollout_step_interval_seconds is None
    ):
        raise AppValidationError(
            "rollout_step_percent and rollout_step_interval_seconds "
            "must both be provided or both omitted"
        )

    now = datetime.now(UTC)
    is_gradual = request.rollout_step_percent is not None

    deployment = Deployment(
        id=str(uuid7()),
        plan_name=plan_name,
        plan_version=request.plan_version,
        status=DeploymentStatus.ROLLING_OUT.value if is_gradual else DeploymentStatus.PENDING.value,
        traffic_weight=0.0,
        rollout_step_percent=request.rollout_step_percent,
        rollout_step_interval_seconds=request.rollout_step_interval_seconds,
        rollback_recall_threshold=request.rollback_recall_threshold,
        rollback_error_rate_threshold=request.rollback_error_rate_threshold,
        change_note=request.change_note,
        created_at=now,
        updated_at=now,
        created_by=request.created_by,
    )

    deployment = await deployment_repo.create(session, deployment)

    if not is_gradual:
        # Instant promotion to ACTIVE
        deployment = await _activate(session, deployment, plan_name, version, now)

    plan_config = _plan_config_from_version(plan_name, version)
    await set_active_deployment(plan_name, deployment.id, plan_config)

    metrics.deployment_status.labels(
        deployment_id=deployment.id,
        plan_name=plan_name,
        environment="default",
        status=deployment.status,
    ).set(1)
    metrics.deployment_traffic_weight.labels(
        deployment_id=deployment.id,
        plan_name=plan_name,
        environment="default",
    ).set(deployment.traffic_weight)

    return DeploymentResponse.model_validate(deployment)


async def _activate(
    session: AsyncSession,
    deployment: Deployment,
    plan_name: str,
    version: object,
    now: datetime,
) -> Deployment:
    deployment.status = DeploymentStatus.ACTIVE.value
    deployment.traffic_weight = 1.0
    deployment.activated_at = now
    deployment.updated_at = now
    await session.flush()
    await session.refresh(deployment)
    return deployment


async def rollback_deployment(
    session: AsyncSession,
    plan_name: str,
    deployment_id: str,
    request: RollbackRequest,
) -> DeploymentResponse:
    deployment = await deployment_repo.get_by_id(session, deployment_id)
    if not deployment:
        raise DeploymentNotFoundError(f"Deployment '{deployment_id}' not found")
    if deployment.plan_name != plan_name:
        raise DeploymentNotFoundError(
            f"Deployment '{deployment_id}' does not belong to plan '{plan_name}'"
        )
    if not deployment.is_live:
        raise DeploymentStateError(
            f"Deployment '{deployment_id}' is not live (status={deployment.status})"
        )

    now = datetime.now(UTC)
    await deployment_repo.update_status(
        session,
        deployment_id,
        DeploymentStatus.ROLLED_BACK.value,
        traffic_weight=0.0,
        rolled_back_at=now,
        rollback_reason=request.reason,
        updated_at=now,
    )
    await session.refresh(deployment)
    await clear_active_deployment(plan_name)

    metrics.rollback_events_total.labels(
        deployment_id=deployment_id,
        plan_name=plan_name,
        triggered_by="api",
    ).inc()

    log.info(
        "deployment.rolled_back",
        extra={
            "deployment_id": deployment_id,
            "plan_name": plan_name,
            "reason": request.reason,
        },
    )
    return DeploymentResponse.model_validate(deployment)


async def get_deployment(
    session: AsyncSession, plan_name: str, deployment_id: str
) -> DeploymentResponse:
    deployment = await deployment_repo.get_by_id(session, deployment_id)
    if not deployment or deployment.plan_name != plan_name:
        raise DeploymentNotFoundError(
            f"Deployment '{deployment_id}' not found for plan '{plan_name}'"
        )
    return DeploymentResponse.model_validate(deployment)


async def list_deployments(
    session: AsyncSession, plan_name: str
) -> list[DeploymentResponse]:
    plan = await plan_repo.get_by_name(session, plan_name)
    if not plan:
        raise PlanNotFoundError(f"Plan '{plan_name}' not found")
    deployments = await deployment_repo.list_for_plan(session, plan_name)
    return [DeploymentResponse.model_validate(d) for d in deployments]


# ── Rollout stepper (called by background loop) ───────────────────────────────


async def step_rolling_deployments(session: AsyncSession) -> int:
    """Advance all ROLLING_OUT deployments by one step.

    Returns the number of deployments advanced.
    """
    rolling = await deployment_repo.list_rolling_out(session)
    advanced = 0

    for deployment in rolling:
        step = deployment.rollout_step_percent or 10.0
        new_weight = min(1.0, deployment.traffic_weight + step / 100.0)
        now = datetime.now(UTC)

        if new_weight >= 1.0:
            # Fully ramped — promote to ACTIVE
            await deployment_repo.update_status(
                session,
                deployment.id,
                DeploymentStatus.ACTIVE.value,
                traffic_weight=1.0,
                activated_at=now,
                updated_at=now,
            )
            log.info(
                "deployment.activated",
                extra={"deployment_id": deployment.id, "plan_name": deployment.plan_name},
            )
            metrics.rollout_duration_seconds.labels(
                plan_name=deployment.plan_name
            ).observe((now - deployment.created_at).total_seconds())
        else:
            await deployment_repo.update_status(
                session,
                deployment.id,
                DeploymentStatus.ROLLING_OUT.value,
                traffic_weight=new_weight,
                updated_at=now,
            )

        metrics.deployment_traffic_weight.labels(
            deployment_id=deployment.id,
            plan_name=deployment.plan_name,
            environment="default",
        ).set(new_weight)
        advanced += 1

    return advanced


# ── Rollback watchdog (called by background loop) ─────────────────────────────


async def check_rollback_thresholds(session: AsyncSession) -> int:
    """Trigger automatic rollback for deployments that breach guard rails.

    Phase 6 (Evaluation) populates the metrics this watchdog reads.
    For now this is a no-op stub that returns 0 — the infrastructure is wired.

    Returns the number of rollbacks triggered.
    """
    # Phase 6 will populate eval metrics into Redis; watchdog reads them here.
    return 0
