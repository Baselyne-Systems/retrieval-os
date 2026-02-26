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
    IndexConfigNotFoundError,
    ProjectNotFoundError,
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
from retrieval_os.evaluations.repository import eval_repo
from retrieval_os.plans.repository import project_repo
from retrieval_os.webhooks.delivery import fire_webhook_event
from retrieval_os.webhooks.events import WebhookEvent

log = logging.getLogger(__name__)


def _build_serving_config(plan_name: str, deployment: Deployment, index_config: object) -> dict:
    """Build the Redis serving config dict by merging Deployment + IndexConfig."""
    return {
        "project_name": plan_name,
        "index_config_version": deployment.index_config_version,
        "embedding_provider": index_config.embedding_provider,  # type: ignore[attr-defined]
        "embedding_model": index_config.embedding_model,  # type: ignore[attr-defined]
        "embedding_normalize": index_config.embedding_normalize,  # type: ignore[attr-defined]
        "embedding_batch_size": index_config.embedding_batch_size,  # type: ignore[attr-defined]
        "index_backend": index_config.index_backend,  # type: ignore[attr-defined]
        "index_collection": index_config.index_collection,  # type: ignore[attr-defined]
        "distance_metric": index_config.distance_metric,  # type: ignore[attr-defined]
        "top_k": deployment.top_k,
        "reranker": deployment.reranker,
        "rerank_top_k": deployment.rerank_top_k,
        "metadata_filters": deployment.metadata_filters,
        "cache_enabled": deployment.cache_enabled,
        "cache_ttl_seconds": deployment.cache_ttl_seconds,
        "hybrid_alpha": deployment.hybrid_alpha,
    }


async def create_deployment(
    session: AsyncSession,
    plan_name: str,
    request: CreateDeploymentRequest,
) -> DeploymentResponse:
    # 1. Project must exist and not be archived
    project = await project_repo.get_by_name(session, plan_name)
    if not project:
        raise ProjectNotFoundError(f"Project '{plan_name}' not found")
    if project.is_archived:
        raise ConflictError(f"Project '{plan_name}' is archived")

    # 2. Requested index config version must exist
    index_config = await project_repo.get_index_config(
        session, project.id, request.index_config_version
    )
    if not index_config:
        raise IndexConfigNotFoundError(
            f"Index config version {request.index_config_version} "
            f"of project '{plan_name}' not found"
        )

    # 3. No other live deployment allowed
    existing = await deployment_repo.get_active_for_plan(session, plan_name)
    if existing:
        raise ConflictError(
            f"Project '{plan_name}' already has an active deployment "
            f"(id={existing.id}, status={existing.status}). "
            "Rollback the current deployment before creating a new one."
        )

    # 4. Validate gradual rollout params
    if (request.rollout_step_percent is None) != (request.rollout_step_interval_seconds is None):
        raise AppValidationError(
            "rollout_step_percent and rollout_step_interval_seconds "
            "must both be provided or both omitted"
        )

    now = datetime.now(UTC)
    is_gradual = request.rollout_step_percent is not None

    deployment = Deployment(
        id=str(uuid7()),
        plan_name=plan_name,
        project_id=project.id,
        index_config_id=index_config.id,
        index_config_version=request.index_config_version,
        top_k=request.top_k,
        rerank_top_k=request.rerank_top_k,
        reranker=request.reranker,
        hybrid_alpha=request.hybrid_alpha,
        metadata_filters=request.metadata_filters,
        tenant_isolation_field=request.tenant_isolation_field,
        cache_enabled=request.cache_enabled,
        cache_ttl_seconds=request.cache_ttl_seconds,
        max_tokens_per_query=request.max_tokens_per_query,
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
        deployment = await _activate(session, deployment, plan_name, now)

    plan_config = _build_serving_config(plan_name, deployment, index_config)
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
    now: datetime,
) -> Deployment:
    deployment.status = DeploymentStatus.ACTIVE.value
    deployment.traffic_weight = 1.0
    deployment.activated_at = now
    deployment.updated_at = now
    await session.flush()
    await session.refresh(deployment)
    await fire_webhook_event(
        WebhookEvent.DEPLOYMENT_STATUS_CHANGED,
        {
            "deployment_id": deployment.id,
            "plan_name": plan_name,
            "status": DeploymentStatus.ACTIVE.value,
        },
        session,
    )
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
            f"Deployment '{deployment_id}' does not belong to project '{plan_name}'"
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
    await fire_webhook_event(
        WebhookEvent.DEPLOYMENT_STATUS_CHANGED,
        {
            "deployment_id": deployment_id,
            "plan_name": plan_name,
            "status": DeploymentStatus.ROLLED_BACK.value,
            "reason": request.reason,
        },
        session,
    )

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
            f"Deployment '{deployment_id}' not found for project '{plan_name}'"
        )
    return DeploymentResponse.model_validate(deployment)


async def list_deployments(session: AsyncSession, plan_name: str) -> list[DeploymentResponse]:
    project = await project_repo.get_by_name(session, plan_name)
    if not project:
        raise ProjectNotFoundError(f"Project '{plan_name}' not found")
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
            await fire_webhook_event(
                WebhookEvent.DEPLOYMENT_STATUS_CHANGED,
                {
                    "deployment_id": deployment.id,
                    "plan_name": deployment.plan_name,
                    "status": DeploymentStatus.ACTIVE.value,
                },
                session,
            )
            metrics.rollout_duration_seconds.labels(plan_name=deployment.plan_name).observe(
                (now - deployment.created_at).total_seconds()
            )
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


async def _watchdog_rollback(
    session: AsyncSession,
    deployment: Deployment,
    reason: str,
) -> None:
    """Roll back a live deployment due to a guard-rail breach."""
    now = datetime.now(UTC)
    await deployment_repo.update_status(
        session,
        deployment.id,
        DeploymentStatus.ROLLED_BACK.value,
        traffic_weight=0.0,
        rolled_back_at=now,
        rollback_reason=reason,
        updated_at=now,
    )
    await clear_active_deployment(deployment.plan_name)
    await fire_webhook_event(
        WebhookEvent.DEPLOYMENT_STATUS_CHANGED,
        {
            "deployment_id": deployment.id,
            "plan_name": deployment.plan_name,
            "status": DeploymentStatus.ROLLED_BACK.value,
            "reason": reason,
            "triggered_by": "watchdog",
        },
        session,
    )
    metrics.rollback_events_total.labels(
        deployment_id=deployment.id,
        plan_name=deployment.plan_name,
        triggered_by="watchdog",
    ).inc()
    log.warning(
        "deployment.watchdog.rollback",
        extra={
            "deployment_id": deployment.id,
            "plan_name": deployment.plan_name,
            "reason": reason,
        },
    )


async def check_rollback_thresholds(session: AsyncSession) -> int:
    """Trigger automatic rollback for live deployments that breach guard rails.

    Returns the number of rollbacks triggered.
    """
    live = await deployment_repo.list_live(session)
    triggered = 0

    for deployment in live:
        has_recall = deployment.rollback_recall_threshold is not None
        has_error = deployment.rollback_error_rate_threshold is not None
        if not has_recall and not has_error:
            continue

        latest_eval = await eval_repo.get_latest_completed_for_plan(session, deployment.plan_name)
        if latest_eval is None:
            continue

        should_rollback = False
        reason = ""

        # ── Recall@5 guard rail ───────────────────────────────────────────────
        if has_recall and latest_eval.recall_at_5 is not None:
            if latest_eval.recall_at_5 < deployment.rollback_recall_threshold:  # type: ignore[operator]
                should_rollback = True
                reason = (
                    f"recall@5 {latest_eval.recall_at_5:.4f} "
                    f"< threshold {deployment.rollback_recall_threshold:.4f}"
                )

        # ── Error-rate guard rail ─────────────────────────────────────────────
        if not should_rollback and has_error:
            total = latest_eval.total_queries or 0
            failed = latest_eval.failed_queries or 0
            if total > 0:
                error_rate = failed / total
                if error_rate > deployment.rollback_error_rate_threshold:  # type: ignore[operator]
                    should_rollback = True
                    reason = (
                        f"error_rate {error_rate:.4f} "
                        f"> threshold {deployment.rollback_error_rate_threshold:.4f}"
                    )

        if should_rollback:
            await _watchdog_rollback(session, deployment, reason)
            triggered += 1

    return triggered
