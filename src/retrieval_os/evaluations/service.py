"""Business logic for the Evaluation Engine domain."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core import metrics
from retrieval_os.core.exceptions import EvalJobNotFoundError
from retrieval_os.core.ids import uuid7
from retrieval_os.evaluations.metrics import check_regression
from retrieval_os.evaluations.models import EvalJob, EvalJobStatus
from retrieval_os.evaluations.repository import eval_repo
from retrieval_os.evaluations.runner import EvalResults, execute_eval_job, load_eval_dataset
from retrieval_os.evaluations.schemas import EvalJobListResponse, EvalJobResponse
from retrieval_os.plans.models import IndexConfig, Project
from retrieval_os.webhooks.delivery import fire_webhook_event
from retrieval_os.webhooks.events import WebhookEvent

log = logging.getLogger(__name__)

# Fractional drop that constitutes a regression (5%)
_REGRESSION_THRESHOLD = 0.05

# Metrics included in regression check
_REGRESSION_METRICS = ("recall_at_5", "mrr", "ndcg_at_5")


async def _load_index_config(session: AsyncSession, project_name: str, version: int) -> IndexConfig:
    """Load an IndexConfig by (project_name, version_number)."""
    result = await session.execute(
        select(IndexConfig)
        .join(Project, IndexConfig.project_id == Project.id)
        .where(Project.name == project_name, IndexConfig.version == version)
    )
    ic = result.scalar_one_or_none()
    if ic is None:
        from retrieval_os.core.exceptions import IndexConfigNotFoundError

        raise IndexConfigNotFoundError(
            f"Project '{project_name}' index config version {version} not found"
        )
    return ic


async def queue_eval_job(
    session: AsyncSession,
    request,  # QueueEvalJobRequest
) -> EvalJobResponse:
    """Create a new QUEUED eval job."""
    # Verify the project index config version exists before queuing
    await _load_index_config(session, request.project_name, request.index_config_version)

    job = EvalJob(
        id=str(uuid7()),
        project_name=request.project_name,
        index_config_version=request.index_config_version,
        status=EvalJobStatus.QUEUED,
        dataset_uri=request.dataset_uri,
        top_k=request.top_k,
        created_at=datetime.now(UTC),
        created_by=request.created_by,
    )
    job = await eval_repo.create_job(session, job)
    return EvalJobResponse.model_validate(job)


async def get_eval_job(session: AsyncSession, job_id: str) -> EvalJobResponse:
    job = await eval_repo.get_job(session, job_id)
    if not job:
        raise EvalJobNotFoundError(f"Eval job '{job_id}' not found")
    return EvalJobResponse.model_validate(job)


async def list_eval_jobs(
    session: AsyncSession,
    project_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> EvalJobListResponse:
    jobs, total = await eval_repo.list_jobs(
        session, project_name=project_name, limit=limit, offset=offset
    )
    return EvalJobListResponse(
        items=[EvalJobResponse.model_validate(j) for j in jobs],
        total=total,
    )


async def process_next_eval_job(session: AsyncSession) -> str | None:
    """Claim and execute the next QUEUED eval job.

    Uses SELECT FOR UPDATE SKIP LOCKED so multiple runner instances never
    process the same job. Returns the job ID if a job was processed, else None.

    The session is committed by the caller (background loop) after this returns.
    """
    job = await eval_repo.claim_next_queued(session)
    if job is None:
        return None

    job_id = job.id
    start = datetime.now(UTC)
    log.info("eval.job.started", extra={"job_id": job_id, "project": job.project_name})

    try:
        # Load index config (new nested session not needed — same tx is fine for reads)
        ic = await _load_index_config(session, job.project_name, job.index_config_version)

        # Load dataset from S3
        records = await load_eval_dataset(job.dataset_uri)
        if not records:
            await eval_repo.fail_job(
                session,
                job_id,
                error_message="Eval dataset is empty or could not be parsed",
                total_queries=0,
                failed_queries=0,
            )
            return job_id

        # Execute retrieval for every query
        # reranker/rerank_top_k live on Deployment (search config); eval tests raw retrieval
        results: EvalResults = await execute_eval_job(
            records,
            project_name=job.project_name,
            index_config_version=job.index_config_version,
            embedding_provider=ic.embedding_provider,
            embedding_model=ic.embedding_model,
            embedding_normalize=ic.embedding_normalize,
            embedding_batch_size=ic.embedding_batch_size,
            index_backend=ic.index_backend,
            index_collection=ic.index_collection,
            distance_metric=ic.distance_metric,
            top_k=job.top_k,
            reranker=None,
            rerank_top_k=None,
        )

        # Regression detection against previous completed job
        previous = await eval_repo.get_latest_completed_for_project(
            session, job.project_name, exclude_job_id=job_id
        )
        regressions: list[dict] = []
        if previous is not None:
            curr_metrics = {
                "recall_at_5": results.recall_at_5,
                "mrr": results.mrr,
                "ndcg_at_5": results.ndcg_at_5,
            }
            prev_metrics = {
                "recall_at_5": previous.recall_at_5 or 0.0,
                "mrr": previous.mrr or 0.0,
                "ndcg_at_5": previous.ndcg_at_5 or 0.0,
            }
            regressions = check_regression(
                curr_metrics, prev_metrics, threshold=_REGRESSION_THRESHOLD
            )
            if regressions:
                for r in regressions:
                    metrics.eval_regression_alerts_total.labels(
                        project_name=job.project_name,
                        metric_name=r["metric"],
                        severity="warning",
                    ).inc()

        regression_detected = len(regressions) > 0

        if regression_detected:
            await fire_webhook_event(
                WebhookEvent.EVAL_REGRESSION_DETECTED,
                {
                    "job_id": job_id,
                    "project_name": job.project_name,
                    "index_config_version": job.index_config_version,
                    "regressions": regressions,
                },
                session,
            )

        await eval_repo.complete_job(
            session,
            job_id,
            recall_at_1=results.recall_at_1,
            recall_at_3=results.recall_at_3,
            recall_at_5=results.recall_at_5,
            recall_at_10=results.recall_at_10,
            mrr=results.mrr,
            ndcg_at_5=results.ndcg_at_5,
            ndcg_at_10=results.ndcg_at_10,
            total_queries=results.total_queries,
            failed_queries=results.failed_queries,
            regression_detected=regression_detected,
            regression_detail=regressions,
        )

        # Prometheus gauges
        dep_id = f"{job.project_name}-v{job.index_config_version}"
        metrics.eval_recall_at_k.labels(
            project_name=job.project_name, deployment_id=dep_id, k="5"
        ).set(results.recall_at_5)
        metrics.eval_mrr.labels(project_name=job.project_name, deployment_id=dep_id).set(
            results.mrr
        )
        metrics.eval_ndcg_at_k.labels(
            project_name=job.project_name, deployment_id=dep_id, k="5"
        ).set(results.ndcg_at_5)

        elapsed = (datetime.now(UTC) - start).total_seconds()
        metrics.eval_job_duration_seconds.labels(project_name=job.project_name).observe(elapsed)

        log.info(
            "eval.job.completed",
            extra={
                "job_id": job_id,
                "project": job.project_name,
                "recall_at_5": round(results.recall_at_5, 4),
                "mrr": round(results.mrr, 4),
                "regression": regression_detected,
                "elapsed_s": round(elapsed, 1),
            },
        )

    except Exception as exc:
        log.exception("eval.job.failed", extra={"job_id": job_id})
        await eval_repo.fail_job(session, job_id, error_message=str(exc))

    return job_id
