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
from retrieval_os.plans.models import PlanVersion, RetrievalPlan
from retrieval_os.webhooks.delivery import fire_webhook_event
from retrieval_os.webhooks.events import WebhookEvent

log = logging.getLogger(__name__)

# Fractional drop that constitutes a regression (5%)
_REGRESSION_THRESHOLD = 0.05

# Metrics included in regression check
_REGRESSION_METRICS = ("recall_at_5", "mrr", "ndcg_at_5")


async def _load_plan_version(session: AsyncSession, plan_name: str, version: int) -> PlanVersion:
    """Load a PlanVersion by (plan_name, version_number)."""
    result = await session.execute(
        select(PlanVersion)
        .join(RetrievalPlan, PlanVersion.plan_id == RetrievalPlan.id)
        .where(RetrievalPlan.name == plan_name, PlanVersion.version == version)
    )
    pv = result.scalar_one_or_none()
    if pv is None:
        from retrieval_os.core.exceptions import PlanVersionNotFoundError

        raise PlanVersionNotFoundError(f"Plan '{plan_name}' version {version} not found")
    return pv


async def queue_eval_job(
    session: AsyncSession,
    request,  # QueueEvalJobRequest
) -> EvalJobResponse:
    """Create a new QUEUED eval job."""
    # Verify the plan version exists before queuing
    await _load_plan_version(session, request.plan_name, request.plan_version)

    job = EvalJob(
        id=str(uuid7()),
        plan_name=request.plan_name,
        plan_version=request.plan_version,
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
    plan_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> EvalJobListResponse:
    jobs, total = await eval_repo.list_jobs(
        session, plan_name=plan_name, limit=limit, offset=offset
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
    log.info("eval.job.started", extra={"job_id": job_id, "plan": job.plan_name})

    try:
        # Load plan config (new nested session not needed — same tx is fine for reads)
        pv = await _load_plan_version(session, job.plan_name, job.plan_version)

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
        results: EvalResults = await execute_eval_job(
            records,
            plan_name=job.plan_name,
            plan_version=job.plan_version,
            embedding_provider=pv.embedding_provider,
            embedding_model=pv.embedding_model,
            embedding_normalize=pv.embedding_normalize,
            embedding_batch_size=pv.embedding_batch_size,
            index_backend=pv.index_backend,
            index_collection=pv.index_collection,
            distance_metric=pv.distance_metric,
            top_k=job.top_k,
            reranker=pv.reranker,
            rerank_top_k=pv.rerank_top_k,
        )

        # Regression detection against previous completed job
        previous = await eval_repo.get_latest_completed_for_plan(
            session, job.plan_name, exclude_job_id=job_id
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
                        plan_name=job.plan_name,
                        metric_name=r["metric"],
                        severity="warning",
                    ).inc()

        regression_detected = len(regressions) > 0

        if regression_detected:
            await fire_webhook_event(
                WebhookEvent.EVAL_REGRESSION_DETECTED,
                {
                    "job_id": job_id,
                    "plan_name": job.plan_name,
                    "plan_version": job.plan_version,
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
        dep_id = f"{job.plan_name}-v{job.plan_version}"
        metrics.eval_recall_at_k.labels(plan_name=job.plan_name, deployment_id=dep_id, k="5").set(
            results.recall_at_5
        )
        metrics.eval_mrr.labels(plan_name=job.plan_name, deployment_id=dep_id).set(results.mrr)
        metrics.eval_ndcg_at_k.labels(plan_name=job.plan_name, deployment_id=dep_id, k="5").set(
            results.ndcg_at_5
        )

        elapsed = (datetime.now(UTC) - start).total_seconds()
        metrics.eval_job_duration_seconds.labels(plan_name=job.plan_name).observe(elapsed)

        log.info(
            "eval.job.completed",
            extra={
                "job_id": job_id,
                "plan": job.plan_name,
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
