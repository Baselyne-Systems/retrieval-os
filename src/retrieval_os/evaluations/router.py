"""FastAPI router for the Evaluation Engine domain."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.database import get_db
from retrieval_os.evaluations import service
from retrieval_os.evaluations.schemas import (
    EvalJobListResponse,
    EvalJobResponse,
    QueueEvalJobRequest,
)

router = APIRouter(prefix="/v1/eval", tags=["evaluation"])


@router.post("/jobs", status_code=201, response_model=EvalJobResponse)
async def queue_eval_job(
    request: QueueEvalJobRequest,
    db: AsyncSession = Depends(get_db),
) -> EvalJobResponse:
    """Queue a new evaluation job for a plan version.

    The job will be picked up by the background eval_job_runner. The plan
    version must exist. The dataset must be a JSONL file at the given S3 URI.
    """
    return await service.queue_eval_job(db, request)


@router.get("/jobs/{job_id}", response_model=EvalJobResponse)
async def get_eval_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> EvalJobResponse:
    """Get the status and results of an eval job."""
    return await service.get_eval_job(db, job_id)


@router.get("/jobs", response_model=EvalJobListResponse)
async def list_eval_jobs(
    plan_name: str | None = Query(None, description="Filter by plan name"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> EvalJobListResponse:
    """List eval jobs, newest first. Optionally filter by plan name."""
    return await service.list_eval_jobs(
        db, plan_name=plan_name, limit=limit, offset=offset
    )


@router.get(
    "/plans/{plan_name}/jobs",
    response_model=EvalJobListResponse,
)
async def list_plan_eval_jobs(
    plan_name: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> EvalJobListResponse:
    """List eval jobs for a specific plan, newest first."""
    return await service.list_eval_jobs(
        db, plan_name=plan_name, limit=limit, offset=offset
    )
