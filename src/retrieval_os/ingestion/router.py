"""REST endpoints for the Document Ingestion domain.

Routes
------
POST   /v1/projects/{project}/ingest            — submit a new ingestion job
GET    /v1/projects/{project}/ingest            — list jobs for a project
GET    /v1/projects/{project}/ingest/{job_id}   — get a single job
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.database import get_db
from retrieval_os.ingestion.repository import ingestion_repo
from retrieval_os.ingestion.schemas import (
    IngestionJobListResponse,
    IngestionJobResponse,
    IngestRequest,
)
from retrieval_os.ingestion.service import create_ingestion_job

router = APIRouter(prefix="/v1/projects/{project_name}/ingest", tags=["ingestion"])


@router.post("", response_model=IngestionJobResponse, status_code=202)
async def submit_ingestion_job(
    project_name: str,
    request: IngestRequest,
    session: AsyncSession = Depends(get_db),
) -> IngestionJobResponse:
    """Submit a batch ingestion job. Returns immediately with job status QUEUED."""
    job = await create_ingestion_job(session, project_name, request)
    return IngestionJobResponse.model_validate(job)


@router.get("", response_model=IngestionJobListResponse)
async def list_ingestion_jobs(
    project_name: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
) -> IngestionJobListResponse:
    items, total = await ingestion_repo.list_for_project(
        session, project_name, offset=offset, limit=limit
    )
    return IngestionJobListResponse(
        items=[IngestionJobResponse.model_validate(i) for i in items],
        total=total,
    )


@router.get("/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_job(
    project_name: str,
    job_id: str,
    session: AsyncSession = Depends(get_db),
) -> IngestionJobResponse:
    job = await ingestion_repo.get(session, job_id)
    if job is None or job.project_name != project_name:
        raise HTTPException(
            status_code=404,
            detail=f"Ingestion job '{job_id}' not found for project '{project_name}'",
        )
    return IngestionJobResponse.model_validate(job)
