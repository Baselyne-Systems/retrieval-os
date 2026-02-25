"""FastAPI router for the Lineage domain."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.database import get_db
from retrieval_os.lineage import service
from retrieval_os.lineage.schemas import (
    ArtifactResponse,
    ArtifactWithEdgesResponse,
    CreateEdgeRequest,
    EdgeResponse,
    LineageGraphResponse,
    OrphansResponse,
    RegisterArtifactRequest,
)

router = APIRouter(prefix="/v1/lineage", tags=["lineage"])


@router.post("/artifacts", status_code=201, response_model=ArtifactResponse)
async def register_artifact(
    request: RegisterArtifactRequest,
    db: AsyncSession = Depends(get_db),
) -> ArtifactResponse:
    """Register a dataset snapshot, embedding artifact, or index artifact.

    Idempotent by storage_uri — re-registering the same URI returns the
    existing record. S3 URIs are verified for existence before registration.
    """
    return await service.register_artifact(db, request)


@router.get("/artifacts", response_model=dict)
async def list_artifacts(
    artifact_type: str | None = Query(None, description="Filter by artifact type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List artifacts, newest first. Optionally filter by type."""
    artifacts, total = await service.list_artifacts(
        db, artifact_type=artifact_type, limit=limit, offset=offset
    )
    return {"items": [a.model_dump() for a in artifacts], "total": total}


@router.get("/artifacts/{artifact_id}", response_model=ArtifactWithEdgesResponse)
async def get_artifact(
    artifact_id: str,
    db: AsyncSession = Depends(get_db),
) -> ArtifactWithEdgesResponse:
    """Get an artifact with its immediate parents and children."""
    return await service.get_artifact(db, artifact_id)


@router.post("/edges", status_code=201, response_model=EdgeResponse)
async def create_edge(
    request: CreateEdgeRequest,
    db: AsyncSession = Depends(get_db),
) -> EdgeResponse:
    """Add a directed edge between two artifacts.

    Idempotent — if the edge already exists, the existing record is returned.
    Returns HTTP 400 if the edge would create a cycle.
    """
    return await service.create_edge(db, request)


@router.get(
    "/artifacts/{artifact_id}/ancestors",
    response_model=list[ArtifactResponse],
)
async def get_ancestors(
    artifact_id: str,
    max_depth: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> list[ArtifactResponse]:
    """Return all ancestor artifacts, ordered by ascending depth from the start artifact."""
    return await service.get_ancestors(db, artifact_id, max_depth=max_depth)


@router.get(
    "/artifacts/{artifact_id}/descendants",
    response_model=list[ArtifactResponse],
)
async def get_descendants(
    artifact_id: str,
    max_depth: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> list[ArtifactResponse]:
    """Return all descendant artifacts, ordered by ascending depth from the start artifact."""
    return await service.get_descendants(db, artifact_id, max_depth=max_depth)


@router.get("/plans/{plan_name}/graph", response_model=LineageGraphResponse)
async def get_plan_graph(
    plan_name: str,
    db: AsyncSession = Depends(get_db),
) -> LineageGraphResponse:
    """Return the full lineage graph (artifacts + edges) connected to a plan."""
    return await service.get_plan_lineage_graph(db, plan_name)


@router.get("/orphans", response_model=OrphansResponse)
async def get_orphans(
    db: AsyncSession = Depends(get_db),
) -> OrphansResponse:
    """Return artifacts with no downstream artifacts (leaf nodes not reachable
    from any active deployment).

    Use this to identify stale datasets or embedding artifacts that can be
    safely archived or deleted.
    """
    return await service.get_orphans(db)
