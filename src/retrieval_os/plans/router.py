"""FastAPI router for the Plans domain."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.database import get_db
from retrieval_os.core.schemas.pagination import CursorPage
from retrieval_os.plans import service
from retrieval_os.plans.schemas import (
    ClonePlanRequest,
    CreatePlanRequest,
    CreateVersionRequest,
    PlanResponse,
    PlanVersionResponse,
)

router = APIRouter(prefix="/v1/plans", tags=["plans"])


@router.post("", status_code=201, response_model=PlanResponse)
async def create_plan(
    request: CreatePlanRequest,
    db: AsyncSession = Depends(get_db),
) -> PlanResponse:
    """Create a new retrieval plan with its first version."""
    return await service.create_plan(db, request)


@router.get("", response_model=CursorPage[PlanResponse])
async def list_plans(
    cursor: str | None = Query(None, description="Pagination cursor from previous response"),
    limit: int = Query(20, ge=1, le=100),
    include_archived: bool = Query(False),
    db: AsyncSession = Depends(get_db),
) -> CursorPage[PlanResponse]:
    """List retrieval plans, newest first."""
    return await service.list_plans(
        db, cursor=cursor, limit=limit, include_archived=include_archived
    )


@router.get("/{name}", response_model=PlanResponse)
async def get_plan(
    name: str,
    db: AsyncSession = Depends(get_db),
) -> PlanResponse:
    """Get a plan and its current version."""
    return await service.get_plan(db, name)


@router.post("/{name}/versions", status_code=201, response_model=PlanVersionResponse)
async def create_version(
    name: str,
    request: CreateVersionRequest,
    db: AsyncSession = Depends(get_db),
) -> PlanVersionResponse:
    """
    Create a new immutable version of a plan.
    Returns HTTP 409 if the config is identical to an existing version.
    """
    return await service.create_version(db, name, request)


@router.get("/{name}/versions", response_model=list[PlanVersionResponse])
async def list_versions(
    name: str,
    db: AsyncSession = Depends(get_db),
) -> list[PlanVersionResponse]:
    """List all versions of a plan, oldest first."""
    return await service.list_versions(db, name)


@router.get("/{name}/versions/{version_num}", response_model=PlanVersionResponse)
async def get_version(
    name: str,
    version_num: int,
    db: AsyncSession = Depends(get_db),
) -> PlanVersionResponse:
    """Get a specific version of a plan."""
    return await service.get_version(db, name, version_num)


@router.post("/{name}/clone", status_code=201, response_model=PlanResponse)
async def clone_plan(
    name: str,
    request: ClonePlanRequest,
    db: AsyncSession = Depends(get_db),
) -> PlanResponse:
    """
    Clone a plan's current version as version 1 of a new plan.
    The clone shares no state with the source after creation.
    """
    return await service.clone_plan(db, name, request)


@router.delete("/{name}", status_code=204)
async def archive_plan(
    name: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Soft-delete (archive) a plan. Archived plans cannot receive new versions."""
    await service.archive_plan(db, name)
