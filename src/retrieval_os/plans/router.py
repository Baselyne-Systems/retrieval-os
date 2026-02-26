"""FastAPI router for the Projects domain."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.database import get_db
from retrieval_os.core.schemas.pagination import CursorPage
from retrieval_os.plans import service
from retrieval_os.plans.schemas import (
    CloneProjectRequest,
    CreateIndexConfigRequest,
    CreateProjectRequest,
    IndexConfigResponse,
    ProjectResponse,
)

router = APIRouter(prefix="/v1/projects", tags=["projects"])


@router.post("", status_code=201, response_model=ProjectResponse)
async def create_project(
    request: CreateProjectRequest,
    db: AsyncSession = Depends(get_db),
) -> ProjectResponse:
    """Create a new project with its first index config."""
    return await service.create_project(db, request)


@router.get("", response_model=CursorPage[ProjectResponse])
async def list_projects(
    cursor: str | None = Query(None, description="Pagination cursor from previous response"),
    limit: int = Query(20, ge=1, le=100),
    include_archived: bool = Query(False),
    db: AsyncSession = Depends(get_db),
) -> CursorPage[ProjectResponse]:
    """List projects, newest first."""
    return await service.list_projects(
        db, cursor=cursor, limit=limit, include_archived=include_archived
    )


@router.get("/{name}", response_model=ProjectResponse)
async def get_project(
    name: str,
    db: AsyncSession = Depends(get_db),
) -> ProjectResponse:
    """Get a project and its current index config."""
    return await service.get_project(db, name)


@router.post("/{name}/index-configs", status_code=201, response_model=IndexConfigResponse)
async def create_index_config(
    name: str,
    request: CreateIndexConfigRequest,
    db: AsyncSession = Depends(get_db),
) -> IndexConfigResponse:
    """
    Create a new immutable index config for a project.
    Returns HTTP 409 if the config is identical to an existing one.
    """
    return await service.create_index_config(db, name, request)


@router.get("/{name}/index-configs", response_model=list[IndexConfigResponse])
async def list_index_configs(
    name: str,
    db: AsyncSession = Depends(get_db),
) -> list[IndexConfigResponse]:
    """List all index configs for a project, oldest first."""
    return await service.list_index_configs(db, name)


@router.get("/{name}/index-configs/{version_num}", response_model=IndexConfigResponse)
async def get_index_config(
    name: str,
    version_num: int,
    db: AsyncSession = Depends(get_db),
) -> IndexConfigResponse:
    """Get a specific index config version."""
    return await service.get_index_config(db, name, version_num)


@router.post("/{name}/clone", status_code=201, response_model=ProjectResponse)
async def clone_project(
    name: str,
    request: CloneProjectRequest,
    db: AsyncSession = Depends(get_db),
) -> ProjectResponse:
    """
    Clone a project's current index config as version 1 of a new project.
    The clone shares no state with the source after creation.
    """
    return await service.clone_project(db, name, request)


@router.delete("/{name}", status_code=204)
async def archive_project(
    name: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Soft-delete (archive) a project. Archived projects cannot receive new index configs."""
    await service.archive_project(db, name)
