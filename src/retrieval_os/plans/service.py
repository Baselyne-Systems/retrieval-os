"""Business logic for the Projects domain."""

from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core import metrics
from retrieval_os.core.exceptions import (
    AppValidationError,
    ConflictError,
    DuplicateConfigError,
    IndexConfigNotFoundError,
    ProjectNotFoundError,
)
from retrieval_os.core.ids import uuid7
from retrieval_os.core.schemas.pagination import CursorPage
from retrieval_os.plans.models import IndexConfig, Project
from retrieval_os.plans.repository import project_repo
from retrieval_os.plans.schemas import (
    CloneProjectRequest,
    CreateIndexConfigRequest,
    CreateProjectRequest,
    IndexConfigResponse,
    ProjectResponse,
    decode_cursor,
    encode_cursor,
)
from retrieval_os.plans.validators import compute_config_hash, validate_index_config


async def create_project(
    session: AsyncSession,
    request: CreateProjectRequest,
) -> ProjectResponse:
    # 1. Name uniqueness
    if await project_repo.get_by_name(session, request.name):
        raise ConflictError(f"A project named '{request.name}' already exists")

    # 2. Validate config semantics
    config_dict = request.config.model_dump()
    validate_index_config(config_dict)

    # 3. Config hash
    config_hash = compute_config_hash(config_dict)

    now = datetime.now(UTC)

    # 4. Persist project
    project = Project(
        id=uuid7(),
        name=request.name,
        description=request.description,
        created_at=now,
        updated_at=now,
        created_by=request.created_by,
    )
    session.add(project)
    await session.flush()

    # 5. Persist index config version 1
    index_config = IndexConfig(
        id=uuid7(),
        project_id=project.id,
        version=1,
        is_current=True,
        config_hash=config_hash,
        created_at=now,
        created_by=request.created_by,
        **config_dict,
    )
    session.add(index_config)
    await session.flush()
    await session.refresh(project)

    metrics.projects_total.inc()
    metrics.index_configs_total.labels(project_name=project.name).inc()

    return ProjectResponse.model_validate(project)


async def get_project(session: AsyncSession, name: str) -> ProjectResponse:
    project = await project_repo.get_by_name(session, name)
    if not project:
        raise ProjectNotFoundError(f"Project '{name}' not found")
    return ProjectResponse.model_validate(project)


async def list_projects(
    session: AsyncSession,
    cursor: str | None = None,
    limit: int = 20,
    include_archived: bool = False,
) -> CursorPage[ProjectResponse]:
    offset = decode_cursor(cursor) if cursor else 0
    projects, total = await project_repo.list_projects(
        session, offset=offset, limit=limit, include_archived=include_archived
    )
    next_offset = offset + len(projects)
    has_more = next_offset < total
    return CursorPage(
        items=[ProjectResponse.model_validate(p) for p in projects],
        total=total,
        cursor=encode_cursor(next_offset) if has_more else None,
        has_more=has_more,
    )


async def create_index_config(
    session: AsyncSession,
    project_name: str,
    request: CreateIndexConfigRequest,
) -> IndexConfigResponse:
    # 1. Project must exist and not be archived
    project = await project_repo.get_by_name(session, project_name)
    if not project:
        raise ProjectNotFoundError(f"Project '{project_name}' not found")
    if project.is_archived:
        raise ConflictError(f"Project '{project_name}' is archived and cannot be versioned")

    # 2. Validate
    config_dict = request.config.model_dump()
    validate_index_config(config_dict)

    # 3. Dedup by config hash
    config_hash = compute_config_hash(config_dict)
    existing = await project_repo.get_index_config_by_config_hash(session, project.id, config_hash)
    if existing:
        raise DuplicateConfigError(
            f"Index config {existing.version} of '{project_name}' already has this exact config",
            detail={"existing_version": existing.version, "config_hash": config_hash},
        )

    # 4. Serialised version number (locks project row)
    next_num = await project_repo.get_next_version_number(session, project)

    # 5. Demote previous current index config
    await project_repo.unset_current_index_config(session, project.id)

    # 6. Create new index config
    now = datetime.now(UTC)
    index_config = IndexConfig(
        id=uuid7(),
        project_id=project.id,
        version=next_num,
        is_current=True,
        config_hash=config_hash,
        created_at=now,
        created_by=request.created_by,
        **config_dict,
    )
    try:
        index_config = await project_repo.create_index_config(session, index_config)
    except IntegrityError:
        raise DuplicateConfigError(
            "Concurrent index config creation conflict — please retry",
            detail={"config_hash": config_hash},
        )

    metrics.index_configs_total.labels(project_name=project.name).inc()

    return IndexConfigResponse.model_validate(index_config)


async def list_index_configs(session: AsyncSession, project_name: str) -> list[IndexConfigResponse]:
    project = await project_repo.get_by_name(session, project_name)
    if not project:
        raise ProjectNotFoundError(f"Project '{project_name}' not found")
    configs = await project_repo.list_index_configs(session, project.id)
    return [IndexConfigResponse.model_validate(c) for c in configs]


async def get_index_config(
    session: AsyncSession, project_name: str, version_num: int
) -> IndexConfigResponse:
    project = await project_repo.get_by_name(session, project_name)
    if not project:
        raise ProjectNotFoundError(f"Project '{project_name}' not found")
    config = await project_repo.get_index_config(session, project.id, version_num)
    if not config:
        raise IndexConfigNotFoundError(
            f"Index config {version_num} of project '{project_name}' not found"
        )
    return IndexConfigResponse.model_validate(config)


async def clone_project(
    session: AsyncSession,
    source_name: str,
    request: CloneProjectRequest,
) -> ProjectResponse:
    # 1. Source must exist
    source = await project_repo.get_by_name(session, source_name)
    if not source:
        raise ProjectNotFoundError(f"Project '{source_name}' not found")

    source_config = source.current_index_config
    if not source_config:
        raise AppValidationError(f"Project '{source_name}' has no current index config to clone")

    # 2. New name must be free
    if await project_repo.get_by_name(session, request.new_name):
        raise ConflictError(f"A project named '{request.new_name}' already exists")

    now = datetime.now(UTC)
    new_project = Project(
        id=uuid7(),
        name=request.new_name,
        description=request.description or source.description,
        created_at=now,
        updated_at=now,
        created_by=request.created_by,
    )
    session.add(new_project)
    await session.flush()

    new_config = IndexConfig(
        id=uuid7(),
        project_id=new_project.id,
        version=1,
        is_current=True,
        config_hash=source_config.config_hash,
        created_at=now,
        created_by=request.created_by,
        change_comment=f"Cloned from {source_name} v{source_config.version}",
        embedding_provider=source_config.embedding_provider,
        embedding_model=source_config.embedding_model,
        embedding_dimensions=source_config.embedding_dimensions,
        modalities=source_config.modalities,
        embedding_batch_size=source_config.embedding_batch_size,
        embedding_normalize=source_config.embedding_normalize,
        index_backend=source_config.index_backend,
        index_collection=source_config.index_collection,
        distance_metric=source_config.distance_metric,
        quantization=source_config.quantization,
    )
    session.add(new_config)
    await session.flush()
    await session.refresh(new_project)

    metrics.projects_total.inc()
    metrics.index_configs_total.labels(project_name=new_project.name).inc()

    return ProjectResponse.model_validate(new_project)


async def archive_project(session: AsyncSession, name: str) -> None:
    project = await project_repo.get_by_name(session, name)
    if not project:
        raise ProjectNotFoundError(f"Project '{name}' not found")
    await project_repo.archive(session, project.id)
