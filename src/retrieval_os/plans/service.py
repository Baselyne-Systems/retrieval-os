"""Business logic for the Plans domain."""

from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core import metrics
from retrieval_os.core.exceptions import (
    AppValidationError,
    ConflictError,
    DuplicateConfigError,
    PlanNotFoundError,
)
from retrieval_os.core.ids import uuid7
from retrieval_os.core.schemas.pagination import CursorPage
from retrieval_os.plans.models import PlanVersion, RetrievalPlan
from retrieval_os.plans.repository import plan_repo
from retrieval_os.plans.schemas import (
    ClonePlanRequest,
    CreatePlanRequest,
    CreateVersionRequest,
    PlanResponse,
    PlanVersionResponse,
    decode_cursor,
    encode_cursor,
)
from retrieval_os.plans.validators import compute_config_hash, validate_plan_config


async def create_plan(
    session: AsyncSession,
    request: CreatePlanRequest,
) -> PlanResponse:
    # 1. Name uniqueness
    if await plan_repo.get_by_name(session, request.name):
        raise ConflictError(f"A plan named '{request.name}' already exists")

    # 2. Validate config semantics
    config_dict = request.config.model_dump()
    validate_plan_config(config_dict)

    # 3. Config hash
    config_hash = compute_config_hash(config_dict)

    now = datetime.now(UTC)

    # 4. Persist plan
    plan = RetrievalPlan(
        id=uuid7(),
        name=request.name,
        description=request.description,
        created_at=now,
        updated_at=now,
        created_by=request.created_by,
    )
    session.add(plan)
    await session.flush()

    # 5. Persist version 1
    version = PlanVersion(
        id=uuid7(),
        plan_id=plan.id,
        version=1,
        is_current=True,
        config_hash=config_hash,
        created_at=now,
        created_by=request.created_by,
        **config_dict,
    )
    session.add(version)
    await session.flush()
    await session.refresh(plan)

    metrics.plans_total.inc()
    metrics.plan_versions_total.labels(plan_name=plan.name).inc()

    return PlanResponse.model_validate(plan)


async def get_plan(session: AsyncSession, name: str) -> PlanResponse:
    plan = await plan_repo.get_by_name(session, name)
    if not plan:
        raise PlanNotFoundError(f"Plan '{name}' not found")
    return PlanResponse.model_validate(plan)


async def list_plans(
    session: AsyncSession,
    cursor: str | None = None,
    limit: int = 20,
    include_archived: bool = False,
) -> CursorPage[PlanResponse]:
    offset = decode_cursor(cursor) if cursor else 0
    plans, total = await plan_repo.list_plans(
        session, offset=offset, limit=limit, include_archived=include_archived
    )
    next_offset = offset + len(plans)
    has_more = next_offset < total
    return CursorPage(
        items=[PlanResponse.model_validate(p) for p in plans],
        total=total,
        cursor=encode_cursor(next_offset) if has_more else None,
        has_more=has_more,
    )


async def create_version(
    session: AsyncSession,
    plan_name: str,
    request: CreateVersionRequest,
) -> PlanVersionResponse:
    # 1. Plan must exist and not be archived
    plan = await plan_repo.get_by_name(session, plan_name)
    if not plan:
        raise PlanNotFoundError(f"Plan '{plan_name}' not found")
    if plan.is_archived:
        raise ConflictError(f"Plan '{plan_name}' is archived and cannot be versioned")

    # 2. Validate
    config_dict = request.config.model_dump()
    validate_plan_config(config_dict)

    # 3. Dedup by config hash (same retrieval behaviour = no-op)
    config_hash = compute_config_hash(config_dict)
    existing = await plan_repo.get_version_by_config_hash(session, plan.id, config_hash)
    if existing:
        raise DuplicateConfigError(
            f"Version {existing.version} of '{plan_name}' already has this exact config",
            detail={"existing_version": existing.version, "config_hash": config_hash},
        )

    # 4. Serialised version number (locks plan row)
    next_num = await plan_repo.get_next_version_number(session, plan)

    # 5. Demote previous current version
    await plan_repo.unset_current_version(session, plan.id)

    # 6. Create new version
    now = datetime.now(UTC)
    version = PlanVersion(
        id=uuid7(),
        plan_id=plan.id,
        version=next_num,
        is_current=True,
        config_hash=config_hash,
        created_at=now,
        created_by=request.created_by,
        **config_dict,
    )
    try:
        version = await plan_repo.create_version(session, version)
    except IntegrityError:
        # Race condition: another concurrent request won; surface as duplicate
        raise DuplicateConfigError(
            "Concurrent version creation conflict — please retry",
            detail={"config_hash": config_hash},
        )

    metrics.plan_versions_total.labels(plan_name=plan.name).inc()

    return PlanVersionResponse.model_validate(version)


async def list_versions(
    session: AsyncSession, plan_name: str
) -> list[PlanVersionResponse]:
    plan = await plan_repo.get_by_name(session, plan_name)
    if not plan:
        raise PlanNotFoundError(f"Plan '{plan_name}' not found")
    versions = await plan_repo.list_versions(session, plan.id)
    return [PlanVersionResponse.model_validate(v) for v in versions]


async def get_version(
    session: AsyncSession, plan_name: str, version_num: int
) -> PlanVersionResponse:
    plan = await plan_repo.get_by_name(session, plan_name)
    if not plan:
        raise PlanNotFoundError(f"Plan '{plan_name}' not found")
    version = await plan_repo.get_version(session, plan.id, version_num)
    if not version:
        from retrieval_os.core.exceptions import PlanVersionNotFoundError
        raise PlanVersionNotFoundError(
            f"Version {version_num} of plan '{plan_name}' not found"
        )
    return PlanVersionResponse.model_validate(version)


async def clone_plan(
    session: AsyncSession,
    source_name: str,
    request: ClonePlanRequest,
) -> PlanResponse:
    # 1. Source must exist
    source = await plan_repo.get_by_name(session, source_name)
    if not source:
        raise PlanNotFoundError(f"Plan '{source_name}' not found")

    source_version = source.current_version
    if not source_version:
        raise AppValidationError(f"Plan '{source_name}' has no current version to clone")

    # 2. New name must be free
    if await plan_repo.get_by_name(session, request.new_name):
        raise ConflictError(f"A plan named '{request.new_name}' already exists")

    now = datetime.now(UTC)
    new_plan = RetrievalPlan(
        id=uuid7(),
        name=request.new_name,
        description=request.description or source.description,
        created_at=now,
        updated_at=now,
        created_by=request.created_by,
    )
    session.add(new_plan)
    await session.flush()

    new_version = PlanVersion(
        id=uuid7(),
        plan_id=new_plan.id,
        version=1,
        is_current=True,
        config_hash=source_version.config_hash,
        created_at=now,
        created_by=request.created_by,
        change_comment=f"Cloned from {source_name} v{source_version.version}",
        embedding_provider=source_version.embedding_provider,
        embedding_model=source_version.embedding_model,
        embedding_dimensions=source_version.embedding_dimensions,
        modalities=source_version.modalities,
        embedding_batch_size=source_version.embedding_batch_size,
        embedding_normalize=source_version.embedding_normalize,
        index_backend=source_version.index_backend,
        index_collection=source_version.index_collection,
        distance_metric=source_version.distance_metric,
        quantization=source_version.quantization,
        top_k=source_version.top_k,
        rerank_top_k=source_version.rerank_top_k,
        reranker=source_version.reranker,
        hybrid_alpha=source_version.hybrid_alpha,
        metadata_filters=source_version.metadata_filters,
        tenant_isolation_field=source_version.tenant_isolation_field,
        cache_enabled=source_version.cache_enabled,
        cache_ttl_seconds=source_version.cache_ttl_seconds,
        max_tokens_per_query=source_version.max_tokens_per_query,
    )
    session.add(new_version)
    await session.flush()
    await session.refresh(new_plan)

    metrics.plans_total.inc()
    metrics.plan_versions_total.labels(plan_name=new_plan.name).inc()

    return PlanResponse.model_validate(new_plan)


async def archive_plan(session: AsyncSession, name: str) -> None:
    plan = await plan_repo.get_by_name(session, name)
    if not plan:
        raise PlanNotFoundError(f"Plan '{name}' not found")
    await plan_repo.archive(session, plan.id)
