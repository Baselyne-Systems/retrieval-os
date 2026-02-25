"""Business logic for the Lineage domain."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core import metrics
from retrieval_os.core.exceptions import (
    ArtifactNotFoundError,
    ArtifactStorageNotFoundError,
    LineageCycleError,
)
from retrieval_os.core.ids import uuid7
from retrieval_os.core.s3_client import object_exists
from retrieval_os.lineage.dag import would_create_cycle
from retrieval_os.lineage.models import LineageArtifact, LineageEdge
from retrieval_os.lineage.repository import lineage_repo
from retrieval_os.lineage.schemas import (
    ArtifactResponse,
    ArtifactWithEdgesResponse,
    CreateEdgeRequest,
    EdgeResponse,
    LineageGraphResponse,
    OrphansResponse,
    RegisterArtifactRequest,
)

log = logging.getLogger(__name__)

_S3_SCHEMES = ("s3://", "s3a://", "s3n://")


def _is_s3_uri(uri: str) -> bool:
    return any(uri.startswith(s) for s in _S3_SCHEMES)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key) from s3://bucket/key/path."""
    path = uri.split("://", 1)[1]
    bucket, _, key = path.partition("/")
    return bucket, key


async def _verify_s3_artifact(storage_uri: str) -> dict:
    """Fetch S3 object metadata. Raises ArtifactStorageNotFoundError if missing."""
    bucket, key = _parse_s3_uri(storage_uri)
    try:
        exists = await object_exists(key, bucket=bucket)
        if not exists:
            raise ArtifactStorageNotFoundError(
                f"S3 object not found: {storage_uri}",
                detail={"storage_uri": storage_uri},
            )
        from retrieval_os.core.s3_client import get_object_metadata

        meta = await get_object_metadata(key, bucket=bucket)
        return {
            "s3_size_bytes": meta.get("size_bytes"),
            "s3_etag": meta.get("etag", ""),
        }
    except ArtifactStorageNotFoundError:
        raise
    except Exception as exc:
        log.warning("lineage.s3_verify_failed", extra={"uri": storage_uri, "error": str(exc)})
        raise ArtifactStorageNotFoundError(
            f"Could not verify S3 object: {storage_uri}",
            detail={"storage_uri": storage_uri, "error": str(exc)},
        )


async def register_artifact(
    session: AsyncSession,
    request: RegisterArtifactRequest,
) -> ArtifactResponse:
    # 1. Dedup by storage_uri — idempotent registration
    existing = await lineage_repo.get_artifact_by_uri(session, request.storage_uri)
    if existing:
        return ArtifactResponse.model_validate(existing)

    # 2. Verify storage exists for S3 artifacts
    extra_metadata: dict = {}
    if _is_s3_uri(request.storage_uri):
        extra_metadata = await _verify_s3_artifact(request.storage_uri)

    now = datetime.now(UTC)
    artifact = LineageArtifact(
        id=str(uuid7()),
        artifact_type=request.artifact_type.value,
        name=request.name,
        version=request.version,
        storage_uri=request.storage_uri,
        content_hash=request.content_hash,
        artifact_metadata={**(request.metadata or {}), **extra_metadata} or None,
        created_at=now,
        created_by=request.created_by,
    )
    artifact = await lineage_repo.create_artifact(session, artifact)

    metrics.lineage_artifacts_total.labels(artifact_type=request.artifact_type.value).inc()

    return ArtifactResponse.model_validate(artifact)


async def get_artifact(session: AsyncSession, artifact_id: str) -> ArtifactWithEdgesResponse:
    artifact = await lineage_repo.get_artifact(session, artifact_id)
    if not artifact:
        raise ArtifactNotFoundError(f"Artifact '{artifact_id}' not found")

    parent_ids = [e.parent_artifact_id for e in artifact.parent_edges]
    child_ids = [e.child_artifact_id for e in artifact.child_edges]
    parent_artifacts = await lineage_repo.get_artifacts_by_ids(session, parent_ids)
    child_artifacts = await lineage_repo.get_artifacts_by_ids(session, child_ids)

    resp = ArtifactWithEdgesResponse.model_validate(artifact)
    resp.parents = [ArtifactResponse.model_validate(a) for a in parent_artifacts]
    resp.children = [ArtifactResponse.model_validate(a) for a in child_artifacts]
    return resp


async def list_artifacts(
    session: AsyncSession,
    artifact_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[ArtifactResponse], int]:
    artifacts, total = await lineage_repo.list_artifacts(
        session, artifact_type=artifact_type, limit=limit, offset=offset
    )
    return [ArtifactResponse.model_validate(a) for a in artifacts], total


async def create_edge(
    session: AsyncSession,
    request: CreateEdgeRequest,
) -> EdgeResponse:
    # 1. Both artifacts must exist
    parent = await lineage_repo.get_artifact(session, request.parent_artifact_id)
    if not parent:
        raise ArtifactNotFoundError(f"Parent artifact '{request.parent_artifact_id}' not found")
    child = await lineage_repo.get_artifact(session, request.child_artifact_id)
    if not child:
        raise ArtifactNotFoundError(f"Child artifact '{request.child_artifact_id}' not found")

    # 2. Idempotent — return existing edge if it already exists
    if await lineage_repo.edge_exists(
        session, request.parent_artifact_id, request.child_artifact_id
    ):
        result = await session.execute(
            select(LineageEdge).where(
                LineageEdge.parent_artifact_id == request.parent_artifact_id,
                LineageEdge.child_artifact_id == request.child_artifact_id,
            )
        )
        existing_edge = result.scalar_one()
        return EdgeResponse.model_validate(existing_edge)

    # 3. Cycle prevention
    if await would_create_cycle(session, request.parent_artifact_id, request.child_artifact_id):
        raise LineageCycleError(
            f"Adding edge '{request.parent_artifact_id}' → "
            f"'{request.child_artifact_id}' would create a cycle",
            detail={
                "parent_artifact_id": request.parent_artifact_id,
                "child_artifact_id": request.child_artifact_id,
            },
        )

    now = datetime.now(UTC)
    edge = LineageEdge(
        id=str(uuid7()),
        parent_artifact_id=request.parent_artifact_id,
        child_artifact_id=request.child_artifact_id,
        relationship_type=request.relationship_type.value,
        created_at=now,
        created_by=request.created_by,
    )
    edge = await lineage_repo.create_edge(session, edge)
    return EdgeResponse.model_validate(edge)


async def get_ancestors(
    session: AsyncSession,
    artifact_id: str,
    max_depth: int = 20,
) -> list[ArtifactResponse]:
    """Return all ancestors of an artifact, ordered by ascending depth."""
    artifact = await lineage_repo.get_artifact(session, artifact_id)
    if not artifact:
        raise ArtifactNotFoundError(f"Artifact '{artifact_id}' not found")

    rows = await lineage_repo.get_ancestors(session, artifact_id, max_depth)
    ancestor_ids = [r["artifact_id"] for r in rows]
    artifacts = await lineage_repo.get_artifacts_by_ids(session, ancestor_ids)
    id_to_depth = {r["artifact_id"]: r["depth"] for r in rows}
    artifacts.sort(key=lambda a: id_to_depth.get(a.id, 999))
    return [ArtifactResponse.model_validate(a) for a in artifacts]


async def get_descendants(
    session: AsyncSession,
    artifact_id: str,
    max_depth: int = 20,
) -> list[ArtifactResponse]:
    """Return all descendants of an artifact, ordered by ascending depth."""
    artifact = await lineage_repo.get_artifact(session, artifact_id)
    if not artifact:
        raise ArtifactNotFoundError(f"Artifact '{artifact_id}' not found")

    rows = await lineage_repo.get_descendants(session, artifact_id, max_depth)
    descendant_ids = [r["artifact_id"] for r in rows]
    artifacts = await lineage_repo.get_artifacts_by_ids(session, descendant_ids)
    id_to_depth = {r["artifact_id"]: r["depth"] for r in rows}
    artifacts.sort(key=lambda a: id_to_depth.get(a.id, 999))
    return [ArtifactResponse.model_validate(a) for a in artifacts]


async def get_plan_lineage_graph(
    session: AsyncSession,
    plan_name: str,
) -> LineageGraphResponse:
    """Return all artifacts and edges reachable from any artifact named after the plan.

    Convention: index artifacts for a plan are named '{plan_name}-index-*'.
    This searches for any artifact whose name starts with plan_name and
    returns the full connected subgraph.
    """
    all_artifacts, _ = await lineage_repo.list_artifacts(session, limit=1000)
    plan_artifacts = [a for a in all_artifacts if a.name.startswith(plan_name)]

    if not plan_artifacts:
        return LineageGraphResponse(plan_name=plan_name, artifacts=[], edges=[])

    # Expand to full connected component via ancestor + descendant traversal
    all_ids: set[str] = {a.id for a in plan_artifacts}
    for artifact in plan_artifacts:
        anc = await lineage_repo.get_ancestors(session, artifact.id)
        desc = await lineage_repo.get_descendants(session, artifact.id)
        all_ids.update(r["artifact_id"] for r in anc)
        all_ids.update(r["artifact_id"] for r in desc)

    connected = await lineage_repo.get_artifacts_by_ids(session, list(all_ids))
    edges = await lineage_repo.get_edges_for_artifacts(session, list(all_ids))

    return LineageGraphResponse(
        plan_name=plan_name,
        artifacts=[ArtifactResponse.model_validate(a) for a in connected],
        edges=[EdgeResponse.model_validate(e) for e in edges],
    )


async def get_orphans(session: AsyncSession) -> OrphansResponse:
    """Return artifacts with no downstream artifacts (leaf nodes)."""
    orphans = await lineage_repo.get_orphaned_artifacts(session)
    total = len(orphans)

    metrics.lineage_orphaned_artifacts_total.labels(artifact_type="all").set(total)

    return OrphansResponse(
        total=total,
        artifacts=[ArtifactResponse.model_validate(a) for a in orphans],
    )
