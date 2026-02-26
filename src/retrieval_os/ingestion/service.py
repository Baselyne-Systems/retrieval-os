"""Business logic for the Document Ingestion domain.

Pipeline per job:
  1. Claim next QUEUED job (SELECT FOR UPDATE SKIP LOCKED).
  2. Load index config from Postgres.
  3. Fetch raw documents from S3 (JSONL) or use inline payload.
  4. Chunk each document using the word-boundary chunker.
  5. Embed chunks in batches via embed_router.
  6. Upsert dense vectors into the configured Qdrant collection.
  7. Register lineage artifacts (DatasetSnapshot → EmbeddingArtifact →
     IndexArtifact) and edges in the lineage DAG.
  8. Fire ``plan.version_created`` webhook if any chunks were indexed.
  9. Mark job COMPLETED (or FAILED on unrecoverable error).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.ids import uuid7
from retrieval_os.ingestion.chunker import chunk_text
from retrieval_os.ingestion.models import IngestionJob
from retrieval_os.ingestion.repository import ingestion_repo
from retrieval_os.ingestion.schemas import IngestRequest
from retrieval_os.lineage.models import (
    ArtifactType,
    EdgeRelationship,
    LineageArtifact,
    LineageEdge,
)
from retrieval_os.lineage.repository import lineage_repo
from retrieval_os.plans.models import IndexConfig, Project
from retrieval_os.serving.embed_router import embed_text
from retrieval_os.serving.index_proxy import ensure_collection, upsert_vectors
from retrieval_os.webhooks.delivery import fire_webhook_event
from retrieval_os.webhooks.events import WebhookEvent

log = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 50


# ── Index config loader ────────────────────────────────────────────────────────


async def _load_index_config(session: AsyncSession, project_name: str, version: int) -> IndexConfig:
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


# ── S3 document loader ────────────────────────────────────────────────────────


async def _load_docs_from_s3(source_uri: str) -> list[dict]:
    """Download a JSONL file from S3 and parse each line as a document dict."""
    from retrieval_os.core.s3_client import download_object

    # Parse s3://bucket/key
    path = source_uri.split("://", 1)[1]
    bucket, _, key = path.partition("/")

    raw_bytes = await download_object(key, bucket=bucket)
    docs: list[dict] = []
    for line in raw_bytes.decode().splitlines():
        line = line.strip()
        if line:
            docs.append(json.loads(line))
    return docs


# ── Lineage registration ──────────────────────────────────────────────────────


async def _register_ingestion_lineage(
    session: AsyncSession,
    job: IngestionJob,
    ic: IndexConfig,
    total_chunks: int,
    indexed_chunks: int,
) -> None:
    now = datetime.now(UTC)
    creator = job.created_by or "ingestion-pipeline"

    dataset_uri = job.source_uri or f"inline://{job.id}"
    embed_uri = f"embed://{job.id}"
    index_uri = f"qdrant://{ic.index_collection}"

    # Idempotent — skip if already registered (re-runs of same job)
    dataset_art = await lineage_repo.get_artifact_by_uri(session, dataset_uri)
    if dataset_art is None:
        dataset_art = await lineage_repo.create_artifact(
            session,
            LineageArtifact(
                id=str(uuid7()),
                artifact_type=ArtifactType.DATASET_SNAPSHOT.value,
                name=f"{job.plan_name}-dataset-v{job.index_config_version}",
                version=str(job.index_config_version),
                storage_uri=dataset_uri,
                content_hash=None,
                artifact_metadata={
                    "job_id": job.id,
                    "total_docs": job.total_docs,
                    "total_chunks": total_chunks,
                },
                created_at=now,
                created_by=creator,
            ),
        )

    embed_art = await lineage_repo.get_artifact_by_uri(session, embed_uri)
    if embed_art is None:
        embed_art = await lineage_repo.create_artifact(
            session,
            LineageArtifact(
                id=str(uuid7()),
                artifact_type=ArtifactType.EMBEDDING_ARTIFACT.value,
                name=f"{job.plan_name}-embeddings-v{job.index_config_version}",
                version=str(job.index_config_version),
                storage_uri=embed_uri,
                content_hash=None,
                artifact_metadata={
                    "provider": ic.embedding_provider,
                    "model": ic.embedding_model,
                    "indexed_chunks": indexed_chunks,
                },
                created_at=now,
                created_by=creator,
            ),
        )

    index_art = await lineage_repo.get_artifact_by_uri(session, index_uri)
    if index_art is None:
        index_art = await lineage_repo.create_artifact(
            session,
            LineageArtifact(
                id=str(uuid7()),
                artifact_type=ArtifactType.INDEX_ARTIFACT.value,
                name=f"{job.plan_name}-index-v{job.index_config_version}",
                version=str(job.index_config_version),
                storage_uri=index_uri,
                content_hash=None,
                artifact_metadata={
                    "backend": ic.index_backend,
                    "collection": ic.index_collection,
                },
                created_at=now,
                created_by=creator,
            ),
        )

    # Edges: Dataset → Embed → Index
    existing_de = await lineage_repo.edge_exists(session, dataset_art.id, embed_art.id)
    if not existing_de:
        await lineage_repo.create_edge(
            session,
            LineageEdge(
                id=str(uuid7()),
                parent_artifact_id=dataset_art.id,
                child_artifact_id=embed_art.id,
                relationship_type=EdgeRelationship.PRODUCED_FROM.value,
                created_at=now,
                created_by=creator,
            ),
        )

    existing_ei = await lineage_repo.edge_exists(session, embed_art.id, index_art.id)
    if not existing_ei:
        await lineage_repo.create_edge(
            session,
            LineageEdge(
                id=str(uuid7()),
                parent_artifact_id=embed_art.id,
                child_artifact_id=index_art.id,
                relationship_type=EdgeRelationship.DERIVED_FROM.value,
                created_at=now,
                created_by=creator,
            ),
        )


# ── Public API ────────────────────────────────────────────────────────────────


async def create_ingestion_job(
    session: AsyncSession,
    plan_name: str,
    request: IngestRequest,
) -> IngestionJob:
    """Validate the index config exists, persist a QUEUED job, return it."""
    ic = await _load_index_config(session, plan_name, request.index_config_version)

    now = datetime.now(UTC)
    job = IngestionJob(
        id=str(uuid7()),
        plan_name=plan_name,
        index_config_id=ic.id,
        index_config_version=request.index_config_version,
        source_uri=request.source_uri,
        document_payload=(
            [doc.model_dump() for doc in request.documents]
            if request.documents is not None
            else None
        ),
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        status="QUEUED",
        created_at=now,
        created_by=request.created_by,
    )
    return await ingestion_repo.create(session, job)


async def process_next_ingestion_job(session: AsyncSession) -> str | None:
    """Claim and execute the next QUEUED ingestion job.

    Uses SELECT FOR UPDATE SKIP LOCKED so concurrent runners never
    process the same job.  Returns the job ID if a job was processed,
    else None.  The session is committed by the caller (background loop).
    """
    job = await ingestion_repo.claim_next_queued(session)
    if job is None:
        return None

    job_id = job.id
    log.info("ingestion.job.started", extra={"job_id": job_id, "plan": job.plan_name})

    try:
        # 1. Load index config
        ic = await _load_index_config(session, job.plan_name, job.index_config_version)

        # 2. Load documents
        if job.source_uri:
            docs = await _load_docs_from_s3(job.source_uri)
        else:
            docs = job.document_payload or []

        total_docs = len(docs)

        # 3. Build chunk list
        all_chunks: list[dict] = []
        for doc in docs:
            doc_id = doc.get("id", str(uuid7()))
            content = doc.get("content", "")
            metadata = doc.get("metadata") or {}
            for idx, chunk in enumerate(
                chunk_text(content, chunk_size=job.chunk_size, overlap=job.overlap)
            ):
                all_chunks.append(
                    {
                        "id": f"{doc_id}_c{idx}",
                        "text": chunk,
                        "payload": {
                            "doc_id": doc_id,
                            "chunk_idx": idx,
                            "plan_name": job.plan_name,
                            "index_config_version": job.index_config_version,
                            **metadata,
                        },
                    }
                )

        total_chunks = len(all_chunks)
        indexed_chunks = 0
        failed_chunks = 0
        collection_ensured = False

        # 4. Embed + upsert in batches
        for batch_start in range(0, total_chunks, _EMBED_BATCH_SIZE):
            batch = all_chunks[batch_start : batch_start + _EMBED_BATCH_SIZE]
            texts = [c["text"] for c in batch]
            try:
                vectors = await embed_text(
                    texts,
                    provider=ic.embedding_provider,
                    model=ic.embedding_model,
                    normalize=ic.embedding_normalize,
                    batch_size=ic.embedding_batch_size,
                )
                # Ensure the Qdrant collection exists before the first upsert
                if not collection_ensured and vectors:
                    await ensure_collection(
                        backend=ic.index_backend,
                        collection=ic.index_collection,
                        dimension=len(vectors[0]),
                        distance=ic.distance_metric,
                    )
                    collection_ensured = True
                points = [
                    {"id": c["id"], "vector": v, "payload": c["payload"]}
                    for c, v in zip(batch, vectors)
                ]
                await upsert_vectors(
                    backend=ic.index_backend,
                    collection=ic.index_collection,
                    points=points,
                )
                indexed_chunks += len(batch)
            except Exception:
                failed_chunks += len(batch)
                log.warning(
                    "ingestion.batch_failed",
                    extra={"job_id": job_id, "batch_start": batch_start},
                )

        # 5. Register lineage
        await _register_ingestion_lineage(session, job, ic, total_chunks, indexed_chunks)

        # 6. Fire webhook if anything was indexed
        if indexed_chunks > 0:
            await fire_webhook_event(
                WebhookEvent.PLAN_VERSION_CREATED,
                {
                    "job_id": job_id,
                    "plan_name": job.plan_name,
                    "index_config_version": job.index_config_version,
                    "indexed_chunks": indexed_chunks,
                    "failed_chunks": failed_chunks,
                },
                session,
            )

        # 7. Mark completed
        await ingestion_repo.complete_job(
            session,
            job_id,
            total_docs=total_docs,
            total_chunks=total_chunks,
            indexed_chunks=indexed_chunks,
            failed_chunks=failed_chunks,
        )

        log.info(
            "ingestion.job.completed",
            extra={
                "job_id": job_id,
                "plan": job.plan_name,
                "total_docs": total_docs,
                "indexed_chunks": indexed_chunks,
                "failed_chunks": failed_chunks,
            },
        )

    except Exception as exc:
        log.exception("ingestion.job.failed", extra={"job_id": job_id})
        await ingestion_repo.fail_job(session, job_id, error_message=str(exc))

    return job_id
