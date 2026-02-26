"""Query Router — resolves project config and dispatches to the retrieval executor.

Design constraints:
- NEVER reads Postgres on the hot path.
- Serving config is served from Redis (key: ros:project:{name}:active).
- On Redis miss, falls back to a single Postgres read and warms the cache.
"""

from __future__ import annotations

import asyncio
import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.config import settings
from retrieval_os.core.exceptions import ProjectNotFoundError, QueryTimeoutError
from retrieval_os.core.redis_client import get_redis
from retrieval_os.deployments.repository import deployment_repo
from retrieval_os.plans.repository import project_repo
from retrieval_os.serving.embed_router import embed_audio, embed_images
from retrieval_os.serving.executor import RetrievedChunk, execute_retrieval
from retrieval_os.serving.index_proxy import IndexHit, vector_search

log = logging.getLogger(__name__)

_PROJECT_CACHE_TTL = 30  # seconds; background loop refreshes this every 5 s


def _project_redis_key(name: str) -> str:
    return f"ros:project:{name}:active"


async def _load_project_config(project_name: str, db: AsyncSession) -> dict:
    """Return serving config dict; check Redis first, fall back to Postgres."""
    redis = await get_redis()
    key = _project_redis_key(project_name)

    raw = await redis.get(key)
    if raw:
        return json.loads(raw)

    # Cache miss — load from Postgres and warm Redis.
    project = await project_repo.get_by_name(db, project_name)
    if not project:
        raise ProjectNotFoundError(f"Project '{project_name}' not found")
    if project.is_archived:
        raise ProjectNotFoundError(f"Project '{project_name}' is archived")

    # Load the active deployment
    deployment = await deployment_repo.get_active_for_project(db, project_name)
    if not deployment:
        raise ProjectNotFoundError(f"Project '{project_name}' has no active deployment")

    # Load the index config
    index_config = await project_repo.get_index_config_by_id(db, deployment.index_config_id)
    if not index_config:
        raise ProjectNotFoundError(f"Project '{project_name}' index config not found")

    config = {
        "project_name": project_name,
        "index_config_version": deployment.index_config_version,
        "embedding_provider": index_config.embedding_provider,
        "embedding_model": index_config.embedding_model,
        "embedding_normalize": index_config.embedding_normalize,
        "embedding_batch_size": index_config.embedding_batch_size,
        "index_backend": index_config.index_backend,
        "index_collection": index_config.index_collection,
        "distance_metric": index_config.distance_metric,
        "top_k": deployment.top_k,
        "reranker": deployment.reranker,
        "rerank_top_k": deployment.rerank_top_k,
        "metadata_filters": deployment.metadata_filters,
        "cache_enabled": deployment.cache_enabled,
        "cache_ttl_seconds": deployment.cache_ttl_seconds,
        "hybrid_alpha": deployment.hybrid_alpha,
    }

    try:
        await redis.set(key, json.dumps(config, default=str), ex=_PROJECT_CACHE_TTL)
    except Exception:
        log.warning("query_router.redis_warm_failed", extra={"project": project_name})

    return config


async def route_query(
    *,
    project_name: str,
    query: str,
    db: AsyncSession,
    metadata_filter_override: dict | None = None,
) -> tuple[list[RetrievedChunk], dict]:
    """Resolve project config and execute retrieval.

    Args:
        project_name:            Name of the project.
        query:                   The natural-language query string.
        db:                      AsyncSession — only used on Redis miss.
        metadata_filter_override: Request-level filter merged over deployment filters.

    Returns:
        (chunks, info_dict) where info_dict carries version, cache_hit, etc.
    """
    config = await _load_project_config(project_name, db)

    # Merge metadata filters: request-level overrides deployment-level.
    filters = config.get("metadata_filters") or {}
    if metadata_filter_override:
        filters = {**filters, **metadata_filter_override}

    try:
        chunks, cache_hit = await asyncio.wait_for(
            execute_retrieval(
                project_name=project_name,
                version=config["index_config_version"],
                query=query,
                embedding_provider=config["embedding_provider"],
                embedding_model=config["embedding_model"],
                embedding_normalize=config["embedding_normalize"],
                embedding_batch_size=config["embedding_batch_size"],
                index_backend=config["index_backend"],
                index_collection=config["index_collection"],
                distance_metric=config["distance_metric"],
                top_k=config["top_k"],
                reranker=config.get("reranker"),
                rerank_top_k=config.get("rerank_top_k"),
                metadata_filters=filters or None,
                cache_enabled=config["cache_enabled"],
                cache_ttl_seconds=config["cache_ttl_seconds"],
                hybrid_alpha=config.get("hybrid_alpha"),
            ),
            timeout=settings.query_timeout_seconds,
        )
    except TimeoutError:
        raise QueryTimeoutError(settings.query_timeout_seconds)

    info = {
        "project_name": project_name,
        "version": config["index_config_version"],
        "cache_hit": cache_hit,
        "result_count": len(chunks),
    }
    return chunks, info


# ── Multimodal query routes ───────────────────────────────────────────────────


async def route_image_query(
    *,
    project_name: str,
    image_bytes: bytes,
    db: AsyncSession,
) -> tuple[list[RetrievedChunk], dict]:
    """Embed an image with CLIP and search the project's vector index."""
    config = await _load_project_config(project_name, db)

    hits: list[IndexHit] = await _embed_and_search(
        vectors=await embed_images(
            [image_bytes],
            provider=config["embedding_provider"],
            model=config["embedding_model"],
        ),
        config=config,
    )

    chunks = _hits_to_chunks(hits)
    return chunks, {
        "project_name": project_name,
        "version": config["index_config_version"],
        "cache_hit": False,
        "result_count": len(chunks),
    }


async def route_audio_query(
    *,
    project_name: str,
    audio_bytes: bytes,
    db: AsyncSession,
    whisper_model_size: str = "base",
) -> tuple[list[RetrievedChunk], dict]:
    """Transcribe audio with Whisper, embed the transcript, and search."""
    config = await _load_project_config(project_name, db)

    hits = await _embed_and_search(
        vectors=await embed_audio(
            [audio_bytes],
            whisper_model_size=whisper_model_size,
            text_provider=config["embedding_provider"],
            text_model=config["embedding_model"],
        ),
        config=config,
    )

    chunks = _hits_to_chunks(hits)
    return chunks, {
        "project_name": project_name,
        "version": config["index_config_version"],
        "cache_hit": False,
        "result_count": len(chunks),
    }


# ── Shared helpers ────────────────────────────────────────────────────────────


async def _embed_and_search(
    *,
    vectors: list[list[float]],
    config: dict,
) -> list[IndexHit]:
    """Run vector search with the first vector in *vectors* against project config."""
    return await vector_search(
        backend=config["index_backend"],
        collection=config["index_collection"],
        vector=vectors[0],
        top_k=config["top_k"],
        distance_metric=config["distance_metric"],
        metadata_filters=config.get("metadata_filters"),
    )


def _hits_to_chunks(hits: list[IndexHit]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            id=h.id,
            score=h.score,
            text=h.payload.get("text", ""),
            metadata={k: v for k, v in h.payload.items() if k != "text"},
        )
        for h in hits
    ]
