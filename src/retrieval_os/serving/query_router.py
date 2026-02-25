"""Query Router — resolves plan config and dispatches to the retrieval executor.

Design constraints:
- NEVER reads Postgres on the hot path.
- Plan config is served from Redis (key: ros:plan:{name}:current).
- On Redis miss, falls back to a single Postgres read and warms the cache.
- Traffic splitting (Phase 4) will be handled here once Deployments exist.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.exceptions import PlanNotFoundError
from retrieval_os.core.redis_client import get_redis
from retrieval_os.plans.repository import plan_repo
from retrieval_os.serving.embed_router import embed_audio, embed_images
from retrieval_os.serving.executor import RetrievedChunk, execute_retrieval
from retrieval_os.serving.index_proxy import IndexHit, vector_search

log = logging.getLogger(__name__)

_PLAN_CACHE_TTL = 30  # seconds; background loop refreshes this every 5 s


def _plan_redis_key(name: str) -> str:
    return f"ros:plan:{name}:current"


async def _load_plan_config(plan_name: str, db: AsyncSession) -> dict:
    """Return plan config dict; check Redis first, fall back to Postgres."""
    redis = await get_redis()
    key = _plan_redis_key(plan_name)

    raw = await redis.get(key)
    if raw:
        return json.loads(raw)

    # Cache miss — load from Postgres and warm Redis.
    plan = await plan_repo.get_by_name(db, plan_name)
    if not plan:
        raise PlanNotFoundError(f"Plan '{plan_name}' not found")
    if plan.is_archived:
        raise PlanNotFoundError(f"Plan '{plan_name}' is archived")

    version = plan.current_version
    if not version:
        raise PlanNotFoundError(f"Plan '{plan_name}' has no current version")

    config = {
        "plan_name": plan.name,
        "version": version.version,
        "embedding_provider": version.embedding_provider,
        "embedding_model": version.embedding_model,
        "embedding_normalize": version.embedding_normalize,
        "embedding_batch_size": version.embedding_batch_size,
        "index_backend": version.index_backend,
        "index_collection": version.index_collection,
        "distance_metric": version.distance_metric,
        "top_k": version.top_k,
        "reranker": version.reranker,
        "rerank_top_k": version.rerank_top_k,
        "metadata_filters": version.metadata_filters,
        "cache_enabled": version.cache_enabled,
        "cache_ttl_seconds": version.cache_ttl_seconds,
        "hybrid_alpha": version.hybrid_alpha,
    }

    try:
        await redis.set(key, json.dumps(config, default=str), ex=_PLAN_CACHE_TTL)
    except Exception:
        log.warning("query_router.redis_warm_failed", extra={"plan": plan_name})

    return config


async def route_query(
    *,
    plan_name: str,
    query: str,
    db: AsyncSession,
    metadata_filter_override: dict | None = None,
) -> tuple[list[RetrievedChunk], dict]:
    """Resolve plan config and execute retrieval.

    Args:
        plan_name:               Name of the retrieval plan.
        query:                   The natural-language query string.
        db:                      AsyncSession — only used on Redis miss.
        metadata_filter_override: Request-level filter merged over plan filters.

    Returns:
        (chunks, info_dict) where info_dict carries version, cache_hit, etc.
    """
    config = await _load_plan_config(plan_name, db)

    # Merge metadata filters: request-level overrides plan-level.
    filters = config.get("metadata_filters") or {}
    if metadata_filter_override:
        filters = {**filters, **metadata_filter_override}

    chunks, cache_hit = await execute_retrieval(
        plan_name=plan_name,
        version=config["version"],
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
    )

    info = {
        "plan_name": plan_name,
        "version": config["version"],
        "cache_hit": cache_hit,
        "result_count": len(chunks),
    }
    return chunks, info


# ── Multimodal query routes ───────────────────────────────────────────────────


async def route_image_query(
    *,
    plan_name: str,
    image_bytes: bytes,
    db: AsyncSession,
) -> tuple[list[RetrievedChunk], dict]:
    """Embed an image with CLIP and search the plan's vector index.

    The plan must have ``embedding_provider = "clip"`` set.

    Returns:
        (chunks, info_dict) matching the shape of route_query.
    """
    config = await _load_plan_config(plan_name, db)

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
        "plan_name": plan_name,
        "version": config["version"],
        "cache_hit": False,
        "result_count": len(chunks),
    }


async def route_audio_query(
    *,
    plan_name: str,
    audio_bytes: bytes,
    db: AsyncSession,
    whisper_model_size: str = "base",
) -> tuple[list[RetrievedChunk], dict]:
    """Transcribe audio with Whisper, embed the transcript, and search.

    The plan's ``embedding_provider`` / ``embedding_model`` are used for the
    text embedding step after transcription.

    Returns:
        (chunks, info_dict) matching the shape of route_query.
    """
    config = await _load_plan_config(plan_name, db)

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
        "plan_name": plan_name,
        "version": config["version"],
        "cache_hit": False,
        "result_count": len(chunks),
    }


# ── Shared helpers ────────────────────────────────────────────────────────────


async def _embed_and_search(
    *,
    vectors: list[list[float]],
    config: dict,
) -> list[IndexHit]:
    """Run vector search with the first vector in *vectors* against plan config."""
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
