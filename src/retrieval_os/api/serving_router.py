"""FastAPI router for the Serving path.

POST /v1/query/{project_name}  — hot path, P99 target < 200 ms.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.database import async_session_factory, get_db
from retrieval_os.core.ids import uuid7
from retrieval_os.serving.query_router import route_query
from retrieval_os.serving.schemas import ChunkResponse, QueryRequest, QueryResponse
from retrieval_os.serving.usage import fire_usage_record

router = APIRouter(prefix="/v1/query", tags=["serving"])


@router.post("/{project_name}", response_model=QueryResponse)
async def query(
    project_name: str,
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """Execute a retrieval query against the named project's current deployment.

    Returns ranked chunks from the project's configured vector index.
    """
    t0 = time.perf_counter()

    chunks, info = await route_query(
        project_name=project_name,
        query=request.query,
        db=db,
        metadata_filter_override=request.metadata_filters,
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    fire_usage_record(
        async_session_factory,
        record_id=str(uuid7()),
        project_name=project_name,
        index_config_version=info["version"],
        query_chars=len(request.query),
        result_count=info["result_count"],
        cache_hit=info["cache_hit"],
        latency_ms=latency_ms,
    )

    return QueryResponse(
        project_name=info["project_name"],
        version=info["version"],
        cache_hit=info["cache_hit"],
        results=[
            ChunkResponse(
                id=c.id,
                score=c.score,
                text=c.text,
                metadata=c.metadata,
            )
            for c in chunks
        ],
        result_count=info["result_count"],
    )
