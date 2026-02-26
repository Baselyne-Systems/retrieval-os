"""FastAPI router for multimodal (image / audio) retrieval queries.

Endpoints
---------
POST /v1/query/{project_name}/image  — Upload an image; CLIP-embed and search.
POST /v1/query/{project_name}/audio  — Upload audio; Whisper-transcribe, embed, and search.

Both endpoints return the same ``QueryResponse`` schema as the text query
endpoint so clients need no special handling.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.database import get_db
from retrieval_os.serving.query_router import route_audio_query, route_image_query
from retrieval_os.serving.schemas import ChunkResponse, QueryResponse

router = APIRouter(prefix="/v1/query", tags=["multimodal"])

_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB
_MAX_AUDIO_BYTES = 100 * 1024 * 1024  # 100 MB


@router.post("/{project_name}/image", response_model=QueryResponse)
async def query_image(
    project_name: str,
    image: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, …)"),
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """Embed an image with CLIP and retrieve the top-k most similar chunks.

    The project must be configured with ``embedding_provider = "clip"``.
    Image bytes are read fully into memory; max size is 20 MB.
    """
    raw = await image.read(_MAX_IMAGE_BYTES)
    chunks, info = await route_image_query(
        project_name=project_name,
        image_bytes=raw,
        db=db,
    )
    return _build_response(chunks, info)


@router.post("/{project_name}/audio", response_model=QueryResponse)
async def query_audio(
    project_name: str,
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, …)"),
    whisper_model_size: str = Query(
        "base",
        description="Whisper model size: tiny, base, small, medium, large-v3",
    ),
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """Transcribe audio with Whisper, embed the transcript, and retrieve chunks.

    The project's text ``embedding_provider``/``embedding_model`` are used for
    the final embedding step.  Audio bytes are read fully into memory; max
    size is 100 MB.
    """
    raw = await audio.read(_MAX_AUDIO_BYTES)
    chunks, info = await route_audio_query(
        project_name=project_name,
        audio_bytes=raw,
        db=db,
        whisper_model_size=whisper_model_size,
    )
    return _build_response(chunks, info)


def _build_response(chunks: list, info: dict) -> QueryResponse:
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
