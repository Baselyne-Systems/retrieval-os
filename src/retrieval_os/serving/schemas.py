"""Request/response schemas for the serving path."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8192)
    metadata_filters: dict | None = Field(
        None,
        description="Request-level metadata filters merged over plan-level filters.",
    )


class ChunkResponse(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict


class QueryResponse(BaseModel):
    plan_name: str
    version: int
    cache_hit: bool
    results: list[ChunkResponse]
    result_count: int
