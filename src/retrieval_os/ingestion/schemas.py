"""Pydantic schemas for the Ingestion domain."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator


class IngestDocumentRequest(BaseModel):
    """A single document to be chunked, embedded, and upserted."""

    id: str = Field(..., min_length=1, description="Stable document identifier.")
    content: str = Field(..., min_length=1, description="Raw text content.")
    metadata: dict = Field(
        default_factory=dict, description="Arbitrary key-value payload stored with each chunk."
    )


class IngestRequest(BaseModel):
    """Request body for ``POST /v1/projects/{project}/ingest``."""

    # Source: either inline documents OR an S3 URI (exactly one required)
    documents: list[IngestDocumentRequest] | None = Field(
        default=None,
        description="Inline documents to ingest (mutually exclusive with source_uri).",
    )
    source_uri: str | None = Field(
        default=None,
        description=(
            "S3 URI of a JSONL file where each line is "
            '{"id": str, "content": str, "metadata": {}}. '
            "Mutually exclusive with documents."
        ),
    )

    index_config_version: int = Field(
        ..., ge=1, description="Index config version whose collection to populate."
    )
    chunk_size: int = Field(default=512, ge=16, le=4096)
    overlap: int = Field(default=64, ge=0)
    created_by: str | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        has_docs = self.documents is not None
        has_uri = self.source_uri is not None
        if not has_docs and not has_uri:
            raise ValueError("Either 'documents' or 'source_uri' must be provided.")
        if has_docs and has_uri:
            raise ValueError("Only one of 'documents' or 'source_uri' may be provided.")
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self


class IngestionJobResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    plan_name: str
    index_config_id: uuid.UUID
    index_config_version: int
    source_uri: str | None
    chunk_size: int
    overlap: int
    status: str
    total_docs: int | None
    total_chunks: int | None
    indexed_chunks: int | None
    failed_chunks: int | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    created_by: str | None


class IngestionJobListResponse(BaseModel):
    items: list[IngestionJobResponse]
    total: int
