"""Pydantic request/response schemas for the Plans domain."""

import base64
import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SLUG_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,253}[a-z0-9])?$")


def _validate_slug(v: str, field_name: str = "name") -> str:
    if not _SLUG_RE.match(v):
        raise ValueError(
            f"{field_name} must be a lowercase slug "
            "(letters, numbers, hyphens; cannot start or end with a hyphen)"
        )
    return v


# ── Request schemas ────────────────────────────────────────────────────────────

class PlanVersionConfig(BaseModel):
    """All fields that define the retrieval behaviour of a plan version."""

    # Embedding
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int = Field(768, gt=0)
    modalities: list[str] = Field(default_factory=lambda: ["text"], min_length=1)
    embedding_batch_size: int = Field(32, gt=0)
    embedding_normalize: bool = True

    # Index
    index_backend: str = "qdrant"
    index_collection: str
    distance_metric: str = "cosine"
    quantization: str | None = None

    # Retrieval
    top_k: int = Field(10, gt=0)
    rerank_top_k: int | None = Field(None, gt=0)
    reranker: str | None = None
    hybrid_alpha: float | None = Field(None, ge=0.0, le=1.0)

    # Filters
    metadata_filters: dict[str, Any] | None = None
    tenant_isolation_field: str | None = None

    # Cost / caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(3600, ge=0)
    max_tokens_per_query: int | None = Field(None, gt=0)

    # Governance
    change_comment: str = ""


class CreatePlanRequest(BaseModel):
    name: str
    description: str = ""
    config: PlanVersionConfig
    created_by: str = "system"

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return _validate_slug(v, "name")


class CreateVersionRequest(BaseModel):
    config: PlanVersionConfig
    created_by: str = "system"


class ClonePlanRequest(BaseModel):
    new_name: str
    description: str = ""
    created_by: str = "system"

    @field_validator("new_name")
    @classmethod
    def validate_new_name(cls, v: str) -> str:
        return _validate_slug(v, "new_name")


# ── Response schemas ───────────────────────────────────────────────────────────

class PlanVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version: int
    is_current: bool

    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    modalities: list[str]
    embedding_batch_size: int
    embedding_normalize: bool

    index_backend: str
    index_collection: str
    distance_metric: str
    quantization: str | None

    top_k: int
    rerank_top_k: int | None
    reranker: str | None
    hybrid_alpha: float | None

    metadata_filters: dict[str, Any] | None
    tenant_isolation_field: str | None

    cache_enabled: bool
    cache_ttl_seconds: int
    max_tokens_per_query: int | None

    change_comment: str
    config_hash: str
    created_at: datetime
    created_by: str


class PlanResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime
    created_by: str
    current_version: PlanVersionResponse | None


# ── Cursor pagination helpers ──────────────────────────────────────────────────

def encode_cursor(offset: int) -> str:
    return base64.b64encode(str(offset).encode()).decode()


def decode_cursor(cursor: str) -> int:
    try:
        return int(base64.b64decode(cursor).decode())
    except Exception:
        return 0
