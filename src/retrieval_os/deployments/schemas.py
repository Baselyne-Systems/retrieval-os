"""Pydantic schemas for the Deployments domain."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CreateDeploymentRequest(BaseModel):
    index_config_version: int = Field(..., ge=1)
    # Search config — all optional with sensible defaults
    top_k: int = Field(10, gt=0)
    rerank_top_k: int | None = Field(None, gt=0)
    reranker: str | None = None
    hybrid_alpha: float | None = Field(None, ge=0.0, le=1.0)
    metadata_filters: dict[str, Any] | None = None
    tenant_isolation_field: str | None = None
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(3600, ge=0)
    max_tokens_per_query: int | None = Field(None, gt=0)
    # Instant deploy = no rollout fields; gradual deploy = both required
    rollout_step_percent: float | None = Field(None, gt=0, le=100)
    rollout_step_interval_seconds: int | None = Field(None, ge=10)
    # Rollback guard rails (optional)
    rollback_recall_threshold: float | None = Field(None, ge=0.0, le=1.0)
    rollback_error_rate_threshold: float | None = Field(None, ge=0.0, le=1.0)
    change_note: str = Field("", max_length=2048)
    created_by: str = Field(..., min_length=1, max_length=255)


class RollbackRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=2048)
    created_by: str = Field(..., min_length=1, max_length=255)


class DeploymentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    plan_name: str
    index_config_id: uuid.UUID
    index_config_version: int
    # Search config
    top_k: int
    rerank_top_k: int | None
    reranker: str | None
    hybrid_alpha: float | None
    metadata_filters: dict[str, Any] | None
    tenant_isolation_field: str | None
    cache_enabled: bool
    cache_ttl_seconds: int
    max_tokens_per_query: int | None
    status: str
    traffic_weight: float
    rollout_step_percent: float | None
    rollout_step_interval_seconds: int | None
    rollback_recall_threshold: float | None
    rollback_error_rate_threshold: float | None
    change_note: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    activated_at: datetime | None
    rolled_back_at: datetime | None
    rollback_reason: str | None
