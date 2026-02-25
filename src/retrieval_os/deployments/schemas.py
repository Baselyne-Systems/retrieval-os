"""Pydantic schemas for the Deployments domain."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CreateDeploymentRequest(BaseModel):
    plan_version: int = Field(..., ge=1)
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
    plan_version: int
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
