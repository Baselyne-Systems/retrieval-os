"""Pydantic request/response schemas for the Tenants domain."""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SLUG_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,253}[a-z0-9])?$")


# ── Tenant schemas ────────────────────────────────────────────────────────────


class CreateTenantRequest(BaseModel):
    name: str
    max_requests_per_minute: int = Field(60, ge=1, le=10_000)
    max_plans: int = Field(10, ge=1, le=1000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not _SLUG_RE.match(v):
            raise ValueError(
                "name must be a lowercase slug "
                "(letters, numbers, hyphens; cannot start or end with a hyphen)"
            )
        return v


class TenantResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    is_active: bool
    max_requests_per_minute: int
    max_plans: int
    created_at: datetime
    updated_at: datetime


# ── API key schemas ───────────────────────────────────────────────────────────


class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    expires_at: datetime | None = None


class ApiKeyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    tenant_id: str
    name: str
    key_prefix: str
    is_active: bool
    created_at: datetime
    expires_at: datetime | None


class ApiKeyCreatedResponse(ApiKeyResponse):
    """Returned exactly once at key creation.  The raw key is never stored."""

    key: str
