"""FastAPI router for the Tenants domain."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.database import get_db
from retrieval_os.tenants import service
from retrieval_os.tenants.schemas import (
    ApiKeyCreatedResponse,
    ApiKeyResponse,
    CreateApiKeyRequest,
    CreateTenantRequest,
    TenantResponse,
)

router = APIRouter(prefix="/v1/tenants", tags=["tenants"])


@router.post("", status_code=201, response_model=TenantResponse)
async def create_tenant(
    request: CreateTenantRequest,
    db: AsyncSession = Depends(get_db),
) -> TenantResponse:
    """Create a new tenant with its rate-limit quota."""
    return await service.create_tenant(db, request)


@router.get("", response_model=list[TenantResponse])
async def list_tenants(
    db: AsyncSession = Depends(get_db),
) -> list[TenantResponse]:
    """List all tenants."""
    return await service.list_tenants(db)


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
) -> TenantResponse:
    """Get a single tenant by ID."""
    return await service.get_tenant(db, tenant_id)


@router.delete("/{tenant_id}", response_model=TenantResponse)
async def deactivate_tenant(
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
) -> TenantResponse:
    """Deactivate a tenant (soft-delete).  All API keys stop working immediately."""
    return await service.deactivate_tenant(db, tenant_id)


# ── API keys ──────────────────────────────────────────────────────────────────


@router.post("/{tenant_id}/api-keys", status_code=201, response_model=ApiKeyCreatedResponse)
async def create_api_key(
    tenant_id: str,
    request: CreateApiKeyRequest,
    db: AsyncSession = Depends(get_db),
) -> ApiKeyCreatedResponse:
    """Create an API key for a tenant.

    The ``key`` field in the response contains the raw secret.  It is shown
    **exactly once** — store it securely before the request ends.
    """
    return await service.create_api_key(db, tenant_id, request)


@router.get("/{tenant_id}/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[ApiKeyResponse]:
    """List all API keys for a tenant (raw key values are never returned)."""
    return await service.list_api_keys(db, tenant_id)


@router.delete("/{tenant_id}/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    tenant_id: str,
    key_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Revoke (soft-delete) an API key.  Takes effect on the next request."""
    await service.revoke_api_key(db, tenant_id, key_id)
