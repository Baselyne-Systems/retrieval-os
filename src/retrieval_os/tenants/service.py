"""Service layer for the Tenants domain.

Key design decisions:
- Raw API keys are generated once and returned to the caller; only the SHA-256
  hash is stored.  If a key is lost it cannot be recovered — a new one must be
  issued.
- Key format: ``ros_<8-hex>_<56-hex>``  (total 69 chars, prefix = first 12).
- Prefix is used as a fast index-lookup; the hash confirms identity.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.core.exceptions import ConflictError, NotFoundError
from retrieval_os.core.ids import uuid7
from retrieval_os.tenants import repository
from retrieval_os.tenants.models import ApiKey, Tenant
from retrieval_os.tenants.schemas import (
    ApiKeyCreatedResponse,
    ApiKeyResponse,
    CreateApiKeyRequest,
    CreateTenantRequest,
    TenantResponse,
)

_KEY_RANDOM_BYTES = 32  # 64 hex chars after token_hex()


# ── Key helpers ───────────────────────────────────────────────────────────────


def generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key triple.

    Returns:
        (full_key, prefix, hash_hex) where:
        - ``full_key``  is the raw key returned to the caller exactly once.
        - ``prefix``    is ``ros_<8-hex>`` (12 chars) used for DB lookup.
        - ``hash_hex``  is ``SHA-256(full_key)`` stored in the DB.
    """
    random_hex = secrets.token_hex(_KEY_RANDOM_BYTES)  # 64 hex chars
    prefix = f"ros_{random_hex[:8]}"  # "ros_" + 8 hex = 12 chars
    full_key = f"{prefix}_{random_hex[8:]}"  # 12 + 1 + 56 = 69 chars
    hash_hex = hashlib.sha256(full_key.encode()).hexdigest()
    return full_key, prefix, hash_hex


def hash_api_key(key: str) -> str:
    """Return SHA-256 hex digest of an API key string."""
    return hashlib.sha256(key.encode()).hexdigest()


# ── Tenant CRUD ───────────────────────────────────────────────────────────────


async def create_tenant(db: AsyncSession, request: CreateTenantRequest) -> TenantResponse:
    existing = await repository.get_tenant_by_name(db, request.name)
    if existing:
        raise ConflictError(f"Tenant '{request.name}' already exists")

    now = datetime.now(UTC)
    tenant = Tenant(
        id=str(uuid7()),
        name=request.name,
        is_active=True,
        max_requests_per_minute=request.max_requests_per_minute,
        max_plans=request.max_plans,
        created_at=now,
        updated_at=now,
    )
    db.add(tenant)
    await db.flush()
    return TenantResponse.model_validate(tenant)


async def list_tenants(db: AsyncSession) -> list[TenantResponse]:
    tenants = await repository.list_tenants(db)
    return [TenantResponse.model_validate(t) for t in tenants]


async def get_tenant(db: AsyncSession, tenant_id: str) -> TenantResponse:
    tenant = await repository.get_tenant(db, tenant_id)
    if not tenant:
        raise NotFoundError(f"Tenant '{tenant_id}' not found")
    return TenantResponse.model_validate(tenant)


async def deactivate_tenant(db: AsyncSession, tenant_id: str) -> TenantResponse:
    tenant = await repository.get_tenant(db, tenant_id)
    if not tenant:
        raise NotFoundError(f"Tenant '{tenant_id}' not found")
    tenant.is_active = False
    tenant.updated_at = datetime.now(UTC)
    await db.flush()
    return TenantResponse.model_validate(tenant)


# ── API key CRUD ──────────────────────────────────────────────────────────────


async def create_api_key(
    db: AsyncSession,
    tenant_id: str,
    request: CreateApiKeyRequest,
) -> ApiKeyCreatedResponse:
    """Create a new API key.  The raw key is returned in the response and
    never stored — this is the caller's only chance to save it."""
    tenant = await repository.get_tenant(db, tenant_id)
    if not tenant:
        raise NotFoundError(f"Tenant '{tenant_id}' not found")

    full_key, prefix, key_hash = generate_api_key()
    now = datetime.now(UTC)
    api_key = ApiKey(
        id=str(uuid7()),
        tenant_id=tenant_id,
        name=request.name,
        key_prefix=prefix,
        key_hash=key_hash,
        is_active=True,
        created_at=now,
        expires_at=request.expires_at,
    )
    db.add(api_key)
    await db.flush()

    return ApiKeyCreatedResponse(
        id=api_key.id,
        tenant_id=api_key.tenant_id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        is_active=api_key.is_active,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        key=full_key,
    )


async def list_api_keys(db: AsyncSession, tenant_id: str) -> list[ApiKeyResponse]:
    tenant = await repository.get_tenant(db, tenant_id)
    if not tenant:
        raise NotFoundError(f"Tenant '{tenant_id}' not found")
    keys = await repository.list_api_keys(db, tenant_id)
    return [ApiKeyResponse.model_validate(k) for k in keys]


async def revoke_api_key(db: AsyncSession, tenant_id: str, key_id: str) -> None:
    key = await repository.get_api_key(db, key_id)
    if not key or key.tenant_id != tenant_id:
        raise NotFoundError(f"API key '{key_id}' not found for tenant '{tenant_id}'")
    key.is_active = False
    await db.flush()
