"""Database queries for the Tenants domain."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.tenants.models import ApiKey, Tenant


async def get_tenant(db: AsyncSession, tenant_id: str) -> Tenant | None:
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    return result.scalar_one_or_none()


async def get_tenant_by_name(db: AsyncSession, name: str) -> Tenant | None:
    result = await db.execute(select(Tenant).where(Tenant.name == name))
    return result.scalar_one_or_none()


async def list_tenants(db: AsyncSession) -> list[Tenant]:
    result = await db.execute(select(Tenant).order_by(Tenant.created_at.desc()))
    return list(result.scalars().all())


async def get_api_key_by_prefix_and_hash(
    db: AsyncSession, prefix: str, key_hash: str
) -> ApiKey | None:
    """Look up an API key by its prefix and hash.

    The ``tenant`` relationship is eagerly loaded (lazy="selectin" on the model)
    so callers can access ``api_key.tenant.max_requests_per_minute`` without an
    additional query.
    """
    result = await db.execute(
        select(ApiKey).where(
            ApiKey.key_prefix == prefix,
            ApiKey.key_hash == key_hash,
        )
    )
    return result.scalar_one_or_none()


async def list_api_keys(db: AsyncSession, tenant_id: str) -> list[ApiKey]:
    result = await db.execute(
        select(ApiKey).where(ApiKey.tenant_id == tenant_id).order_by(ApiKey.created_at.desc())
    )
    return list(result.scalars().all())


async def get_api_key(db: AsyncSession, key_id: str) -> ApiKey | None:
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_id))
    return result.scalar_one_or_none()
