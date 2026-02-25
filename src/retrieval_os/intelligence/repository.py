"""Database access layer for the Cost Intelligence domain."""

from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.intelligence.models import CostEntry, ModelPricing


class IntelligenceRepository:

    # ── ModelPricing ───────────────────────────────────────────────────────────

    async def get_pricing(
        self,
        session: AsyncSession,
        provider: str,
        model: str,
    ) -> ModelPricing | None:
        """Return current pricing for (provider, model), or None if not found."""
        result = await session.execute(
            select(ModelPricing)
            .where(
                ModelPricing.provider == provider,
                ModelPricing.model == model,
                ModelPricing.valid_until.is_(None),
            )
            .order_by(ModelPricing.valid_from.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_pricing(self, session: AsyncSession) -> list[ModelPricing]:
        """Return all current (valid_until = NULL) pricing entries."""
        result = await session.execute(
            select(ModelPricing)
            .where(ModelPricing.valid_until.is_(None))
            .order_by(ModelPricing.provider, ModelPricing.model)
        )
        return list(result.scalars().all())

    async def add_pricing(
        self, session: AsyncSession, entry: ModelPricing
    ) -> ModelPricing:
        """Add a new pricing entry (expires any previous one for same provider/model)."""
        from datetime import UTC

        # Expire previous pricing for this provider/model
        await session.execute(
            sa.update(ModelPricing)
            .where(
                ModelPricing.provider == entry.provider,
                ModelPricing.model == entry.model,
                ModelPricing.valid_until.is_(None),
            )
            .values(valid_until=datetime.now(UTC))
        )
        session.add(entry)
        await session.flush()
        await session.refresh(entry)
        return entry

    # ── CostEntry ──────────────────────────────────────────────────────────────

    async def upsert_cost_entry(
        self,
        session: AsyncSession,
        *,
        id: str,
        plan_name: str,
        plan_version: int,
        window_start: datetime,
        window_end: datetime,
        provider: str,
        model: str,
        total_queries: int,
        cache_hits: int,
        token_count: int,
        estimated_cost_usd: float,
        now: datetime,
    ) -> None:
        """Insert or update a cost entry for the given window."""
        stmt = pg_insert(CostEntry).values(
            id=id,
            plan_name=plan_name,
            plan_version=plan_version,
            window_start=window_start,
            window_end=window_end,
            provider=provider,
            model=model,
            total_queries=total_queries,
            cache_hits=cache_hits,
            token_count=token_count,
            estimated_cost_usd=estimated_cost_usd,
            created_at=now,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_cost_entry_window",
            set_={
                "total_queries": stmt.excluded.total_queries,
                "cache_hits": stmt.excluded.cache_hits,
                "token_count": stmt.excluded.token_count,
                "estimated_cost_usd": stmt.excluded.estimated_cost_usd,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        await session.execute(stmt)

    async def list_entries(
        self,
        session: AsyncSession,
        plan_name: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> tuple[list[CostEntry], int]:
        q = select(CostEntry)
        if plan_name:
            q = q.where(CostEntry.plan_name == plan_name)
        if since:
            q = q.where(CostEntry.window_start >= since)
        if until:
            q = q.where(CostEntry.window_start < until)
        q = q.order_by(CostEntry.window_start.desc())

        count_q = sa.select(sa.func.count()).select_from(q.subquery())
        total = (await session.execute(count_q)).scalar_one()
        results = await session.execute(q.offset(offset).limit(limit))
        return list(results.scalars().all()), total

    async def get_summary(
        self,
        session: AsyncSession,
        since: datetime,
        until: datetime,
        plan_name: str | None = None,
    ) -> list[dict]:
        """Return per-plan aggregated cost summary for the time window."""
        q = (
            sa.select(
                CostEntry.plan_name,
                sa.func.sum(CostEntry.total_queries).label("total_queries"),
                sa.func.sum(CostEntry.cache_hits).label("cache_hits"),
                sa.func.sum(CostEntry.token_count).label("token_count"),
                sa.func.sum(CostEntry.estimated_cost_usd).label("estimated_cost_usd"),
            )
            .where(CostEntry.window_start >= since, CostEntry.window_start < until)
            .group_by(CostEntry.plan_name)
            .order_by(sa.func.sum(CostEntry.estimated_cost_usd).desc())
        )
        if plan_name:
            q = q.where(CostEntry.plan_name == plan_name)

        result = await session.execute(q)
        return [row._asdict() for row in result]


intel_repo = IntelligenceRepository()
