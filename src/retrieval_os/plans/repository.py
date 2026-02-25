"""Database access layer for the Plans domain.

All queries live here. The service layer calls these and owns the business logic.
"""

import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.plans.models import PlanVersion, RetrievalPlan


class PlanRepository:
    # ── RetrievalPlan ──────────────────────────────────────────────────────────

    async def create_plan(
        self, session: AsyncSession, plan: RetrievalPlan
    ) -> RetrievalPlan:
        session.add(plan)
        await session.flush()
        await session.refresh(plan)
        return plan

    async def get_by_name(
        self, session: AsyncSession, name: str
    ) -> RetrievalPlan | None:
        result = await session.execute(
            select(RetrievalPlan).where(RetrievalPlan.name == name)
        )
        return result.scalar_one_or_none()

    async def get_by_id(
        self, session: AsyncSession, plan_id: uuid.UUID
    ) -> RetrievalPlan | None:
        result = await session.execute(
            select(RetrievalPlan).where(RetrievalPlan.id == plan_id)
        )
        return result.scalar_one_or_none()

    async def list_plans(
        self,
        session: AsyncSession,
        offset: int = 0,
        limit: int = 20,
        include_archived: bool = False,
    ) -> tuple[list[RetrievalPlan], int]:
        q = select(RetrievalPlan)
        if not include_archived:
            q = q.where(RetrievalPlan.is_archived.is_(False))
        q = q.order_by(RetrievalPlan.created_at.desc())

        total_result = await session.execute(
            select(func.count()).select_from(q.subquery())
        )
        total = total_result.scalar_one()

        result = await session.execute(q.offset(offset).limit(limit))
        plans = list(result.scalars().all())
        return plans, total

    async def archive(self, session: AsyncSession, plan_id: uuid.UUID) -> None:
        await session.execute(
            update(RetrievalPlan)
            .where(RetrievalPlan.id == plan_id)
            .values(is_archived=True)
        )

    # ── PlanVersion ────────────────────────────────────────────────────────────

    async def create_version(
        self, session: AsyncSession, version: PlanVersion
    ) -> PlanVersion:
        session.add(version)
        await session.flush()
        await session.refresh(version)
        return version

    async def get_version(
        self, session: AsyncSession, plan_id: uuid.UUID, version_num: int
    ) -> PlanVersion | None:
        result = await session.execute(
            select(PlanVersion).where(
                PlanVersion.plan_id == plan_id,
                PlanVersion.version == version_num,
            )
        )
        return result.scalar_one_or_none()

    async def get_version_by_config_hash(
        self, session: AsyncSession, plan_id: uuid.UUID, config_hash: str
    ) -> PlanVersion | None:
        result = await session.execute(
            select(PlanVersion).where(
                PlanVersion.plan_id == plan_id,
                PlanVersion.config_hash == config_hash,
            )
        )
        return result.scalar_one_or_none()

    async def list_versions(
        self, session: AsyncSession, plan_id: uuid.UUID
    ) -> list[PlanVersion]:
        result = await session.execute(
            select(PlanVersion)
            .where(PlanVersion.plan_id == plan_id)
            .order_by(PlanVersion.version.asc())
        )
        return list(result.scalars().all())

    async def unset_current_version(
        self, session: AsyncSession, plan_id: uuid.UUID
    ) -> None:
        """Marks all versions of this plan as not current."""
        await session.execute(
            update(PlanVersion)
            .where(PlanVersion.plan_id == plan_id)
            .values(is_current=False)
        )

    async def get_next_version_number(
        self, session: AsyncSession, plan: RetrievalPlan
    ) -> int:
        """
        Locks the parent plan row with SELECT FOR UPDATE so concurrent version
        creates queue behind each other, guaranteeing monotonic version numbers.
        """
        await session.execute(
            select(RetrievalPlan)
            .where(RetrievalPlan.id == plan.id)
            .with_for_update()
        )
        max_result = await session.execute(
            select(func.coalesce(func.max(PlanVersion.version), 0)).where(
                PlanVersion.plan_id == plan.id
            )
        )
        return max_result.scalar_one() + 1


plan_repo = PlanRepository()
