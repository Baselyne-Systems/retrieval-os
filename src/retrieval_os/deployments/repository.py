"""Database access layer for the Deployments domain."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.deployments.models import Deployment, DeploymentStatus


class DeploymentRepository:
    async def create(self, session: AsyncSession, deployment: Deployment) -> Deployment:
        session.add(deployment)
        await session.flush()
        await session.refresh(deployment)
        return deployment

    async def get_by_id(self, session: AsyncSession, deployment_id: str) -> Deployment | None:
        result = await session.execute(select(Deployment).where(Deployment.id == deployment_id))
        return result.scalar_one_or_none()

    async def get_active_for_plan(self, session: AsyncSession, plan_name: str) -> Deployment | None:
        """Return the ACTIVE or ROLLING_OUT deployment for a plan, if any."""
        result = await session.execute(
            select(Deployment).where(
                Deployment.plan_name == plan_name,
                Deployment.status.in_(
                    [DeploymentStatus.ACTIVE.value, DeploymentStatus.ROLLING_OUT.value]
                ),
            )
        )
        return result.scalar_one_or_none()

    async def list_for_plan(
        self, session: AsyncSession, plan_name: str, limit: int = 20
    ) -> list[Deployment]:
        result = await session.execute(
            select(Deployment)
            .where(Deployment.plan_name == plan_name)
            .order_by(Deployment.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        session: AsyncSession,
        deployment_id: str,
        status: str,
        **extra_fields: object,
    ) -> None:
        values = {"status": status, **extra_fields}
        await session.execute(
            update(Deployment).where(Deployment.id == deployment_id).values(**values)
        )

    async def list_rolling_out(self, session: AsyncSession) -> list[Deployment]:
        """Return all deployments in ROLLING_OUT status (for the rollout stepper)."""
        result = await session.execute(
            select(Deployment).where(Deployment.status == DeploymentStatus.ROLLING_OUT.value)
        )
        return list(result.scalars().all())

    async def list_live(self, session: AsyncSession) -> list[Deployment]:
        """Return all ACTIVE and ROLLING_OUT deployments across every plan.

        Used by the rollback watchdog to check guard rails.
        """
        result = await session.execute(
            select(Deployment).where(
                Deployment.status.in_(
                    [DeploymentStatus.ACTIVE.value, DeploymentStatus.ROLLING_OUT.value]
                )
            )
        )
        return list(result.scalars().all())


deployment_repo = DeploymentRepository()
