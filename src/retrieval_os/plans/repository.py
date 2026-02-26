"""Database access layer for the Projects domain.

All queries live here. The service layer calls these and owns the business logic.
"""

import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.plans.models import IndexConfig, Project


class ProjectRepository:
    # ── Project ────────────────────────────────────────────────────────────────

    async def create_project(self, session: AsyncSession, project: Project) -> Project:
        session.add(project)
        await session.flush()
        await session.refresh(project)
        return project

    async def get_by_name(self, session: AsyncSession, name: str) -> Project | None:
        result = await session.execute(select(Project).where(Project.name == name))
        return result.scalar_one_or_none()

    async def get_by_id(self, session: AsyncSession, project_id: uuid.UUID) -> Project | None:
        result = await session.execute(select(Project).where(Project.id == project_id))
        return result.scalar_one_or_none()

    async def list_projects(
        self,
        session: AsyncSession,
        offset: int = 0,
        limit: int = 20,
        include_archived: bool = False,
    ) -> tuple[list[Project], int]:
        q = select(Project)
        if not include_archived:
            q = q.where(Project.is_archived.is_(False))
        q = q.order_by(Project.created_at.desc())

        total_result = await session.execute(select(func.count()).select_from(q.subquery()))
        total = total_result.scalar_one()

        result = await session.execute(q.offset(offset).limit(limit))
        projects = list(result.scalars().all())
        return projects, total

    async def archive(self, session: AsyncSession, project_id: uuid.UUID) -> None:
        await session.execute(
            update(Project).where(Project.id == project_id).values(is_archived=True)
        )

    # ── IndexConfig ────────────────────────────────────────────────────────────

    async def create_index_config(self, session: AsyncSession, config: IndexConfig) -> IndexConfig:
        session.add(config)
        await session.flush()
        await session.refresh(config)
        return config

    async def get_index_config(
        self, session: AsyncSession, project_id: uuid.UUID, version_num: int
    ) -> IndexConfig | None:
        result = await session.execute(
            select(IndexConfig).where(
                IndexConfig.project_id == project_id,
                IndexConfig.version == version_num,
            )
        )
        return result.scalar_one_or_none()

    async def get_index_config_by_id(
        self, session: AsyncSession, config_id: uuid.UUID
    ) -> IndexConfig | None:
        result = await session.execute(select(IndexConfig).where(IndexConfig.id == config_id))
        return result.scalar_one_or_none()

    async def get_index_config_by_config_hash(
        self, session: AsyncSession, project_id: uuid.UUID, config_hash: str
    ) -> IndexConfig | None:
        result = await session.execute(
            select(IndexConfig).where(
                IndexConfig.project_id == project_id,
                IndexConfig.config_hash == config_hash,
            )
        )
        return result.scalar_one_or_none()

    async def list_index_configs(
        self, session: AsyncSession, project_id: uuid.UUID
    ) -> list[IndexConfig]:
        result = await session.execute(
            select(IndexConfig)
            .where(IndexConfig.project_id == project_id)
            .order_by(IndexConfig.version.asc())
        )
        return list(result.scalars().all())

    async def unset_current_index_config(
        self, session: AsyncSession, project_id: uuid.UUID
    ) -> None:
        """Marks all index configs of this project as not current."""
        await session.execute(
            update(IndexConfig).where(IndexConfig.project_id == project_id).values(is_current=False)
        )

    async def get_next_version_number(self, session: AsyncSession, project: Project) -> int:
        """
        Locks the parent project row with SELECT FOR UPDATE so concurrent index
        config creates queue behind each other, guaranteeing monotonic version numbers.
        """
        await session.execute(select(Project).where(Project.id == project.id).with_for_update())
        max_result = await session.execute(
            select(func.coalesce(func.max(IndexConfig.version), 0)).where(
                IndexConfig.project_id == project.id
            )
        )
        return max_result.scalar_one() + 1


project_repo = ProjectRepository()
