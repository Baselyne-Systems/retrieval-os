"""Database repository for IngestionJob."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.ingestion.models import IngestionJob, IngestionJobStatus


class IngestionRepository:
    async def create(
        self, session: AsyncSession, job: IngestionJob
    ) -> IngestionJob:
        session.add(job)
        await session.flush()
        await session.refresh(job)
        return job

    async def get(
        self, session: AsyncSession, job_id: str
    ) -> IngestionJob | None:
        return await session.get(IngestionJob, job_id)

    async def list_for_plan(
        self,
        session: AsyncSession,
        plan_name: str,
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[IngestionJob], int]:
        count_result = await session.execute(
            select(func.count())
            .select_from(IngestionJob)
            .where(IngestionJob.plan_name == plan_name)
        )
        total: int = count_result.scalar_one()

        result = await session.execute(
            select(IngestionJob)
            .where(IngestionJob.plan_name == plan_name)
            .order_by(IngestionJob.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def claim_next_queued(
        self, session: AsyncSession
    ) -> IngestionJob | None:
        """Claim the oldest QUEUED job using SELECT FOR UPDATE SKIP LOCKED.

        Marks the claimed job as RUNNING before returning so concurrent
        runners never process the same job.
        """
        result = await session.execute(
            select(IngestionJob)
            .where(IngestionJob.status == IngestionJobStatus.QUEUED.value)
            .order_by(IngestionJob.created_at.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = result.scalar_one_or_none()
        if job is None:
            return None

        job.status = IngestionJobStatus.RUNNING.value
        job.started_at = datetime.now(UTC)
        await session.flush()
        return job

    async def complete_job(
        self,
        session: AsyncSession,
        job_id: str,
        *,
        total_docs: int,
        total_chunks: int,
        indexed_chunks: int,
        failed_chunks: int,
    ) -> None:
        job = await self.get(session, job_id)
        if job is None:
            return
        job.status = IngestionJobStatus.COMPLETED.value
        job.total_docs = total_docs
        job.total_chunks = total_chunks
        job.indexed_chunks = indexed_chunks
        job.failed_chunks = failed_chunks
        job.completed_at = datetime.now(UTC)
        await session.flush()

    async def fail_job(
        self, session: AsyncSession, job_id: str, *, error_message: str
    ) -> None:
        job = await self.get(session, job_id)
        if job is None:
            return
        job.status = IngestionJobStatus.FAILED.value
        job.error_message = error_message
        job.completed_at = datetime.now(UTC)
        await session.flush()


ingestion_repo = IngestionRepository()
