"""Database access layer for the Evaluation Engine domain."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.evaluations.models import EvalJob, EvalJobStatus


class EvaluationRepository:

    async def create_job(self, session: AsyncSession, job: EvalJob) -> EvalJob:
        session.add(job)
        await session.flush()
        await session.refresh(job)
        return job

    async def get_job(self, session: AsyncSession, job_id: str) -> EvalJob | None:
        result = await session.execute(
            select(EvalJob).where(EvalJob.id == job_id)
        )
        return result.scalar_one_or_none()

    async def list_jobs(
        self,
        session: AsyncSession,
        plan_name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[EvalJob], int]:
        q = select(EvalJob)
        if plan_name:
            q = q.where(EvalJob.plan_name == plan_name)
        q = q.order_by(EvalJob.created_at.desc())

        count_q = sa.select(sa.func.count()).select_from(q.subquery())
        total = (await session.execute(count_q)).scalar_one()

        results = await session.execute(q.offset(offset).limit(limit))
        return list(results.scalars().all()), total

    async def claim_next_queued(self, session: AsyncSession) -> EvalJob | None:
        """SELECT FOR UPDATE SKIP LOCKED on the oldest QUEUED job.

        Sets status to RUNNING and returns the job. The caller must commit.
        Returns None if no QUEUED jobs exist.
        """
        from datetime import UTC, datetime

        result = await session.execute(
            select(EvalJob)
            .where(EvalJob.status == EvalJobStatus.QUEUED)
            .order_by(EvalJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = result.scalar_one_or_none()
        if job is None:
            return None

        now = datetime.now(UTC)
        await session.execute(
            update(EvalJob)
            .where(EvalJob.id == job.id)
            .values(status=EvalJobStatus.RUNNING, started_at=now)
        )
        await session.refresh(job)
        return job

    async def complete_job(
        self,
        session: AsyncSession,
        job_id: str,
        *,
        recall_at_1: float,
        recall_at_3: float,
        recall_at_5: float,
        recall_at_10: float,
        mrr: float,
        ndcg_at_5: float,
        ndcg_at_10: float,
        total_queries: int,
        failed_queries: int,
        regression_detected: bool,
        regression_detail: list,
    ) -> None:
        from datetime import UTC, datetime

        await session.execute(
            update(EvalJob)
            .where(EvalJob.id == job_id)
            .values(
                status=EvalJobStatus.COMPLETED,
                completed_at=datetime.now(UTC),
                recall_at_1=recall_at_1,
                recall_at_3=recall_at_3,
                recall_at_5=recall_at_5,
                recall_at_10=recall_at_10,
                mrr=mrr,
                ndcg_at_5=ndcg_at_5,
                ndcg_at_10=ndcg_at_10,
                total_queries=total_queries,
                failed_queries=failed_queries,
                regression_detected=regression_detected,
                regression_detail=regression_detail or None,
            )
        )

    async def fail_job(
        self,
        session: AsyncSession,
        job_id: str,
        *,
        error_message: str,
        total_queries: int | None = None,
        failed_queries: int | None = None,
    ) -> None:
        from datetime import UTC, datetime

        await session.execute(
            update(EvalJob)
            .where(EvalJob.id == job_id)
            .values(
                status=EvalJobStatus.FAILED,
                completed_at=datetime.now(UTC),
                error_message=error_message,
                total_queries=total_queries,
                failed_queries=failed_queries,
            )
        )

    async def get_latest_completed_for_plan(
        self,
        session: AsyncSession,
        plan_name: str,
        exclude_job_id: str | None = None,
    ) -> EvalJob | None:
        """Return the most recently completed job for a plan (for regression comparison)."""
        q = (
            select(EvalJob)
            .where(
                EvalJob.plan_name == plan_name,
                EvalJob.status == EvalJobStatus.COMPLETED,
            )
            .order_by(EvalJob.completed_at.desc())
            .limit(1)
        )
        if exclude_job_id:
            q = q.where(EvalJob.id != exclude_job_id)
        result = await session.execute(q)
        return result.scalar_one_or_none()


eval_repo = EvaluationRepository()
