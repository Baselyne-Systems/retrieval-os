"""Asyncio background task loops wired into the FastAPI lifespan.

Each loop is a stub in Phase 1. Later phases activate them:
  - rollback_watchdog  → Phase 4
  - rollout_stepper    → Phase 4
  - eval_job_runner    → Phase 6
  - cost_aggregator    → Phase 7

The loops handle CancelledError cleanly so graceful shutdown works.
Any unexpected exception is logged and the loop continues — a crashed
background task should not take down the API process.
"""

import asyncio

import structlog

from retrieval_os.core.config import settings
from retrieval_os.core.database import async_session_factory

log = structlog.get_logger(__name__)


async def rollback_watchdog() -> None:
    """
    Monitors active deployments every N seconds.
    Triggers automatic rollback when error_rate, latency_p99, or recall
    exceeds the thresholds configured on each Deployment record.
    Phase 6 populates the eval metrics this watchdog reads.
    """
    from retrieval_os.deployments.service import check_rollback_thresholds

    interval = settings.rollback_watchdog_interval_seconds
    while True:
        try:
            await asyncio.sleep(interval)
            async with async_session_factory() as session:
                triggered = await check_rollback_thresholds(session)
                if triggered:
                    log.info(
                        "background.rollback_watchdog.triggered",
                        count=triggered,
                    )
        except asyncio.CancelledError:
            log.info("background.rollback_watchdog.stopped")
            raise
        except Exception:
            log.exception("background.rollback_watchdog.error")


async def rollout_stepper() -> None:
    """
    Advances gradual deployments one step at their configured interval.
    Increments traffic_weight by rollout_step_percent until 100%, then
    transitions status to ACTIVE.
    """
    from retrieval_os.deployments.service import step_rolling_deployments

    interval = settings.rollout_stepper_interval_seconds
    while True:
        try:
            await asyncio.sleep(interval)
            async with async_session_factory() as session:
                advanced = await step_rolling_deployments(session)
                await session.commit()
                if advanced:
                    log.info(
                        "background.rollout_stepper.stepped",
                        deployments_advanced=advanced,
                    )
        except asyncio.CancelledError:
            log.info("background.rollout_stepper.stopped")
            raise
        except Exception:
            log.exception("background.rollout_stepper.error")


async def eval_job_runner() -> None:
    """
    Drains QUEUED eval jobs from the eval_jobs table.
    Processes one job at a time to avoid overwhelming embedding/index backends.
    Uses SELECT FOR UPDATE SKIP LOCKED so multiple replicas never duplicate work.
    """
    from retrieval_os.evaluations.service import process_next_eval_job

    interval = settings.eval_job_poll_interval_seconds
    while True:
        try:
            await asyncio.sleep(interval)
            async with async_session_factory() as session:
                job_id = await process_next_eval_job(session)
                await session.commit()
                if job_id:
                    log.info(
                        "background.eval_job_runner.processed",
                        job_id=job_id,
                    )
        except asyncio.CancelledError:
            log.info("background.eval_job_runner.stopped")
            raise
        except Exception:
            log.exception("background.eval_job_runner.error")


async def cost_aggregator() -> None:
    """
    Aggregates usage_records into cost_entries in 1-hour windows.
    Processes the last 48 hours on each run (idempotent via upsert).
    Runs once per hour (cost_aggregator_interval_seconds).
    """
    from retrieval_os.intelligence.aggregator import aggregate_usage_costs

    interval = settings.cost_aggregator_interval_seconds
    while True:
        try:
            await asyncio.sleep(interval)
            async with async_session_factory() as session:
                upserted = await aggregate_usage_costs(session)
                await session.commit()
                if upserted:
                    log.info(
                        "background.cost_aggregator.completed",
                        windows_upserted=upserted,
                    )
        except asyncio.CancelledError:
            log.info("background.cost_aggregator.stopped")
            raise
        except Exception:
            log.exception("background.cost_aggregator.error")
