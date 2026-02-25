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
    On restart, stale RUNNING jobs are reset to QUEUED after a staleness check.
    """
    interval = settings.eval_job_poll_interval_seconds
    while True:
        try:
            await asyncio.sleep(interval)
            # Phase 6: SELECT FOR UPDATE SKIP LOCKED on eval_jobs WHERE status='QUEUED',
            # set status='RUNNING', run eval, write EvalRun, check regression,
            # set status='COMPLETED' or 'FAILED'.
        except asyncio.CancelledError:
            log.info("background.eval_job_runner.stopped")
            raise
        except Exception:
            log.exception("background.eval_job_runner.error")


async def cost_aggregator() -> None:
    """
    Aggregates UsageRecords into CostEntries once per hour.
    Idempotent: records already aggregated in a cost_entry window are skipped.
    """
    interval = settings.cost_aggregator_interval_seconds
    while True:
        try:
            await asyncio.sleep(interval)
            # Phase 7: group usage_records by (plan_version_id, hour, model),
            # compute cost from model_pricing table, upsert into cost_entries.
        except asyncio.CancelledError:
            log.info("background.cost_aggregator.stopped")
            raise
        except Exception:
            log.exception("background.cost_aggregator.error")
