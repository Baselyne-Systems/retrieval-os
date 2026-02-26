"""E2E: Concurrency and database-level safety

Production failure modes proven here:

1. Concurrent index config creation — SELECT FOR UPDATE on the project row
   serialises version counter assignment so no two configs share a version number,
   regardless of how many coroutines race to create one.

2. Duplicate config hash detection under concurrency — when 5 goroutines
   submit the exact same index config simultaneously, the unique constraint on
   (project_id, config_hash) guarantees exactly one new row is inserted; the rest
   receive DuplicateConfigError.

3. Deployment exclusivity — attempting to create a second deployment while one
   is already ACTIVE raises ConflictError. Two live deployments for the same
   project would cause split-brain serving: queries would non-deterministically
   use different index versions.

4. Gradual rollout convergence — the rollout stepper must advance traffic weight
   monotonically and terminate at exactly 1.0, never overshooting.

5. Ingestion SKIP LOCKED — two concurrent workers claiming QUEUED jobs must each
   claim a different job. SELECT FOR UPDATE SKIP LOCKED prevents double-processing.
"""

from __future__ import annotations

import asyncio

import pytest

from retrieval_os.core.database import async_session_factory
from retrieval_os.core.exceptions import ConflictError, DuplicateConfigError
from retrieval_os.deployments.models import DeploymentStatus
from retrieval_os.deployments.repository import deployment_repo
from retrieval_os.deployments.schemas import CreateDeploymentRequest
from retrieval_os.deployments.service import create_deployment, step_rolling_deployments
from retrieval_os.ingestion.models import IngestionJobStatus
from retrieval_os.ingestion.repository import ingestion_repo
from retrieval_os.ingestion.schemas import IngestRequest
from retrieval_os.ingestion.service import create_ingestion_job
from retrieval_os.plans.schemas import CreateIndexConfigRequest, IndexConfigInput
from retrieval_os.plans.service import create_index_config
from tests.e2e.conftest import setup_deployment, setup_project_with_config

# ── Index config versioning ────────────────────────────────────────────────────


class TestIndexConfigVersioning:
    async def test_concurrent_unique_configs_get_unique_versions(self, project_name: str) -> None:
        """5 concurrent index config creates with distinct content each receive
        a unique, monotonically assigned version number.

        The SELECT FOR UPDATE on the project row in get_next_version_number
        serialises concurrent creates into a queue, so no two coroutines can
        both read the same max-version and assign the same next version.
        """
        await setup_project_with_config(project_name)

        async def make_config(suffix: str) -> int:
            async with async_session_factory() as session:
                req = CreateIndexConfigRequest(
                    config=IndexConfigInput(
                        embedding_provider="sentence_transformers",
                        embedding_model="BAAI/bge-m3",
                        embedding_dimensions=1024,
                        # Different collection per variant → different hash
                        index_collection=f"{project_name}-{suffix}",
                        index_backend="qdrant",
                        distance_metric="cosine",
                    ),
                    created_by="e2e",
                )
                result = await create_index_config(session, project_name, req)
                await session.commit()
                return result.version

        versions = await asyncio.gather(*[make_config(f"variant-{i}") for i in range(5)])

        assert len(set(versions)) == 5, (
            f"Duplicate version numbers assigned under concurrency: {sorted(versions)}"
        )
        assert sorted(versions) == list(range(2, 7)), (
            f"Expected versions 2–6 (monotonic), got {sorted(versions)}"
        )

    async def test_concurrent_identical_new_config_exactly_one_wins(
        self, project_name: str
    ) -> None:
        """5 concurrent creates of an identical NEW config (not present as v1):
        exactly 1 coroutine wins and inserts version 2; the other 4 receive
        DuplicateConfigError from the unique constraint on config_hash.

        This is the defence against accidental re-indexing: the same index build
        config can only ever occupy one slot in the version history.
        """
        await setup_project_with_config(project_name)

        # Config that differs from v1 (different collection → different hash)
        new_collection = f"{project_name}-new"

        async def try_create() -> str:
            async with async_session_factory() as session:
                req = CreateIndexConfigRequest(
                    config=IndexConfigInput(
                        embedding_provider="sentence_transformers",
                        embedding_model="BAAI/bge-m3",
                        embedding_dimensions=1024,
                        index_collection=new_collection,
                        index_backend="qdrant",
                        distance_metric="cosine",
                    ),
                    created_by="e2e",
                )
                try:
                    await create_index_config(session, project_name, req)
                    await session.commit()
                    return "ok"
                except DuplicateConfigError:
                    await session.rollback()
                    return "duplicate"
                except Exception:
                    await session.rollback()
                    return "other"

        results = await asyncio.gather(*[try_create() for _ in range(5)])
        ok_count = results.count("ok")
        dup_count = results.count("duplicate")

        assert ok_count == 1, (
            f"Expected exactly 1 success under concurrency, got {ok_count}. "
            f"Results: {list(results)}"
        )
        assert dup_count == 4, (
            f"Expected 4 DuplicateConfigErrors, got {dup_count}. Results: {list(results)}"
        )


# ── Deployment state machine ───────────────────────────────────────────────────


class TestDeploymentStateMachine:
    async def test_second_deployment_blocked_while_first_active(self, project_name: str) -> None:
        """Creating a second deployment while one is ACTIVE must raise ConflictError.

        Two live deployments on the same project would split traffic between
        different index versions, making quality metrics meaningless.
        """
        await setup_project_with_config(project_name)
        await setup_deployment(project_name)  # first deployment → ACTIVE

        with pytest.raises(ConflictError, match="already has an active deployment"):
            async with async_session_factory() as session:
                await create_deployment(
                    session,
                    project_name,
                    CreateDeploymentRequest(index_config_version=1, created_by="e2e"),
                )
                await session.commit()

    async def test_gradual_rollout_terminates_at_exactly_full_traffic(
        self, project_name: str
    ) -> None:
        """A 25%-per-step gradual rollout must reach exactly traffic_weight=1.0
        in the DB after 4 steps and transition from ROLLING_OUT to ACTIVE.

        Floating-point accumulation (4 × 0.25 = 1.0000...0002 in some runtimes)
        is prevented by the min(1.0, ...) cap in the stepper — this test verifies
        the cap is working against a real Postgres row.
        """
        await setup_project_with_config(project_name)

        async with async_session_factory() as session:
            dep = await create_deployment(
                session,
                project_name,
                CreateDeploymentRequest(
                    index_config_version=1,
                    rollout_step_percent=25.0,
                    rollout_step_interval_seconds=10,
                    created_by="e2e",
                ),
            )
            await session.commit()
            dep_id = dep.id

        assert dep.status == DeploymentStatus.ROLLING_OUT.value

        # Advance through all 4 steps
        for _ in range(4):
            async with async_session_factory() as session:
                advanced = await step_rolling_deployments(session)
                await session.commit()
                assert advanced >= 1

        async with async_session_factory() as session:
            final = await deployment_repo.get_by_id(session, dep_id)
        assert final is not None
        assert final.traffic_weight == 1.0, (
            f"Expected traffic_weight=1.0, got {final.traffic_weight} — "
            "float accumulation cap not working"
        )
        assert final.status == DeploymentStatus.ACTIVE.value, (
            f"Expected ACTIVE after full ramp, got {final.status}"
        )


# ── Ingestion SKIP LOCKED ──────────────────────────────────────────────────────


class TestIngestionSkipLocked:
    async def test_two_workers_claim_different_jobs(self, project_name: str) -> None:
        """Two concurrent workers using SELECT FOR UPDATE SKIP LOCKED must each
        claim a distinct QUEUED job — no job is processed twice.

        Without SKIP LOCKED, worker B would block on worker A's lock and both
        would attempt to process the same job when A commits.
        """
        await setup_project_with_config(project_name)

        # Create 2 QUEUED jobs
        async with async_session_factory() as session:
            for i in range(2):
                await create_ingestion_job(
                    session,
                    project_name,
                    IngestRequest(
                        source_uri=f"s3://e2e-test/docs-{i}.jsonl",
                        index_config_version=1,
                        created_by="e2e",
                    ),
                )
            await session.commit()

        async def claim_job() -> str | None:
            async with async_session_factory() as session:
                job = await ingestion_repo.claim_next_queued(session)
                await session.commit()
                return job.id if job else None

        claimed = await asyncio.gather(claim_job(), claim_job())
        job_ids = [jid for jid in claimed if jid is not None]

        assert len(job_ids) == 2, (
            f"Expected both jobs claimed, got {len(job_ids)} — one worker may have missed a job"
        )
        assert job_ids[0] != job_ids[1], (
            "Both workers claimed the same job — SKIP LOCKED not working"
        )

        # Verify both jobs are RUNNING in the DB
        async with async_session_factory() as session:
            for jid in job_ids:
                job = await ingestion_repo.get(session, jid)
                assert job is not None
                assert job.status == IngestionJobStatus.RUNNING.value, (
                    f"Job {jid} should be RUNNING, got {job.status}"
                )
