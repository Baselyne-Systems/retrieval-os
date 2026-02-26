"""Load test: Ingestion deduplication skips re-embedding.

Proves that submitting a second ingestion job for the same
(project_name, index_config_version) skips all embedding and Qdrant upserts,
and that the second job's duplicate_of field points to the first job's id.

Infrastructure required: Postgres + Redis + Qdrant.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from sqlalchemy import text

from retrieval_os.core.database import async_session_factory
from retrieval_os.core.ids import uuid7
from retrieval_os.ingestion.models import IngestionJob, IngestionJobStatus
from retrieval_os.ingestion.repository import ingestion_repo
from retrieval_os.ingestion.service import process_next_ingestion_job


class TestIngestionDedup:
    """Second ingest for the same config must not call embed_text."""

    async def test_second_ingest_skips_embedding(
        self, load_project, load_collection, check_load_infra
    ) -> None:
        """Verify dedup: embed_text is called for job 1, not for job 2.

        Creates two ingestion jobs for the same (project_name, index_config_version=1).
        Runs job 1 with a counting embed stub; then runs job 2 and asserts:
          - embed_text call count == 0 on job 2
          - job 2's indexed_chunks == job 1's indexed_chunks
          - job 2's duplicate_of == job 1's id
        """
        now = datetime.now(UTC)
        docs = [
            {"id": f"d{i}", "content": f"dedup test content word{i} " * 20, "metadata": {}}
            for i in range(5)
        ]

        # Create and run job 1
        job1_id = str(uuid7())
        async with async_session_factory() as session:
            job1 = IngestionJob(
                id=job1_id,
                project_name=load_project,
                index_config_version=1,
                source_uri=None,
                document_payload=docs,
                chunk_size=50,
                overlap=5,
                status=IngestionJobStatus.QUEUED.value,
                created_at=now,
                created_by="dedup-test",
            )
            # Store index_config_id from the existing index config
            async with session.begin():
                result = await session.execute(
                    text(
                        "SELECT ic.id FROM index_configs ic "
                        "JOIN projects p ON ic.project_id = p.id "
                        "WHERE p.name = :name AND ic.version = 1"
                    ),
                    {"name": load_project},
                )
                ic_id = result.scalar_one()
                job1.index_config_id = ic_id
                session.add(job1)

        embed_call_count = 0

        async def counting_embed(texts, **kwargs):
            nonlocal embed_call_count
            embed_call_count += len(texts)
            from tests.load.conftest import DIMS, random_unit_vector

            return [random_unit_vector(DIMS) for _ in texts]

        with (
            patch("retrieval_os.ingestion.service.embed_text", side_effect=counting_embed),
            patch("retrieval_os.ingestion.service.ensure_collection", new=AsyncMock()),
            patch("retrieval_os.ingestion.service.upsert_vectors", new=AsyncMock(return_value=1)),
            patch("retrieval_os.ingestion.service._register_ingestion_lineage", new=AsyncMock()),
            patch("retrieval_os.ingestion.service.fire_webhook_event", new=AsyncMock()),
        ):
            async with async_session_factory() as session:
                result1 = await process_next_ingestion_job(session)
                await session.commit()

        assert result1 == job1_id
        assert embed_call_count > 0, "Job 1 should have called embed_text"

        # Load job 1 stats
        async with async_session_factory() as session:
            j1 = await ingestion_repo.get(session, job1_id)
        assert j1 is not None
        assert j1.status == IngestionJobStatus.COMPLETED.value
        job1_indexed = j1.indexed_chunks

        # Create job 2 for same (project, config_version=1)
        job2_id = str(uuid7())
        embed_call_count = 0

        async with async_session_factory() as session:
            job2 = IngestionJob(
                id=job2_id,
                project_name=load_project,
                index_config_version=1,
                source_uri=None,
                document_payload=docs,
                chunk_size=50,
                overlap=5,
                status=IngestionJobStatus.QUEUED.value,
                created_at=datetime.now(UTC),
                created_by="dedup-test",
                index_config_id=ic_id,
            )
            async with session.begin():
                session.add(job2)

        with (
            patch("retrieval_os.ingestion.service.embed_text", side_effect=counting_embed),
            patch("retrieval_os.ingestion.service.ensure_collection", new=AsyncMock()),
            patch("retrieval_os.ingestion.service.upsert_vectors", new=AsyncMock(return_value=1)),
            patch("retrieval_os.ingestion.service._register_ingestion_lineage", new=AsyncMock()),
            patch("retrieval_os.ingestion.service.fire_webhook_event", new=AsyncMock()),
        ):
            async with async_session_factory() as session:
                result2 = await process_next_ingestion_job(session)
                await session.commit()

        assert result2 == job2_id

        # Verify dedup assertions
        async with async_session_factory() as session:
            j2 = await ingestion_repo.get(session, job2_id)

        assert j2 is not None
        assert j2.status == IngestionJobStatus.COMPLETED.value

        assert embed_call_count == 0, (
            f"Job 2 should not have called embed_text; got {embed_call_count} calls. "
            "Ingestion dedup is not working."
        )
        assert j2.indexed_chunks == job1_indexed, (
            f"Job 2 indexed_chunks ({j2.indexed_chunks}) != job1 ({job1_indexed})"
        )
        assert j2.duplicate_of == job1_id, (
            f"Job 2 duplicate_of={j2.duplicate_of!r}, expected {job1_id!r}"
        )
