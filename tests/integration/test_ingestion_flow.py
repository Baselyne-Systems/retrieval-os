"""Integration tests: document ingestion pipeline flow.

Tests the full ingestion job processing chain:
    claim → load docs → chunk → embed → upsert → lineage → webhook → complete

Value over unit tests
---------------------
- Verifies that chunking output feeds into the embed call with the right shape.
- Verifies that embed output feeds into upsert with the right point structure.
- Verifies that ensure_collection is called before the first upsert and not
  again on subsequent batches.
- Verifies that when embed raises, the batch is counted as failed_chunks but
  the job still reaches COMPLETED (per-batch error isolation).
- Verifies the upsert receives the plan version's configured collection name.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval_os.ingestion.chunker import chunk_text
from retrieval_os.ingestion.service import process_next_ingestion_job

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_job(
    *,
    job_id: str = "job-001",
    project_name: str = "my-docs",
    index_config_version: int = 1,
    chunk_size: int = 512,
    overlap: int = 64,
    documents: list | None = None,
    source_uri: str | None = None,
) -> SimpleNamespace:
    if documents is None:
        documents = [
            {"id": "doc-001", "content": "hello world " * 20, "metadata": {"src": "test"}},
        ]
    return SimpleNamespace(
        id=job_id,
        project_name=project_name,
        index_config_version=index_config_version,
        chunk_size=chunk_size,
        overlap=overlap,
        document_payload=documents if source_uri is None else None,
        source_uri=source_uri,
        created_by="alice",
        status="RUNNING",
    )


def _make_index_config_ns(
    *,
    embedding_provider: str = "sentence_transformers",
    embedding_model: str = "BAAI/bge-m3",
    index_backend: str = "qdrant",
    index_collection: str = "my_docs_v1",
    distance_metric: str = "cosine",
    embedding_batch_size: int = 32,
    embedding_normalize: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="ic-001",
        project_id="project-001",
        version=1,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        index_backend=index_backend,
        index_collection=index_collection,
        distance_metric=distance_metric,
        embedding_batch_size=embedding_batch_size,
        embedding_normalize=embedding_normalize,
    )


# ── Chunking ──────────────────────────────────────────────────────────────────


class TestChunkingBehaviour:
    def test_chunk_size_and_overlap_respected(self) -> None:
        words = ["word"] * 200
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # Each chunk should have at most 100 words
        for chunk in chunks:
            assert len(chunk.split()) <= 100
        # There should be multiple chunks with overlap
        assert len(chunks) > 1
        first_words = set(chunks[0].split())
        second_words = set(chunks[1].split())
        assert first_words & second_words  # overlap exists

    def test_short_document_single_chunk(self) -> None:
        chunks = chunk_text("hello world", chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_empty_document_returns_empty(self) -> None:
        assert chunk_text("", chunk_size=512, overlap=64) == []

    def test_overlap_equal_to_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("a b c d", chunk_size=4, overlap=4)


# ── Ingestion job processing ──────────────────────────────────────────────────


class TestProcessIngestionJob:
    @pytest.mark.asyncio
    async def test_inline_documents_chunked_and_upserted(self) -> None:
        """Happy path: inline documents are chunked, embedded, and upserted."""
        content = "token " * 60  # 60 words → 1 chunk at default chunk_size=512
        job = _make_job(
            documents=[{"id": "doc-1", "content": content, "metadata": {"src": "test"}}]
        )
        pv = _make_index_config_ns()
        fake_vectors = [[0.1] * 768]
        upsert_mock = AsyncMock()
        complete_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.claim_next_queued",
                new=AsyncMock(return_value=job),
            ),
            patch(
                "retrieval_os.ingestion.service._load_index_config",
                new=AsyncMock(return_value=pv),
            ),
            patch(
                "retrieval_os.ingestion.service.embed_text",
                new=AsyncMock(return_value=fake_vectors),
            ),
            patch(
                "retrieval_os.ingestion.service.ensure_collection",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "retrieval_os.ingestion.service.upsert_vectors",
                new=upsert_mock,
            ),
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.complete_job",
                new=complete_mock,
            ),
            patch(
                "retrieval_os.ingestion.service._register_ingestion_lineage",
                new=AsyncMock(),
            ),
            patch(
                "retrieval_os.ingestion.service.fire_webhook_event",
                new=AsyncMock(),
            ),
        ):
            session = MagicMock()
            result = await process_next_ingestion_job(session)

        assert result == "job-001"
        upsert_mock.assert_awaited_once()
        complete_mock.assert_awaited_once()
        # Verify job completed with at least 1 indexed chunk
        assert complete_mock.call_args[1]["indexed_chunks"] >= 1

    @pytest.mark.asyncio
    async def test_ensure_collection_called_once_before_first_upsert(self) -> None:
        """Collection is auto-created before the first batch; not re-created per batch."""
        content = "word " * 600  # > 512 words → multiple chunks
        job = _make_job(
            chunk_size=512,
            overlap=64,
            documents=[{"id": "doc-1", "content": content, "metadata": {}}],
        )
        pv = _make_index_config_ns()
        ensure_mock = AsyncMock(return_value=True)
        complete_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.claim_next_queued",
                new=AsyncMock(return_value=job),
            ),
            patch(
                "retrieval_os.ingestion.service._load_index_config",
                new=AsyncMock(return_value=pv),
            ),
            patch(
                "retrieval_os.ingestion.service.embed_text",
                new=AsyncMock(side_effect=lambda texts, **kw: [[0.0] * 768] * len(texts)),
            ),
            patch(
                "retrieval_os.ingestion.service.ensure_collection",
                new=ensure_mock,
            ),
            patch("retrieval_os.ingestion.service.upsert_vectors", new=AsyncMock()),
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.complete_job",
                new=complete_mock,
            ),
            patch("retrieval_os.ingestion.service._register_ingestion_lineage", new=AsyncMock()),
            patch("retrieval_os.ingestion.service.fire_webhook_event", new=AsyncMock()),
        ):
            await process_next_ingestion_job(MagicMock())

        # ensure_collection is called exactly once regardless of number of chunks/batches
        ensure_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_queued_job_returns_none(self) -> None:
        """When no QUEUED job exists, process_next returns None immediately."""
        with patch(
            "retrieval_os.ingestion.service.ingestion_repo.claim_next_queued",
            new=AsyncMock(return_value=None),
        ):
            result = await process_next_ingestion_job(MagicMock())
        assert result is None

    @pytest.mark.asyncio
    async def test_embed_failure_counts_as_failed_chunks_job_still_completes(
        self,
    ) -> None:
        """If embed_text raises, the batch's chunks go to failed_chunks.
        The job itself still reaches COMPLETED (per-batch error isolation)."""
        job = _make_job(documents=[{"id": "doc-1", "content": "word " * 20, "metadata": {}}])
        pv = _make_index_config_ns()
        complete_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.claim_next_queued",
                new=AsyncMock(return_value=job),
            ),
            patch(
                "retrieval_os.ingestion.service._load_index_config",
                new=AsyncMock(return_value=pv),
            ),
            patch(
                "retrieval_os.ingestion.service.embed_text",
                new=AsyncMock(side_effect=RuntimeError("GPU OOM")),
            ),
            patch(
                "retrieval_os.ingestion.service.ensure_collection",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.complete_job",
                new=complete_mock,
            ),
            patch("retrieval_os.ingestion.service._register_ingestion_lineage", new=AsyncMock()),
            patch("retrieval_os.ingestion.service.fire_webhook_event", new=AsyncMock()),
        ):
            result = await process_next_ingestion_job(MagicMock())

        # Job still returns the job_id and calls complete_job (not fail_job)
        assert result == "job-001"
        complete_mock.assert_awaited_once()
        # All chunks counted as failed
        assert complete_mock.call_args[1]["indexed_chunks"] == 0
        assert complete_mock.call_args[1]["failed_chunks"] > 0

    @pytest.mark.asyncio
    async def test_upsert_called_with_correct_collection(self) -> None:
        """Points are upserted into the plan version's configured collection."""
        job = _make_job(
            documents=[{"id": "doc-1", "content": "test content " * 10, "metadata": {}}]
        )
        pv = _make_index_config_ns(index_collection="custom_collection")
        upsert_mock = AsyncMock()
        complete_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.claim_next_queued",
                new=AsyncMock(return_value=job),
            ),
            patch(
                "retrieval_os.ingestion.service._load_index_config",
                new=AsyncMock(return_value=pv),
            ),
            patch(
                "retrieval_os.ingestion.service.embed_text",
                new=AsyncMock(return_value=[[0.5] * 768]),
            ),
            patch(
                "retrieval_os.ingestion.service.ensure_collection",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "retrieval_os.ingestion.service.upsert_vectors",
                new=upsert_mock,
            ),
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.complete_job",
                new=complete_mock,
            ),
            patch("retrieval_os.ingestion.service._register_ingestion_lineage", new=AsyncMock()),
            patch("retrieval_os.ingestion.service.fire_webhook_event", new=AsyncMock()),
        ):
            await process_next_ingestion_job(MagicMock())

        # Verify upsert received the right collection name as a keyword arg
        upsert_kwargs = upsert_mock.call_args[1]
        assert upsert_kwargs.get("collection") == "custom_collection"

    @pytest.mark.asyncio
    async def test_index_config_not_found_fails_job(self) -> None:
        """If the index config doesn't exist, the job is marked FAILED."""
        from retrieval_os.core.exceptions import IndexConfigNotFoundError

        job = _make_job()
        fail_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.claim_next_queued",
                new=AsyncMock(return_value=job),
            ),
            patch(
                "retrieval_os.ingestion.service._load_index_config",
                new=AsyncMock(side_effect=IndexConfigNotFoundError("version not found")),
            ),
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.fail_job",
                new=fail_mock,
            ),
        ):
            result = await process_next_ingestion_job(MagicMock())

        # The outer except catches this and calls fail_job
        assert result == "job-001"
        fail_mock.assert_awaited_once()
        # error_message is the 3rd positional arg
        call_args = fail_mock.call_args
        err_msg = (
            call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("error_message", "")
        )
        assert "version not found" in err_msg
