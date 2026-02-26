"""Unit tests for the Ingestion domain (chunker, models, schemas, service helpers)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval_os.ingestion.chunker import chunk_text, estimate_chunk_count
from retrieval_os.ingestion.models import IngestionJob, IngestionJobStatus
from retrieval_os.ingestion.schemas import IngestDocumentRequest, IngestRequest

# ── IngestionJobStatus ────────────────────────────────────────────────────────


class TestIngestionJobStatus:
    def test_values(self) -> None:
        assert IngestionJobStatus.QUEUED == "QUEUED"
        assert IngestionJobStatus.RUNNING == "RUNNING"
        assert IngestionJobStatus.COMPLETED == "COMPLETED"
        assert IngestionJobStatus.FAILED == "FAILED"

    def test_four_statuses(self) -> None:
        assert len(IngestionJobStatus) == 4


# ── IngestionJob ORM ──────────────────────────────────────────────────────────


class TestIngestionJobModel:
    def test_constructor_inline(self) -> None:
        now = datetime.now(UTC)
        job = IngestionJob(
            id="j-001",
            project_name="acme",
            index_config_version=1,
            source_uri=None,
            document_payload=[{"id": "d1", "content": "hello", "metadata": {}}],
            chunk_size=512,
            overlap=64,
            status=IngestionJobStatus.QUEUED.value,
            created_at=now,
        )
        assert job.project_name == "acme"
        assert job.status == "QUEUED"
        assert job.document_payload is not None

    def test_constructor_s3(self) -> None:
        now = datetime.now(UTC)
        job = IngestionJob(
            id="j-002",
            project_name="acme",
            index_config_version=2,
            source_uri="s3://bucket/key.jsonl",
            document_payload=None,
            chunk_size=256,
            overlap=32,
            status=IngestionJobStatus.QUEUED.value,
            created_at=now,
        )
        assert job.source_uri == "s3://bucket/key.jsonl"
        assert job.document_payload is None


# ── chunk_text ────────────────────────────────────────────────────────────────


class TestChunkText:
    def test_empty_string(self) -> None:
        assert chunk_text("") == []

    def test_whitespace_only(self) -> None:
        assert chunk_text("   \n\t  ") == []

    def test_single_word(self) -> None:
        result = chunk_text("hello", chunk_size=512, overlap=64)
        assert result == ["hello"]

    def test_short_text_returns_single_chunk(self) -> None:
        text = " ".join([f"word{i}" for i in range(10)])
        result = chunk_text(text, chunk_size=512, overlap=64)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_produces_multiple_chunks(self) -> None:
        words = [f"w{i}" for i in range(1000)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1

    def test_all_words_covered(self) -> None:
        """Every word in the input must appear in at least one chunk."""
        words = [f"word{i}" for i in range(300)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=50, overlap=5)
        all_chunk_words: set[str] = set()
        for c in chunks:
            all_chunk_words.update(c.split())
        assert all_chunk_words == set(words)

    def test_overlap_repeats_words(self) -> None:
        """With overlap > 0 the last *overlap* words of chunk N appear at the
        start of chunk N+1."""
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words)
        overlap = 3
        chunks = chunk_text(text, chunk_size=10, overlap=overlap)
        if len(chunks) >= 2:
            tail = chunks[0].split()[-overlap:]
            head = chunks[1].split()[:overlap]
            assert tail == head

    def test_zero_overlap(self) -> None:
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=10, overlap=0)
        # No word should appear in two consecutive chunks
        for i in range(len(chunks) - 1):
            assert not set(chunks[i].split()) & set(chunks[i + 1].split())

    def test_chunk_size_exact_boundary(self) -> None:
        words = [f"w{i}" for i in range(10)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=10, overlap=0)
        assert len(chunks) == 1
        assert len(chunks[0].split()) == 10

    def test_overlap_equal_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello world", chunk_size=5, overlap=5)

    def test_overlap_greater_than_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("hello world", chunk_size=5, overlap=10)

    def test_each_chunk_respects_chunk_size(self) -> None:
        words = [f"w{i}" for i in range(500)]
        text = " ".join(words)
        chunk_size = 50
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=5):
            assert len(chunk.split()) <= chunk_size

    def test_last_chunk_contains_last_word(self) -> None:
        words = [f"w{i}" for i in range(25)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=10, overlap=2)
        last_word = words[-1]
        assert any(last_word in c for c in chunks)

    def test_single_chunk_when_exactly_chunk_size(self) -> None:
        words = [f"w{i}" for i in range(512)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=512, overlap=0)
        assert len(chunks) == 1

    def test_deterministic(self) -> None:
        text = "the quick brown fox jumps over the lazy dog " * 50
        assert chunk_text(text) == chunk_text(text)


# ── estimate_chunk_count ──────────────────────────────────────────────────────


class TestEstimateChunkCount:
    def test_empty_returns_zero(self) -> None:
        assert estimate_chunk_count("") == 0

    def test_short_returns_one(self) -> None:
        assert estimate_chunk_count("hello world") == 1

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            estimate_chunk_count("text", chunk_size=5, overlap=5)

    def test_positive_for_long_text(self) -> None:
        words = " ".join([f"w{i}" for i in range(1000)])
        count = estimate_chunk_count(words, chunk_size=100, overlap=10)
        assert count > 1


# ── IngestRequest schema ──────────────────────────────────────────────────────


class TestIngestRequest:
    def test_inline_documents_valid(self) -> None:
        req = IngestRequest(
            documents=[IngestDocumentRequest(id="d1", content="hello")],
            index_config_version=1,
        )
        assert req.documents is not None
        assert len(req.documents) == 1
        assert req.source_uri is None

    def test_source_uri_valid(self) -> None:
        req = IngestRequest(
            source_uri="s3://bucket/file.jsonl",
            index_config_version=1,
        )
        assert req.source_uri == "s3://bucket/file.jsonl"
        assert req.documents is None

    def test_neither_raises(self) -> None:
        with pytest.raises(Exception):
            IngestRequest(index_config_version=1)

    def test_both_raises(self) -> None:
        with pytest.raises(Exception):
            IngestRequest(
                documents=[IngestDocumentRequest(id="d1", content="hi")],
                source_uri="s3://bucket/file.jsonl",
                index_config_version=1,
            )

    def test_overlap_ge_chunk_size_raises(self) -> None:
        with pytest.raises(Exception):
            IngestRequest(
                source_uri="s3://bucket/f.jsonl",
                index_config_version=1,
                chunk_size=64,
                overlap=64,
            )

    def test_defaults(self) -> None:
        req = IngestRequest(
            source_uri="s3://bucket/f.jsonl",
            index_config_version=1,
        )
        assert req.chunk_size == 512
        assert req.overlap == 64

    def test_created_by_optional(self) -> None:
        req = IngestRequest(
            source_uri="s3://bucket/f.jsonl",
            index_config_version=1,
        )
        assert req.created_by is None


class TestIngestDocumentRequest:
    def test_valid(self) -> None:
        doc = IngestDocumentRequest(id="doc-1", content="some text")
        assert doc.id == "doc-1"
        assert doc.metadata == {}

    def test_with_metadata(self) -> None:
        doc = IngestDocumentRequest(id="d", content="text", metadata={"source": "web"})
        assert doc.metadata["source"] == "web"

    def test_empty_id_raises(self) -> None:
        with pytest.raises(Exception):
            IngestDocumentRequest(id="", content="text")

    def test_empty_content_raises(self) -> None:
        with pytest.raises(Exception):
            IngestDocumentRequest(id="d", content="")


# ── process_next_ingestion_job ────────────────────────────────────────────────


# ── Ingestion dedup ───────────────────────────────────────────────────────────


class TestIngestionDedup:
    @pytest.mark.asyncio
    async def test_dedup_skips_embed_when_completed_job_exists(self) -> None:
        """When a completed job exists for the same config, embed_text is never called."""
        from retrieval_os.ingestion.service import process_next_ingestion_job

        now = datetime.now(UTC)
        fake_job = IngestionJob(
            id="j-new",
            project_name="acme",
            index_config_version=1,
            source_uri=None,
            document_payload=[{"id": "d1", "content": "content", "metadata": {}}],
            chunk_size=512,
            overlap=64,
            status="RUNNING",
            created_at=now,
        )

        existing_completed = IngestionJob(
            id="j-prior",
            project_name="acme",
            index_config_version=1,
            source_uri=None,
            document_payload=None,
            chunk_size=512,
            overlap=64,
            status="COMPLETED",
            created_at=now,
            total_docs=2,
            total_chunks=10,
            indexed_chunks=9,
            failed_chunks=1,
        )

        fake_pv = MagicMock()
        fake_pv.embedding_provider = "sentence_transformers"
        fake_pv.embedding_model = "all-MiniLM-L6-v2"
        fake_pv.embedding_normalize = True
        fake_pv.embedding_batch_size = 32
        fake_pv.index_backend = "qdrant"
        fake_pv.index_collection = "acme-v1"

        complete_mock = AsyncMock()
        embed_mock = AsyncMock(return_value=[[0.1, 0.2]])
        mock_session = MagicMock()

        with (
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.claim_next_queued",
                new=AsyncMock(return_value=fake_job),
            ),
            patch(
                "retrieval_os.ingestion.service._load_index_config",
                new=AsyncMock(return_value=fake_pv),
            ),
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.get_completed_for_config",
                new=AsyncMock(return_value=existing_completed),
            ),
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.complete_job",
                new=complete_mock,
            ),
            patch("retrieval_os.ingestion.service.embed_text", new=embed_mock),
        ):
            result = await process_next_ingestion_job(mock_session)

        assert result == "j-new"
        embed_mock.assert_not_called()
        complete_mock.assert_awaited_once()
        _, kwargs = complete_mock.call_args
        assert kwargs["duplicate_of"] == "j-prior"
        assert kwargs["indexed_chunks"] == 9

    @pytest.mark.asyncio
    async def test_dedup_proceeds_normally_when_no_prior_job(self) -> None:
        """When no completed job exists, embed_text is called normally."""
        from retrieval_os.ingestion.service import process_next_ingestion_job

        now = datetime.now(UTC)
        fake_job = IngestionJob(
            id="j-first",
            project_name="acme",
            index_config_version=1,
            source_uri=None,
            document_payload=[{"id": "d1", "content": "word " * 10, "metadata": {}}],
            chunk_size=512,
            overlap=64,
            status="RUNNING",
            created_at=now,
        )

        fake_pv = MagicMock()
        fake_pv.embedding_provider = "sentence_transformers"
        fake_pv.embedding_model = "all-MiniLM-L6-v2"
        fake_pv.embedding_normalize = True
        fake_pv.embedding_batch_size = 32
        fake_pv.index_backend = "qdrant"
        fake_pv.index_collection = "acme-v1"

        embed_mock = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock_session = MagicMock()

        with (
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.claim_next_queued",
                new=AsyncMock(return_value=fake_job),
            ),
            patch(
                "retrieval_os.ingestion.service._load_index_config",
                new=AsyncMock(return_value=fake_pv),
            ),
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.get_completed_for_config",
                new=AsyncMock(return_value=None),  # no prior job
            ),
            patch("retrieval_os.ingestion.service.embed_text", new=embed_mock),
            patch("retrieval_os.ingestion.service.upsert_vectors", new=AsyncMock(return_value=1)),
            patch("retrieval_os.ingestion.service._register_ingestion_lineage", new=AsyncMock()),
            patch("retrieval_os.ingestion.service.fire_webhook_event", new=AsyncMock()),
            patch("retrieval_os.ingestion.service.ingestion_repo.complete_job", new=AsyncMock()),
        ):
            result = await process_next_ingestion_job(mock_session)

        assert result == "j-first"
        embed_mock.assert_called()


class TestProcessNextIngestionJob:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_jobs(self) -> None:
        from retrieval_os.ingestion.service import process_next_ingestion_job

        mock_session = MagicMock()
        with patch(
            "retrieval_os.ingestion.service.ingestion_repo.claim_next_queued",
            new=AsyncMock(return_value=None),
        ):
            result = await process_next_ingestion_job(mock_session)

        assert result is None

    @pytest.mark.asyncio
    async def test_processes_inline_job(self) -> None:
        from retrieval_os.ingestion.service import process_next_ingestion_job

        now = datetime.now(UTC)
        fake_job = IngestionJob(
            id="j-001",
            project_name="acme",
            index_config_version=1,
            source_uri=None,
            document_payload=[{"id": "d1", "content": "word " * 10, "metadata": {}}],
            chunk_size=512,
            overlap=64,
            status="RUNNING",
            created_at=now,
        )

        fake_pv = MagicMock()
        fake_pv.embedding_provider = "sentence_transformers"
        fake_pv.embedding_model = "all-MiniLM-L6-v2"
        fake_pv.embedding_normalize = True
        fake_pv.embedding_batch_size = 32
        fake_pv.index_backend = "qdrant"
        fake_pv.index_collection = "acme-v1"

        mock_session = MagicMock()

        with (
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.claim_next_queued",
                new=AsyncMock(return_value=fake_job),
            ),
            patch(
                "retrieval_os.ingestion.service._load_index_config",
                new=AsyncMock(return_value=fake_pv),
            ),
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.get_completed_for_config",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "retrieval_os.ingestion.service.embed_text",
                new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]),
            ),
            patch(
                "retrieval_os.ingestion.service.upsert_vectors",
                new=AsyncMock(return_value=1),
            ),
            patch(
                "retrieval_os.ingestion.service._register_ingestion_lineage",
                new=AsyncMock(),
            ),
            patch(
                "retrieval_os.ingestion.service.fire_webhook_event",
                new=AsyncMock(),
            ),
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.complete_job",
                new=AsyncMock(),
            ),
        ):
            result = await process_next_ingestion_job(mock_session)

        assert result == "j-001"

    @pytest.mark.asyncio
    async def test_marks_failed_on_exception(self) -> None:
        from retrieval_os.ingestion.service import process_next_ingestion_job

        now = datetime.now(UTC)
        fake_job = IngestionJob(
            id="j-err",
            project_name="acme",
            index_config_version=1,
            source_uri=None,
            document_payload=[],
            chunk_size=512,
            overlap=64,
            status="RUNNING",
            created_at=now,
        )

        mock_session = MagicMock()
        fail_mock = AsyncMock()

        with (
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.claim_next_queued",
                new=AsyncMock(return_value=fake_job),
            ),
            patch(
                "retrieval_os.ingestion.service._load_index_config",
                new=AsyncMock(side_effect=RuntimeError("db down")),
            ),
            patch(
                "retrieval_os.ingestion.service.ingestion_repo.fail_job",
                new=fail_mock,
            ),
        ):
            result = await process_next_ingestion_job(mock_session)

        assert result == "j-err"
        fail_mock.assert_awaited_once()
        _, kwargs = fail_mock.call_args
        assert "db down" in kwargs["error_message"]
