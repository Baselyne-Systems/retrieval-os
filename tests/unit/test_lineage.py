"""Unit tests for the Lineage domain (no live DB or Redis)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from retrieval_os.lineage.models import ArtifactType, EdgeRelationship, LineageArtifact

# ── ArtifactType enum ─────────────────────────────────────────────────────────


class TestArtifactType:
    def test_all_types_exist(self) -> None:
        values = {t.value for t in ArtifactType}
        assert values == {"DATASET_SNAPSHOT", "EMBEDDING_ARTIFACT", "INDEX_ARTIFACT"}

    def test_values_are_strings(self) -> None:
        for t in ArtifactType:
            assert isinstance(t.value, str)

    def test_str_comparison(self) -> None:
        assert ArtifactType.DATASET_SNAPSHOT == "DATASET_SNAPSHOT"
        assert ArtifactType.EMBEDDING_ARTIFACT == "EMBEDDING_ARTIFACT"
        assert ArtifactType.INDEX_ARTIFACT == "INDEX_ARTIFACT"


# ── EdgeRelationship enum ─────────────────────────────────────────────────────


class TestEdgeRelationship:
    def test_all_relationships_exist(self) -> None:
        values = {r.value for r in EdgeRelationship}
        assert values == {"produced_from", "derived_from", "deployed_as"}

    def test_values_are_lowercase(self) -> None:
        for r in EdgeRelationship:
            assert r.value == r.value.lower()


# ── S3 URI helpers ────────────────────────────────────────────────────────────


class TestS3URIHelpers:
    def test_is_s3_uri_true_for_s3(self) -> None:
        from retrieval_os.lineage.service import _is_s3_uri

        assert _is_s3_uri("s3://my-bucket/key/path.jsonl.gz")

    def test_is_s3_uri_true_for_s3a(self) -> None:
        from retrieval_os.lineage.service import _is_s3_uri

        assert _is_s3_uri("s3a://my-bucket/emr/embeddings/")

    def test_is_s3_uri_true_for_s3n(self) -> None:
        from retrieval_os.lineage.service import _is_s3_uri

        assert _is_s3_uri("s3n://legacy-bucket/data/")

    def test_is_s3_uri_false_for_qdrant(self) -> None:
        from retrieval_os.lineage.service import _is_s3_uri

        assert not _is_s3_uri("qdrant://my-collection")

    def test_is_s3_uri_false_for_pgvector(self) -> None:
        from retrieval_os.lineage.service import _is_s3_uri

        assert not _is_s3_uri("pgvector://my-table")

    def test_is_s3_uri_false_for_http(self) -> None:
        from retrieval_os.lineage.service import _is_s3_uri

        assert not _is_s3_uri("https://example.com/file.bin")

    def test_parse_s3_uri_simple(self) -> None:
        from retrieval_os.lineage.service import _parse_s3_uri

        bucket, key = _parse_s3_uri("s3://my-bucket/datasets/v1.jsonl.gz")
        assert bucket == "my-bucket"
        assert key == "datasets/v1.jsonl.gz"

    def test_parse_s3_uri_nested_key(self) -> None:
        from retrieval_os.lineage.service import _parse_s3_uri

        bucket, key = _parse_s3_uri("s3://acme/embeddings/plan/v3/chunks.npy")
        assert bucket == "acme"
        assert key == "embeddings/plan/v3/chunks.npy"

    def test_parse_s3_uri_root_key(self) -> None:
        from retrieval_os.lineage.service import _parse_s3_uri

        bucket, key = _parse_s3_uri("s3://bucket/file.bin")
        assert bucket == "bucket"
        assert key == "file.bin"


# ── Schemas ───────────────────────────────────────────────────────────────────


class TestRegisterArtifactRequest:
    def test_valid_dataset_snapshot(self) -> None:
        from retrieval_os.lineage.schemas import RegisterArtifactRequest

        req = RegisterArtifactRequest(
            artifact_type=ArtifactType.DATASET_SNAPSHOT,
            name="wiki-en",
            version="2026-02-25",
            storage_uri="s3://bucket/datasets/wiki-en/2026-02-25.jsonl.gz",
            created_by="alice",
        )
        assert req.artifact_type == ArtifactType.DATASET_SNAPSHOT
        assert req.content_hash is None
        assert req.metadata is None

    def test_valid_with_content_hash(self) -> None:
        from retrieval_os.lineage.schemas import RegisterArtifactRequest

        req = RegisterArtifactRequest(
            artifact_type=ArtifactType.EMBEDDING_ARTIFACT,
            name="wiki-en-embed",
            version="v1",
            storage_uri="s3://bucket/embeddings/wiki-en/v1/",
            content_hash="a" * 64,
            created_by="bob",
        )
        assert req.content_hash == "a" * 64

    def test_content_hash_must_be_64_chars(self) -> None:
        from retrieval_os.lineage.schemas import RegisterArtifactRequest

        with pytest.raises(Exception):
            RegisterArtifactRequest(
                artifact_type=ArtifactType.DATASET_SNAPSHOT,
                name="x",
                version="v1",
                storage_uri="s3://bucket/x",
                content_hash="tooshort",
                created_by="alice",
            )

    def test_name_must_not_be_empty(self) -> None:
        from retrieval_os.lineage.schemas import RegisterArtifactRequest

        with pytest.raises(Exception):
            RegisterArtifactRequest(
                artifact_type=ArtifactType.INDEX_ARTIFACT,
                name="",
                version="v1",
                storage_uri="qdrant://my-collection",
                created_by="alice",
            )

    def test_storage_uri_must_not_be_empty(self) -> None:
        from retrieval_os.lineage.schemas import RegisterArtifactRequest

        with pytest.raises(Exception):
            RegisterArtifactRequest(
                artifact_type=ArtifactType.INDEX_ARTIFACT,
                name="my-index",
                version="v1",
                storage_uri="",
                created_by="alice",
            )


class TestCreateEdgeRequest:
    def test_valid_edge_request(self) -> None:
        from retrieval_os.lineage.schemas import CreateEdgeRequest

        req = CreateEdgeRequest(
            parent_artifact_id="parent-uuid",
            child_artifact_id="child-uuid",
            relationship_type=EdgeRelationship.PRODUCED_FROM,
            created_by="alice",
        )
        assert req.relationship_type == EdgeRelationship.PRODUCED_FROM

    def test_parent_id_must_not_be_empty(self) -> None:
        from retrieval_os.lineage.schemas import CreateEdgeRequest

        with pytest.raises(Exception):
            CreateEdgeRequest(
                parent_artifact_id="",
                child_artifact_id="child-uuid",
                relationship_type=EdgeRelationship.DERIVED_FROM,
                created_by="alice",
            )


class TestArtifactWithEdgesResponse:
    def test_defaults_to_empty_lists(self) -> None:
        from retrieval_os.lineage.schemas import ArtifactWithEdgesResponse

        now = datetime.now(UTC)
        resp = ArtifactWithEdgesResponse(
            id="some-id",
            artifact_type="DATASET_SNAPSHOT",
            name="wiki-en",
            version="v1",
            storage_uri="s3://bucket/x",
            content_hash=None,
            metadata=None,
            created_at=now,
            created_by="alice",
        )
        assert resp.parents == []
        assert resp.children == []


class TestLineageGraphResponse:
    def test_empty_graph(self) -> None:
        from retrieval_os.lineage.schemas import LineageGraphResponse

        resp = LineageGraphResponse(plan_name="my-plan", artifacts=[], edges=[])
        assert resp.plan_name == "my-plan"
        assert resp.artifacts == []
        assert resp.edges == []


class TestOrphansResponse:
    def test_zero_orphans(self) -> None:
        from retrieval_os.lineage.schemas import OrphansResponse

        resp = OrphansResponse(total=0, artifacts=[])
        assert resp.total == 0


# ── DAG cycle detection ───────────────────────────────────────────────────────


class TestWouldCreateCycle:
    @pytest.mark.asyncio
    async def test_self_loop_is_cycle(self) -> None:
        from retrieval_os.lineage.dag import would_create_cycle

        mock_session = AsyncMock()
        result = await would_create_cycle(mock_session, "artifact-A", "artifact-A")
        assert result is True

    @pytest.mark.asyncio
    async def test_no_cycle_when_no_ancestors(self) -> None:
        from retrieval_os.lineage.dag import would_create_cycle

        mock_session = AsyncMock()
        with patch(
            "retrieval_os.lineage.dag.lineage_repo.get_ancestors",
            AsyncMock(return_value=[]),
        ):
            result = await would_create_cycle(mock_session, "parent-A", "child-B")
        assert result is False

    @pytest.mark.asyncio
    async def test_cycle_detected_when_child_is_ancestor(self) -> None:
        from retrieval_os.lineage.dag import would_create_cycle

        mock_session = AsyncMock()
        # If child-B is already an ancestor of parent-A, adding parent-A → child-B
        # would create a cycle.
        with patch(
            "retrieval_os.lineage.dag.lineage_repo.get_ancestors",
            AsyncMock(
                return_value=[
                    {"artifact_id": "child-B", "depth": 1},
                    {"artifact_id": "grandparent", "depth": 2},
                ]
            ),
        ):
            result = await would_create_cycle(mock_session, "parent-A", "child-B")
        assert result is True

    @pytest.mark.asyncio
    async def test_no_cycle_when_child_not_in_ancestors(self) -> None:
        from retrieval_os.lineage.dag import would_create_cycle

        mock_session = AsyncMock()
        with patch(
            "retrieval_os.lineage.dag.lineage_repo.get_ancestors",
            AsyncMock(
                return_value=[
                    {"artifact_id": "some-other-node", "depth": 1},
                ]
            ),
        ):
            result = await would_create_cycle(mock_session, "parent-A", "child-B")
        assert result is False


# ── compute_dag_depth ─────────────────────────────────────────────────────────


class TestComputeDagDepth:
    @pytest.mark.asyncio
    async def test_leaf_has_depth_zero(self) -> None:
        from retrieval_os.lineage.dag import compute_dag_depth

        mock_session = AsyncMock()
        with patch(
            "retrieval_os.lineage.dag.lineage_repo.get_descendants",
            AsyncMock(return_value=[]),
        ):
            depth = await compute_dag_depth(mock_session, "leaf-id")
        assert depth == 0

    @pytest.mark.asyncio
    async def test_depth_is_max_descendant_depth(self) -> None:
        from retrieval_os.lineage.dag import compute_dag_depth

        mock_session = AsyncMock()
        with patch(
            "retrieval_os.lineage.dag.lineage_repo.get_descendants",
            AsyncMock(
                return_value=[
                    {"artifact_id": "child", "depth": 1},
                    {"artifact_id": "grandchild", "depth": 2},
                    {"artifact_id": "great-grandchild", "depth": 3},
                ]
            ),
        ):
            depth = await compute_dag_depth(mock_session, "root-id")
        assert depth == 3


# ── LineageArtifact ORM model ─────────────────────────────────────────────────


class TestLineageArtifactModel:
    def test_constructor_sets_fields(self) -> None:
        now = datetime.now(UTC)
        artifact = LineageArtifact(
            id="test-uuid",
            artifact_type=ArtifactType.DATASET_SNAPSHOT.value,
            name="wiki-en",
            version="2026-02-25",
            storage_uri="s3://bucket/datasets/wiki-en.jsonl.gz",
            content_hash=None,
            artifact_metadata={"chunk_count": 1000},
            created_at=now,
            created_by="alice",
        )
        assert artifact.name == "wiki-en"
        assert artifact.artifact_type == "DATASET_SNAPSHOT"
        assert artifact.artifact_metadata == {"chunk_count": 1000}
        assert artifact.content_hash is None

    def test_qdrant_uri_not_s3(self) -> None:
        from retrieval_os.lineage.service import _is_s3_uri

        now = datetime.now(UTC)
        artifact = LineageArtifact(
            id="idx-uuid",
            artifact_type=ArtifactType.INDEX_ARTIFACT.value,
            name="wiki-en-index",
            version="v1",
            storage_uri="qdrant://wiki-en-v1",
            content_hash=None,
            metadata=None,
            created_at=now,
            created_by="alice",
        )
        # Index artifacts use qdrant:// — should not trigger S3 verification
        assert not _is_s3_uri(artifact.storage_uri)
