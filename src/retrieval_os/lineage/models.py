"""SQLAlchemy ORM models for the Lineage domain.

The lineage DAG tracks the provenance chain:
  DatasetSnapshot → EmbeddingArtifact → IndexArtifact → (referenced by Deployment)

Every artifact is an immutable record. Edges connect parent artifacts to child
artifacts. No artifact or edge is ever deleted; orphans are surfaced via the
orphan-detection query rather than by deletion.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from retrieval_os.core.database import Base
from retrieval_os.core.ids import uuid7


class ArtifactType(StrEnum):
    DATASET_SNAPSHOT = "DATASET_SNAPSHOT"
    EMBEDDING_ARTIFACT = "EMBEDDING_ARTIFACT"
    INDEX_ARTIFACT = "INDEX_ARTIFACT"


class EdgeRelationship(StrEnum):
    PRODUCED_FROM = "produced_from"  # EmbeddingArtifact ← DatasetSnapshot
    DERIVED_FROM = "derived_from"  # IndexArtifact ← EmbeddingArtifact
    DEPLOYED_AS = "deployed_as"  # plan version → IndexArtifact


class LineageArtifact(Base):
    """An immutable artifact in the lineage DAG.

    Storage URIs:
      Dataset snapshot:    s3://bucket/datasets/name/version.jsonl.gz
      Embedding artifact:  s3://bucket/embeddings/name/version/
      Index artifact:      qdrant://collection-name  or  pgvector://table-name
    """

    __tablename__ = "lineage_artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid7()))
    artifact_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    # SHA-256 of artifact bytes; populated for S3 artifacts, None for index artifacts
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Type-specific metadata: chunk_count, dimension_count, model_name, etc.
    # Column is "metadata" in DB; attribute renamed to avoid collision with
    # SQLAlchemy DeclarativeBase.metadata reserved name.
    artifact_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)

    # Edges where this artifact is the child (i.e. its parents)
    parent_edges: Mapped[list[LineageEdge]] = relationship(
        "LineageEdge",
        foreign_keys="LineageEdge.child_artifact_id",
        back_populates="child",
        lazy="selectin",
    )
    # Edges where this artifact is the parent (i.e. its children)
    child_edges: Mapped[list[LineageEdge]] = relationship(
        "LineageEdge",
        foreign_keys="LineageEdge.parent_artifact_id",
        back_populates="parent",
        lazy="selectin",
    )


class LineageEdge(Base):
    """A directed edge in the lineage DAG.

    parent → child means parent produced / was used to build child.
    Edges are immutable once created. Cycles are prevented at the service layer.
    """

    __tablename__ = "lineage_edges"
    __table_args__ = (
        UniqueConstraint(
            "parent_artifact_id",
            "child_artifact_id",
            name="uq_lineage_edge",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid7()))
    parent_artifact_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("lineage_artifacts.id"),
        nullable=False,
        index=True,
    )
    child_artifact_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("lineage_artifacts.id"),
        nullable=False,
        index=True,
    )
    relationship_type: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)

    parent: Mapped[LineageArtifact] = relationship(
        "LineageArtifact",
        foreign_keys=[parent_artifact_id],
        back_populates="child_edges",
        lazy="raise",
    )
    child: Mapped[LineageArtifact] = relationship(
        "LineageArtifact",
        foreign_keys=[child_artifact_id],
        back_populates="parent_edges",
        lazy="raise",
    )
