"""Pydantic schemas for the Lineage domain."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from retrieval_os.lineage.models import ArtifactType, EdgeRelationship


class RegisterArtifactRequest(BaseModel):
    artifact_type: ArtifactType
    name: str = Field(..., min_length=1, max_length=255)
    version: str = Field(..., min_length=1, max_length=100)
    storage_uri: str = Field(..., min_length=1)
    # Caller may supply the hash; if absent and artifact is on S3, we fetch it.
    content_hash: str | None = Field(None, min_length=64, max_length=64)
    metadata: dict[str, Any] | None = None
    created_by: str = Field(..., min_length=1, max_length=255)


class CreateEdgeRequest(BaseModel):
    parent_artifact_id: str = Field(..., min_length=1)
    child_artifact_id: str = Field(..., min_length=1)
    relationship_type: EdgeRelationship
    created_by: str = Field(..., min_length=1, max_length=255)


class ArtifactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    artifact_type: str
    name: str
    version: str
    storage_uri: str
    content_hash: str | None
    # ORM attribute is artifact_metadata (metadata is reserved by SQLAlchemy).
    # JSON field is still "metadata" for API compatibility.
    metadata: dict[str, Any] | None = Field(
        None, validation_alias=AliasChoices("artifact_metadata", "metadata")
    )
    created_at: datetime
    created_by: str


class EdgeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    parent_artifact_id: str
    child_artifact_id: str
    relationship_type: str
    created_at: datetime
    created_by: str


class ArtifactWithEdgesResponse(ArtifactResponse):
    """Artifact plus its immediate parents and children."""
    parents: list[ArtifactResponse] = []
    children: list[ArtifactResponse] = []


class LineageGraphResponse(BaseModel):
    """Full lineage graph for a plan — all reachable artifacts and their edges."""
    plan_name: str
    artifacts: list[ArtifactResponse]
    edges: list[EdgeResponse]


class OrphansResponse(BaseModel):
    """Artifacts not reachable from any active deployment."""
    total: int
    artifacts: list[ArtifactResponse]
