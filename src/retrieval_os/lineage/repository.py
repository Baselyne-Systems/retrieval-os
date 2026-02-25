"""Database access layer for the Lineage domain."""

from __future__ import annotations

import textwrap

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.lineage.models import LineageArtifact, LineageEdge


class LineageRepository:

    async def create_artifact(
        self, session: AsyncSession, artifact: LineageArtifact
    ) -> LineageArtifact:
        session.add(artifact)
        await session.flush()
        await session.refresh(artifact)
        return artifact

    async def get_artifact(
        self, session: AsyncSession, artifact_id: str
    ) -> LineageArtifact | None:
        result = await session.execute(
            select(LineageArtifact).where(LineageArtifact.id == artifact_id)
        )
        return result.scalar_one_or_none()

    async def get_artifact_by_uri(
        self, session: AsyncSession, storage_uri: str
    ) -> LineageArtifact | None:
        result = await session.execute(
            select(LineageArtifact).where(
                LineageArtifact.storage_uri == storage_uri
            )
        )
        return result.scalar_one_or_none()

    async def list_artifacts(
        self,
        session: AsyncSession,
        artifact_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[LineageArtifact], int]:
        q = select(LineageArtifact)
        if artifact_type:
            q = q.where(LineageArtifact.artifact_type == artifact_type)
        q = q.order_by(LineageArtifact.created_at.desc())

        count_q = sa.select(sa.func.count()).select_from(q.subquery())
        total = (await session.execute(count_q)).scalar_one()

        results = await session.execute(q.offset(offset).limit(limit))
        return list(results.scalars().all()), total

    async def create_edge(
        self, session: AsyncSession, edge: LineageEdge
    ) -> LineageEdge:
        session.add(edge)
        await session.flush()
        return edge

    async def get_edge(
        self, session: AsyncSession, edge_id: str
    ) -> LineageEdge | None:
        result = await session.execute(
            select(LineageEdge).where(LineageEdge.id == edge_id)
        )
        return result.scalar_one_or_none()

    async def edge_exists(
        self,
        session: AsyncSession,
        parent_id: str,
        child_id: str,
    ) -> bool:
        result = await session.execute(
            select(LineageEdge.id).where(
                LineageEdge.parent_artifact_id == parent_id,
                LineageEdge.child_artifact_id == child_id,
            )
        )
        return result.scalar_one_or_none() is not None

    async def get_ancestors(
        self,
        session: AsyncSession,
        artifact_id: str,
        max_depth: int = 20,
    ) -> list[dict]:
        """Return ancestor artifact IDs and their depth using a recursive CTE.

        Returns list of dicts: {artifact_id, depth}
        """
        cte_sql = textwrap.dedent("""
            WITH RECURSIVE ancestors AS (
                SELECT
                    parent_artifact_id AS artifact_id,
                    1 AS depth
                FROM lineage_edges
                WHERE child_artifact_id = :start_id
                UNION ALL
                SELECT
                    e.parent_artifact_id,
                    a.depth + 1
                FROM lineage_edges e
                JOIN ancestors a ON e.child_artifact_id = a.artifact_id
                WHERE a.depth < :max_depth
            )
            SELECT DISTINCT artifact_id, MIN(depth) AS depth
            FROM ancestors
            GROUP BY artifact_id
            ORDER BY depth
        """)
        result = await session.execute(
            sa.text(cte_sql),
            {"start_id": artifact_id, "max_depth": max_depth},
        )
        return [{"artifact_id": row.artifact_id, "depth": row.depth} for row in result]

    async def get_descendants(
        self,
        session: AsyncSession,
        artifact_id: str,
        max_depth: int = 20,
    ) -> list[dict]:
        """Return descendant artifact IDs and their depth using a recursive CTE."""
        cte_sql = textwrap.dedent("""
            WITH RECURSIVE descendants AS (
                SELECT
                    child_artifact_id AS artifact_id,
                    1 AS depth
                FROM lineage_edges
                WHERE parent_artifact_id = :start_id
                UNION ALL
                SELECT
                    e.child_artifact_id,
                    d.depth + 1
                FROM lineage_edges e
                JOIN descendants d ON e.parent_artifact_id = d.artifact_id
                WHERE d.depth < :max_depth
            )
            SELECT DISTINCT artifact_id, MIN(depth) AS depth
            FROM descendants
            GROUP BY artifact_id
            ORDER BY depth
        """)
        result = await session.execute(
            sa.text(cte_sql),
            {"start_id": artifact_id, "max_depth": max_depth},
        )
        return [{"artifact_id": row.artifact_id, "depth": row.depth} for row in result]

    async def get_orphaned_artifacts(
        self, session: AsyncSession
    ) -> list[LineageArtifact]:
        """Return artifacts that are not reachable from any active deployment.

        An artifact is an orphan if it has no child edges (i.e. it is a leaf
        with no downstream artifact) AND it is not an INDEX_ARTIFACT that is
        referenced by a live deployment.

        Simplified definition for Phase 5: any artifact that has no outgoing
        edges (no children) is considered a candidate orphan. The full
        implementation will cross-reference deployments in Phase 6.
        """
        # Subquery: IDs of artifacts that are parents of something
        has_children = (
            select(LineageEdge.parent_artifact_id)
            .distinct()
            .scalar_subquery()
        )
        result = await session.execute(
            select(LineageArtifact)
            .where(LineageArtifact.id.not_in(has_children))
            .order_by(LineageArtifact.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_artifacts_by_ids(
        self, session: AsyncSession, artifact_ids: list[str]
    ) -> list[LineageArtifact]:
        if not artifact_ids:
            return []
        result = await session.execute(
            select(LineageArtifact).where(
                LineageArtifact.id.in_(artifact_ids)
            )
        )
        return list(result.scalars().all())

    async def get_edges_for_artifacts(
        self, session: AsyncSession, artifact_ids: list[str]
    ) -> list[LineageEdge]:
        """Return all edges where both endpoints are in the given set."""
        if not artifact_ids:
            return []
        result = await session.execute(
            select(LineageEdge).where(
                LineageEdge.parent_artifact_id.in_(artifact_ids),
                LineageEdge.child_artifact_id.in_(artifact_ids),
            )
        )
        return list(result.scalars().all())


lineage_repo = LineageRepository()
