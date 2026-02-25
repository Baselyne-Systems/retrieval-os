"""DAG utilities for the Lineage domain.

Cycle detection uses the same recursive CTE ancestor query as normal
lineage traversal — if the proposed parent appears in the ancestors of
the proposed child, adding the edge would create a cycle.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_os.lineage.repository import lineage_repo


async def would_create_cycle(
    session: AsyncSession,
    parent_artifact_id: str,
    child_artifact_id: str,
) -> bool:
    """Return True if adding parent→child would create a cycle.

    A cycle would form if child is already an ancestor of parent
    (i.e. parent is reachable from child by following existing edges).
    Adding parent→child would then close a loop.

    Also returns True if parent_artifact_id == child_artifact_id (self-loop).
    """
    if parent_artifact_id == child_artifact_id:
        return True

    # If child is already an ancestor of parent, adding parent→child is a cycle.
    ancestors_of_parent = await lineage_repo.get_ancestors(session, parent_artifact_id)
    ancestor_ids = {row["artifact_id"] for row in ancestors_of_parent}
    return child_artifact_id in ancestor_ids


async def compute_dag_depth(
    session: AsyncSession,
    artifact_id: str,
) -> int:
    """Return the maximum depth of the subtree rooted at artifact_id.

    Depth 0 = leaf (no children).
    """
    descendants = await lineage_repo.get_descendants(session, artifact_id)
    if not descendants:
        return 0
    return max(row["depth"] for row in descendants)
