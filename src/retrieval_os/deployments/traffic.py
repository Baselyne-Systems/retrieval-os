"""Traffic weight management for the serving path.

Redis layout:
  ros:deployment:{project_name}:active  →  deployment_id (string)
  ros:project:{project_name}:active     →  JSON serving config (warmed by query_router)

The serving path reads the project config from Redis. When a deployment goes ACTIVE
we push the resolved project config there so the next query uses the new version
without any DB round-trip.
"""

from __future__ import annotations

import json
import logging

from retrieval_os.core.redis_client import get_redis

log = logging.getLogger(__name__)

_ACTIVE_KEY_TTL = 300  # seconds — background loop refreshes well before expiry


def _active_key(project_name: str) -> str:
    return f"ros:deployment:{project_name}:active"


def _project_config_key(project_name: str) -> str:
    return f"ros:project:{project_name}:active"


async def set_active_deployment(
    project_name: str,
    deployment_id: str,
    serving_config: dict,
) -> None:
    """Atomically record the active deployment and push project config to Redis."""
    redis = await get_redis()
    try:
        pipe = redis.pipeline()
        pipe.set(_active_key(project_name), deployment_id, ex=_ACTIVE_KEY_TTL)
        pipe.set(
            _project_config_key(project_name),
            json.dumps(serving_config, default=str),
            ex=_ACTIVE_KEY_TTL,
        )
        await pipe.execute()
    except Exception:
        log.warning(
            "traffic.set_active_failed",
            extra={"project": project_name, "deployment": deployment_id},
        )


async def clear_active_deployment(project_name: str) -> None:
    """Remove the active deployment marker (used on rollback)."""
    redis = await get_redis()
    try:
        await redis.delete(_active_key(project_name))
    except Exception:
        log.warning("traffic.clear_active_failed", extra={"project": project_name})


async def get_active_deployment_id(project_name: str) -> str | None:
    """Return the current active deployment ID from Redis, or None."""
    redis = await get_redis()
    try:
        val = await redis.get(_active_key(project_name))
        return val.decode() if isinstance(val, bytes) else val
    except Exception:
        return None
