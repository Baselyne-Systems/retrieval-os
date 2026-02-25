"""Async circuit breaker for protecting embedding and index backends.

State machine:
  CLOSED   → Normal operation. Tracks failures in a sliding time window.
             Opens when failure_threshold is exceeded.
  OPEN     → Fails fast (CircuitOpenError). After reset_timeout_seconds,
             transitions to HALF_OPEN.
  HALF_OPEN → Allows calls through. After half_open_success_threshold
              consecutive successes, closes the circuit.
              Any failure immediately re-opens.

Thread/task safety: each CircuitBreaker owns an asyncio.Lock; concurrent
callers are serialised only for the brief state-check/update, not during
the actual backend call.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from retrieval_os.core.exceptions import CircuitOpenError

log = logging.getLogger(__name__)


class CircuitState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Sliding-window circuit breaker for an async callable.

    Args:
        name:                       Human-readable identifier (used in logs/errors).
        failure_threshold:          Number of failures within window_seconds needed
                                    to open the circuit.
        window_seconds:             Duration of the sliding failure window.
        reset_timeout_seconds:      How long to stay OPEN before probing with
                                    HALF_OPEN.
        half_open_success_threshold: Consecutive successes required in HALF_OPEN
                                    before the circuit fully closes.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        window_seconds: float = 60.0,
        reset_timeout_seconds: float = 60.0,
        half_open_success_threshold: int = 3,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._window_seconds = window_seconds
        self._reset_timeout_seconds = reset_timeout_seconds
        self._half_open_success_threshold = half_open_success_threshold

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_times: list[float] = []   # monotonic timestamps
        self._opened_at: float | None = None
        self._half_open_successes: int = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *func* through the circuit breaker.

        func may be an async function or a regular callable.
        Raises CircuitOpenError immediately when the circuit is OPEN.
        """
        async with self._lock:
            # Transition OPEN → HALF_OPEN if timeout expired
            if self._state == CircuitState.OPEN:
                assert self._opened_at is not None
                if time.monotonic() - self._opened_at >= self._reset_timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_successes = 0
                    log.info("circuit_breaker.half_open", extra={"breaker_name": self.name})
                else:
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is OPEN — failing fast",
                        detail={"breaker_name": self.name, "state": "OPEN"},
                    )

        # Execute outside the lock so other tasks can check state concurrently
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
        except CircuitOpenError:
            raise
        except Exception:
            async with self._lock:
                self._on_failure()
            raise
        else:
            async with self._lock:
                self._on_success()
            return result

    # ── Internal state transitions ────────────────────────────────────────────

    def _on_failure(self) -> None:
        now = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = now
            self._failure_times.clear()
            log.warning(
                "circuit_breaker.opened",
                extra={"breaker_name": self.name, "reason": "half_open_failure"},
            )
            return

        # Sliding window: discard failures older than window_seconds
        cutoff = now - self._window_seconds
        self._failure_times = [t for t in self._failure_times if t >= cutoff]
        self._failure_times.append(now)

        if len(self._failure_times) >= self._failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = now
            log.warning(
                "circuit_breaker.opened",
                extra={
                    "breaker_name": self.name,
                    "failures": len(self._failure_times),
                    "threshold": self._failure_threshold,
                },
            )

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self._half_open_success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_times.clear()
                self._opened_at = None
                self._half_open_successes = 0
                log.info("circuit_breaker.closed", extra={"breaker_name": self.name})

    def reset(self) -> None:
        """Force the circuit back to CLOSED. Useful for tests."""
        self._state = CircuitState.CLOSED
        self._failure_times.clear()
        self._opened_at = None
        self._half_open_successes = 0


# ── Per-backend singletons ────────────────────────────────────────────────────

# Created here so the event loop isn't required at import time (Python 3.12+).

embed_breakers: dict[str, CircuitBreaker] = {}
index_breakers: dict[str, CircuitBreaker] = {}


def get_embed_breaker(provider: str) -> CircuitBreaker:
    if provider not in embed_breakers:
        embed_breakers[provider] = CircuitBreaker(
            f"embed.{provider}",
            failure_threshold=3,
            window_seconds=30.0,
            reset_timeout_seconds=30.0,
            half_open_success_threshold=2,
        )
    return embed_breakers[provider]


def get_index_breaker(backend: str) -> CircuitBreaker:
    if backend not in index_breakers:
        index_breakers[backend] = CircuitBreaker(
            f"index.{backend}",
            failure_threshold=3,
            window_seconds=30.0,
            reset_timeout_seconds=30.0,
            half_open_success_threshold=2,
        )
    return index_breakers[backend]
