"""Unit tests for the async circuit breaker."""

from __future__ import annotations

import asyncio

import pytest

from retrieval_os.core.circuit_breaker import CircuitBreaker, CircuitState
from retrieval_os.core.exceptions import CircuitOpenError

# ── Helpers ───────────────────────────────────────────────────────────────────


async def _ok(*args, **kwargs) -> str:
    return "ok"


async def _fail(*args, **kwargs) -> None:
    raise ValueError("boom")


def _sync_ok() -> str:
    return "sync-ok"


def _sync_fail() -> None:
    raise RuntimeError("sync-boom")


def _make_breaker(**kwargs) -> CircuitBreaker:
    defaults = dict(
        failure_threshold=3,
        window_seconds=10.0,
        reset_timeout_seconds=10.0,
        half_open_success_threshold=2,
    )
    defaults.update(kwargs)
    return CircuitBreaker("test", **defaults)


# ── State machine ─────────────────────────────────────────────────────────────


class TestInitialState:
    def test_starts_closed(self) -> None:
        cb = _make_breaker()
        assert cb.state == CircuitState.CLOSED

    def test_name_stored(self) -> None:
        cb = CircuitBreaker("my-breaker", failure_threshold=1, window_seconds=1.0,
                            reset_timeout_seconds=1.0, half_open_success_threshold=1)
        assert cb.name == "my-breaker"


class TestClosedState:
    @pytest.mark.asyncio
    async def test_success_stays_closed(self) -> None:
        cb = _make_breaker()
        result = await cb.call(_ok)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_below_threshold_stays_closed(self) -> None:
        cb = _make_breaker(failure_threshold=3)
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_at_threshold_opens(self) -> None:
        cb = _make_breaker(failure_threshold=3)
        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_sync_callable_success(self) -> None:
        cb = _make_breaker()
        result = await cb.call(_sync_ok)
        assert result == "sync-ok"

    @pytest.mark.asyncio
    async def test_sync_callable_failure_counted(self) -> None:
        cb = _make_breaker(failure_threshold=2)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(_sync_fail)
        assert cb.state == CircuitState.OPEN


class TestOpenState:
    @pytest.mark.asyncio
    async def test_open_raises_circuit_open_error(self) -> None:
        cb = _make_breaker(failure_threshold=1, reset_timeout_seconds=9999.0)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            await cb.call(_ok)

    @pytest.mark.asyncio
    async def test_open_transitions_to_half_open_after_timeout(self) -> None:
        cb = _make_breaker(failure_threshold=1, reset_timeout_seconds=0.01)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.state == CircuitState.OPEN
        # Wait for reset timeout
        await asyncio.sleep(0.05)
        # Next call should go through (transitions to HALF_OPEN)
        result = await cb.call(_ok)
        assert result == "ok"
        # Should progress toward CLOSED (1 success so far; need 2)
        assert cb.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_open_error_not_re_counted(self) -> None:
        """CircuitOpenError must not increment the failure counter."""
        cb = _make_breaker(failure_threshold=2, reset_timeout_seconds=9999.0)
        # One real failure
        with pytest.raises(ValueError):
            await cb.call(_fail)
        # Force to OPEN
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.state == CircuitState.OPEN
        # Several CircuitOpenError calls — should not re-open (already open) or crash
        for _ in range(5):
            with pytest.raises(CircuitOpenError):
                await cb.call(_ok)
        assert cb.state == CircuitState.OPEN


class TestHalfOpenState:
    @pytest.mark.asyncio
    async def test_enough_successes_closes_circuit(self) -> None:
        cb = _make_breaker(
            failure_threshold=1,
            reset_timeout_seconds=0.01,
            half_open_success_threshold=2,
        )
        with pytest.raises(ValueError):
            await cb.call(_fail)
        await asyncio.sleep(0.05)
        # Two consecutive successes
        await cb.call(_ok)
        assert cb.state == CircuitState.HALF_OPEN
        await cb.call(_ok)
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens(self) -> None:
        cb = _make_breaker(
            failure_threshold=1,
            reset_timeout_seconds=0.01,
            half_open_success_threshold=3,
        )
        with pytest.raises(ValueError):
            await cb.call(_fail)
        await asyncio.sleep(0.05)
        # One success transitions OPEN→HALF_OPEN then succeeds
        await cb.call(_ok)
        assert cb.state == CircuitState.HALF_OPEN
        # Failure in HALF_OPEN → re-open
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.state == CircuitState.OPEN


class TestSlidingWindow:
    @pytest.mark.asyncio
    async def test_old_failures_expire(self) -> None:
        cb = _make_breaker(failure_threshold=3, window_seconds=0.05)
        # Two failures
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        # Wait for window to expire
        await asyncio.sleep(0.1)
        # Two more failures — but old ones are gone; total in window = 2
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_recent_failures_trigger_open(self) -> None:
        cb = _make_breaker(failure_threshold=3, window_seconds=5.0)
        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        assert cb.state == CircuitState.OPEN


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_state(self) -> None:
        cb = _make_breaker(failure_threshold=1, reset_timeout_seconds=9999.0)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        result = await cb.call(_ok)
        assert result == "ok"


class TestCallWithArgs:
    @pytest.mark.asyncio
    async def test_args_forwarded(self) -> None:
        async def echo(x: int, *, y: int) -> int:
            return x + y

        cb = _make_breaker()
        result = await cb.call(echo, 3, y=4)
        assert result == 7


# ── Singletons ────────────────────────────────────────────────────────────────


class TestSingletons:
    def test_get_embed_breaker_returns_same_instance(self) -> None:
        from retrieval_os.core.circuit_breaker import get_embed_breaker

        b1 = get_embed_breaker("openai")
        b2 = get_embed_breaker("openai")
        assert b1 is b2

    def test_get_embed_breaker_different_providers_are_different(self) -> None:
        from retrieval_os.core.circuit_breaker import get_embed_breaker

        b1 = get_embed_breaker("openai")
        b2 = get_embed_breaker("cohere")
        assert b1 is not b2

    def test_get_index_breaker_returns_same_instance(self) -> None:
        from retrieval_os.core.circuit_breaker import get_index_breaker

        b1 = get_index_breaker("qdrant")
        b2 = get_index_breaker("qdrant")
        assert b1 is b2
