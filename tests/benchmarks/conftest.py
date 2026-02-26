"""Benchmark conftest: collects timing results and prints a summary table."""

from __future__ import annotations

import pytest

# Module-level list — safe because benchmarks always run single-threaded.
_RESULTS: list[dict] = []


@pytest.fixture
def record_bm():
    """Fixture used by timing tests to register a measured result.

    Usage::

        def test_something(record_bm):
            elapsed = ...
            record_bm("label", elapsed, limit_s=2.0, n=10_000, unit="hash")
    """

    def _record(
        label: str,
        elapsed_s: float,
        *,
        limit_s: float,
        n: int,
        unit: str = "op",
    ) -> None:
        _RESULTS.append(
            {
                "label": label,
                "elapsed_ms": elapsed_s * 1000,
                "limit_ms": limit_s * 1000,
                "n": n,
                "unit": unit,
            }
        )

    return _record


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Print a benchmark summary table after all tests complete."""
    if not _RESULTS:
        return

    tw = terminalreporter
    tw.write_sep("=", "benchmark summary")
    tw.write_line(f"  {'Benchmark':<44} {'Actual':>9}  {'Per-op':>12}  {'Limit':>9}  {'Margin':>7}")
    tw.write_line("  " + "-" * 82)
    for r in _RESULTS:
        per_op_us = r["elapsed_ms"] / r["n"] * 1000
        margin = r["limit_ms"] / r["elapsed_ms"]
        status = "✓" if r["elapsed_ms"] < r["limit_ms"] else "✗"
        tw.write_line(
            f"  {r['label']:<44} {r['elapsed_ms']:>7.1f} ms"
            f"  {per_op_us:>9.2f} µs/{r['unit']}"
            f"  {r['limit_ms']:>7.0f} ms"
            f"  {margin:>6.1f}×  {status}"
        )
    tw.write_line("")
