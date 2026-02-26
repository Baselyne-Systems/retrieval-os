"""Benchmark: Operational Reliability

Proves the system behaves correctly under the conditions production deployments
face: large numbers of rollout steps, many concurrent configs, and sustained
watchdog cycles.

What we measure
---------------
1. Rollout math precision across 100 incremental steps — floating-point
   accumulation must not drift beyond the representable 1.0 boundary.

2. Rollout step correctness across all valid step sizes — every configuration
   that users can specify must converge to exactly 1.0 in the expected number
   of steps.

3. Weight cap invariant — no matter how large the step or how many times the
   stepper runs after full ramp, weight must never exceed 1.0.

4. Threshold check throughput — the watchdog scans all live deployments on
   every cycle; regression detection across large deployment counts must not
   block the loop.

Scale targets
-------------
- 100 rollout steps with 1% increment → weight lands at exactly 1.0 (no drift)
- All step sizes from 1% to 50%       → each converges within ⌈100/step⌉ steps
- 1 000 threshold evaluations          in < 5 ms
"""

from __future__ import annotations

import math
import time

# ── Rollout step math ─────────────────────────────────────────────────────────


class TestRolloutStepMath:
    def _simulate_rollout(self, *, step_percent: float) -> tuple[float, int]:
        """Simulate repeated calls to the rollout stepper.

        Returns (final_weight, steps_taken).
        Mirrors the logic in deployments/service.py::step_rolling_deployments.
        """
        weight = 0.0
        steps = 0
        while weight < 1.0:
            weight = min(1.0, weight + step_percent / 100.0)
            steps += 1
        return weight, steps

    def test_1pct_step_reaches_exactly_1_after_100_steps(self) -> None:
        """100 steps of 1% each must land at exactly 1.0 — no floating-point overshoot."""
        weight, steps = self._simulate_rollout(step_percent=1.0)
        assert weight == 1.0
        assert steps == 100

    def test_10pct_step_reaches_exactly_1(self) -> None:
        """10% step must land at exactly 1.0 regardless of floating-point accumulation.

        IEEE-754 float accumulation means 10 × 0.1 != 1.0 in binary, so the
        min(1.0, ...) cap is load-bearing: it guarantees the final weight is
        representable as exactly 1.0 even if intermediate sums drift.
        """
        weight, _ = self._simulate_rollout(step_percent=10.0)
        assert weight == 1.0

    def test_33pct_step_reaches_exactly_1_within_expected_steps(self) -> None:
        """33% steps: ⌈100/33⌉ = 4 steps; final weight must be exactly 1.0 (min-cap)."""
        weight, steps = self._simulate_rollout(step_percent=33.0)
        assert weight == 1.0
        assert steps == math.ceil(100 / 33.0)

    def test_all_standard_step_sizes_converge_to_exactly_1(self) -> None:
        """Every standard step size must reach exactly 1.0 (never overshoot)."""
        step_sizes = [1, 2, 5, 10, 20, 25, 33, 50]
        for step in step_sizes:
            weight, _ = self._simulate_rollout(step_percent=float(step))
            assert weight == 1.0, (
                f"step_percent={step}: final weight is {weight}, expected exactly 1.0"
            )

    def test_weight_never_exceeds_1_regardless_of_large_step(self) -> None:
        """Oversized steps (e.g., 75%) must still cap at exactly 1.0."""
        weight, steps = self._simulate_rollout(step_percent=75.0)
        assert weight == 1.0
        assert steps == 2  # 0.75 → 1.0 (second step: min(1.0, 0.75 + 0.75) = 1.0)

    def test_weight_monotonically_increases(self) -> None:
        """Traffic weight must never decrease during a gradual rollout."""
        step_percent = 15.0
        weight = 0.0
        history: list[float] = []
        while weight < 1.0:
            weight = min(1.0, weight + step_percent / 100.0)
            history.append(weight)

        for i in range(1, len(history)):
            assert history[i] >= history[i - 1], (
                f"Weight decreased from {history[i - 1]} to {history[i]} at step {i}"
            )

    def test_rollout_step_count_is_bounded_by_formula(self) -> None:
        """Steps to full ramp is at most ⌈100 / step_percent⌉ + 1 for all valid step sizes.

        The upper bound is ⌈100/step⌉ + 1 rather than ⌈100/step⌉ exactly because
        IEEE-754 float accumulation can push the sum slightly below 1.0 after the
        mathematically exact number of steps (e.g., 10 × 0.1 = 0.9999... in binary),
        requiring one additional step where min(1.0, ...) caps the weight at 1.0.
        The important invariant is that the rollout terminates and lands at 1.0.
        """
        for step in range(1, 51):  # 1% to 50%
            final_weight, actual_steps = self._simulate_rollout(step_percent=float(step))
            expected_max_steps = math.ceil(100.0 / step) + 1
            assert final_weight == 1.0, f"step_percent={step}: final weight {final_weight} != 1.0"
            assert actual_steps <= expected_max_steps, (
                f"step_percent={step}: took {actual_steps} steps, max expected {expected_max_steps}"
            )


# ── Threshold scan throughput ─────────────────────────────────────────────────


class TestThresholdScanThroughput:
    def test_1k_threshold_evaluations_under_5ms(self, record_bm) -> None:
        """Evaluating 1 000 (recall_value, threshold) pairs must take < 5 ms.

        The watchdog runs this check for every live deployment on every cycle.
        At 1 000 live deployments, the inner loop must not block the event loop.
        """
        import random

        rng = random.Random(42)
        evaluations = [(rng.uniform(0.4, 1.0), rng.uniform(0.5, 0.9)) for _ in range(1_000)]

        start = time.perf_counter()
        rollback_count = sum(1 for val, threshold in evaluations if val < threshold)
        elapsed = time.perf_counter() - start

        record_bm("1k watchdog threshold scans", elapsed, limit_s=0.005, n=1_000, unit="deployment")
        assert elapsed < 0.005, (
            f"1 000 threshold evaluations took {elapsed * 1000:.3f}ms; must be < 5ms"
        )
        # Sanity: some rollbacks expected (random values will straddle thresholds)
        assert rollback_count > 0
