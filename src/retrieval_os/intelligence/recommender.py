"""Rule-based cost and performance recommendations.

All functions are pure (no DB, no async) so they can be unit-tested without
any infrastructure. The service layer feeds them aggregated data.

Rules:
  1. CACHE_DISABLED     — plan has cache_enabled=False
  2. LOW_CACHE_HIT_RATE — cache enabled but hit rate < 30 %
  3. HIGH_COST_PER_Q    — estimated_cost_usd / total_queries > $0.001
  4. HIGH_TOP_K         — top_k > 20 (diminishing quality returns past ~10-15)
"""

from __future__ import annotations

from dataclasses import dataclass

# Thresholds
_LOW_CACHE_HIT_RATE = 0.30  # below this → recommend cache tuning
_HIGH_COST_PER_QUERY_USD = 0.001  # above this → recommend cheaper model
_HIGH_TOP_K = 20  # above this → recommend reducing top_k


@dataclass(frozen=True)
class PlanStats:
    """Aggregated stats for one plan, fed to the recommender."""

    plan_name: str
    total_queries: int
    cache_hits: int
    estimated_cost_usd: float
    cache_enabled: bool
    top_k: int
    embedding_provider: str
    embedding_model: str


@dataclass(frozen=True)
class Recommendation:
    plan_name: str
    category: str  # "cache" | "model" | "top_k"
    priority: str  # "high" | "medium" | "low"
    message: str
    potential_savings_pct: float | None


def generate_recommendations(plans: list[PlanStats]) -> list[Recommendation]:
    """Apply all recommendation rules to a list of PlanStats.

    Returns recommendations sorted by priority (high → medium → low).
    """
    recs: list[Recommendation] = []
    for plan in plans:
        recs.extend(_check_cache_disabled(plan))
        recs.extend(_check_low_cache_hit_rate(plan))
        recs.extend(_check_high_cost_per_query(plan))
        recs.extend(_check_high_top_k(plan))

    priority_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(recs, key=lambda r: (priority_order.get(r.priority, 9), r.plan_name))


# ── Individual rules ──────────────────────────────────────────────────────────


def _check_cache_disabled(plan: PlanStats) -> list[Recommendation]:
    if plan.cache_enabled:
        return []
    return [
        Recommendation(
            plan_name=plan.plan_name,
            category="cache",
            priority="high",
            message=(
                f"Plan '{plan.plan_name}' has semantic cache disabled. "
                "Enable it to avoid re-embedding identical queries."
            ),
            potential_savings_pct=None,
        )
    ]


def _check_low_cache_hit_rate(plan: PlanStats) -> list[Recommendation]:
    if not plan.cache_enabled or plan.total_queries == 0:
        return []
    hit_rate = plan.cache_hits / plan.total_queries
    if hit_rate >= _LOW_CACHE_HIT_RATE:
        return []
    return [
        Recommendation(
            plan_name=plan.plan_name,
            category="cache",
            priority="medium",
            message=(
                f"Plan '{plan.plan_name}' cache hit rate is "
                f"{hit_rate:.0%} (threshold {_LOW_CACHE_HIT_RATE:.0%}). "
                "Consider increasing cache TTL or reviewing query diversity."
            ),
            potential_savings_pct=round((1.0 - hit_rate) * 100, 1),
        )
    ]


def _check_high_cost_per_query(plan: PlanStats) -> list[Recommendation]:
    if plan.total_queries == 0 or plan.estimated_cost_usd == 0:
        return []
    cost_per_q = plan.estimated_cost_usd / plan.total_queries
    if cost_per_q <= _HIGH_COST_PER_QUERY_USD:
        return []
    return [
        Recommendation(
            plan_name=plan.plan_name,
            category="model",
            priority="medium",
            message=(
                f"Plan '{plan.plan_name}' averages ${cost_per_q:.5f}/query "
                f"({plan.embedding_provider}/{plan.embedding_model}). "
                "Consider switching to a smaller or local embedding model."
            ),
            potential_savings_pct=None,
        )
    ]


def _check_high_top_k(plan: PlanStats) -> list[Recommendation]:
    if plan.top_k <= _HIGH_TOP_K:
        return []
    return [
        Recommendation(
            plan_name=plan.plan_name,
            category="top_k",
            priority="low",
            message=(
                f"Plan '{plan.plan_name}' retrieves top_k={plan.top_k}. "
                f"Values above {_HIGH_TOP_K} rarely improve quality but "
                "increase embedding and index costs."
            ),
            potential_savings_pct=None,
        )
    ]
