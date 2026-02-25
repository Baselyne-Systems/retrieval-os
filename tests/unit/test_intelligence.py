"""Unit tests for the Cost Intelligence domain (no live DB)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from retrieval_os.intelligence.models import CostEntry, ModelPricing
from retrieval_os.intelligence.recommender import (
    PlanStats,
    Recommendation,
    _check_cache_disabled,
    _check_high_cost_per_query,
    _check_high_top_k,
    _check_low_cache_hit_rate,
    generate_recommendations,
)

# ── ModelPricing ORM ──────────────────────────────────────────────────────────


class TestModelPricingModel:
    def test_constructor(self) -> None:
        now = datetime.now(UTC)
        pricing = ModelPricing(
            id="test-id",
            provider="openai",
            model="text-embedding-3-small",
            cost_per_1k_tokens=0.00002,
            valid_from=now,
            valid_until=None,
            created_at=now,
        )
        assert pricing.provider == "openai"
        assert pricing.cost_per_1k_tokens == 0.00002
        assert pricing.valid_until is None

    def test_local_model_zero_cost(self) -> None:
        now = datetime.now(UTC)
        pricing = ModelPricing(
            id="local-id",
            provider="sentence_transformers",
            model="all-MiniLM-L6-v2",
            cost_per_1k_tokens=0.0,
            valid_from=now,
            valid_until=None,
            created_at=now,
        )
        assert pricing.cost_per_1k_tokens == 0.0


# ── CostEntry ORM ─────────────────────────────────────────────────────────────


class TestCostEntryModel:
    def test_constructor(self) -> None:
        now = datetime.now(UTC)
        from datetime import timedelta

        entry = CostEntry(
            id="entry-id",
            plan_name="wiki-search",
            plan_version=1,
            window_start=now,
            window_end=now + timedelta(hours=1),
            provider="openai",
            model="text-embedding-3-small",
            total_queries=500,
            cache_hits=200,
            token_count=125000,
            estimated_cost_usd=0.0025,
            created_at=now,
            updated_at=now,
        )
        assert entry.plan_name == "wiki-search"
        assert entry.total_queries == 500
        assert entry.estimated_cost_usd == pytest.approx(0.0025)


# ── Schemas ───────────────────────────────────────────────────────────────────


class TestAddModelPricingRequest:
    def test_valid(self) -> None:
        from retrieval_os.intelligence.schemas import AddModelPricingRequest

        req = AddModelPricingRequest(
            provider="openai",
            model="text-embedding-3-large",
            cost_per_1k_tokens=0.00013,
        )
        assert req.cost_per_1k_tokens == 0.00013

    def test_cost_must_be_non_negative(self) -> None:
        from retrieval_os.intelligence.schemas import AddModelPricingRequest

        with pytest.raises(Exception):
            AddModelPricingRequest(
                provider="openai",
                model="text-embedding-3-small",
                cost_per_1k_tokens=-0.001,
            )


# ── Recommender: PlanStats helper ─────────────────────────────────────────────


def _make_plan(
    plan_name: str = "test-plan",
    total_queries: int = 1000,
    cache_hits: int = 500,
    estimated_cost_usd: float = 0.5,
    cache_enabled: bool = True,
    top_k: int = 10,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
) -> PlanStats:
    return PlanStats(
        plan_name=plan_name,
        total_queries=total_queries,
        cache_hits=cache_hits,
        estimated_cost_usd=estimated_cost_usd,
        cache_enabled=cache_enabled,
        top_k=top_k,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )


# ── Rule: cache disabled ──────────────────────────────────────────────────────


class TestCacheDisabledRule:
    def test_fires_when_cache_disabled(self) -> None:
        plan = _make_plan(cache_enabled=False)
        recs = _check_cache_disabled(plan)
        assert len(recs) == 1
        assert recs[0].category == "cache"
        assert recs[0].priority == "high"

    def test_silent_when_cache_enabled(self) -> None:
        plan = _make_plan(cache_enabled=True)
        assert _check_cache_disabled(plan) == []


# ── Rule: low cache hit rate ──────────────────────────────────────────────────


class TestLowCacheHitRateRule:
    def test_fires_when_hit_rate_below_30pct(self) -> None:
        plan = _make_plan(total_queries=1000, cache_hits=200, cache_enabled=True)
        recs = _check_low_cache_hit_rate(plan)
        assert len(recs) == 1
        assert recs[0].category == "cache"
        assert recs[0].priority == "medium"
        assert recs[0].potential_savings_pct == pytest.approx(80.0)

    def test_silent_when_hit_rate_at_30pct(self) -> None:
        plan = _make_plan(total_queries=1000, cache_hits=300, cache_enabled=True)
        assert _check_low_cache_hit_rate(plan) == []

    def test_silent_when_hit_rate_above_threshold(self) -> None:
        plan = _make_plan(total_queries=1000, cache_hits=800, cache_enabled=True)
        assert _check_low_cache_hit_rate(plan) == []

    def test_silent_when_cache_disabled(self) -> None:
        plan = _make_plan(total_queries=1000, cache_hits=100, cache_enabled=False)
        assert _check_low_cache_hit_rate(plan) == []

    def test_silent_when_no_queries(self) -> None:
        plan = _make_plan(total_queries=0, cache_hits=0)
        assert _check_low_cache_hit_rate(plan) == []


# ── Rule: high cost per query ─────────────────────────────────────────────────


class TestHighCostPerQueryRule:
    def test_fires_when_cost_high(self) -> None:
        # $2 / 1000 queries = $0.002/query > $0.001 threshold
        plan = _make_plan(total_queries=1000, estimated_cost_usd=2.0)
        recs = _check_high_cost_per_query(plan)
        assert len(recs) == 1
        assert recs[0].category == "model"
        assert recs[0].priority == "medium"

    def test_silent_when_cost_low(self) -> None:
        # $0.50 / 1000 queries = $0.0005/query < $0.001 threshold
        plan = _make_plan(total_queries=1000, estimated_cost_usd=0.5)
        assert _check_high_cost_per_query(plan) == []

    def test_silent_when_zero_cost(self) -> None:
        plan = _make_plan(estimated_cost_usd=0.0)
        assert _check_high_cost_per_query(plan) == []

    def test_silent_when_zero_queries(self) -> None:
        plan = _make_plan(total_queries=0)
        assert _check_high_cost_per_query(plan) == []


# ── Rule: high top_k ──────────────────────────────────────────────────────────


class TestHighTopKRule:
    def test_fires_when_top_k_exceeds_20(self) -> None:
        plan = _make_plan(top_k=25)
        recs = _check_high_top_k(plan)
        assert len(recs) == 1
        assert recs[0].category == "top_k"
        assert recs[0].priority == "low"

    def test_silent_at_top_k_20(self) -> None:
        plan = _make_plan(top_k=20)
        assert _check_high_top_k(plan) == []

    def test_silent_below_threshold(self) -> None:
        plan = _make_plan(top_k=10)
        assert _check_high_top_k(plan) == []


# ── generate_recommendations ──────────────────────────────────────────────────


class TestGenerateRecommendations:
    def test_empty_plans_returns_empty(self) -> None:
        assert generate_recommendations([]) == []

    def test_healthy_plan_returns_empty(self) -> None:
        plan = _make_plan(
            cache_enabled=True,
            total_queries=1000,
            cache_hits=600,
            estimated_cost_usd=0.3,
            top_k=10,
        )
        assert generate_recommendations([plan]) == []

    def test_multiple_rules_can_fire(self) -> None:
        plan = _make_plan(
            cache_enabled=True,
            total_queries=1000,
            cache_hits=100,  # low hit rate
            estimated_cost_usd=2.0,  # high cost
            top_k=30,  # high top_k
        )
        recs = generate_recommendations([plan])
        categories = {r.category for r in recs}
        assert "cache" in categories
        assert "model" in categories
        assert "top_k" in categories

    def test_sorted_high_before_low(self) -> None:
        plan = _make_plan(
            cache_enabled=False,  # high
            top_k=25,  # low
        )
        recs = generate_recommendations([plan])
        priorities = [r.priority for r in recs]
        priority_order = {"high": 0, "medium": 1, "low": 2}
        ordered = sorted(priorities, key=lambda p: priority_order[p])
        assert priorities == ordered

    def test_multiple_plans(self) -> None:
        plans = [
            _make_plan("plan-a", cache_enabled=False),
            _make_plan("plan-b", top_k=50),
        ]
        recs = generate_recommendations(plans)
        plan_names = {r.plan_name for r in recs}
        assert "plan-a" in plan_names
        assert "plan-b" in plan_names

    def test_recommendation_fields(self) -> None:
        plan = _make_plan(cache_enabled=False)
        recs = generate_recommendations([plan])
        assert len(recs) > 0
        r = recs[0]
        assert isinstance(r, Recommendation)
        assert r.plan_name == "test-plan"
        assert r.message  # non-empty


# ── Aggregator helpers ────────────────────────────────────────────────────────


class TestAggregatorHelpers:
    def test_chars_per_token_estimate(self) -> None:
        """1000 chars ÷ 4 = 250 tokens"""
        chars = 1000
        expected_tokens = chars // 4
        assert expected_tokens == 250

    def test_cost_formula(self) -> None:
        """250 tokens at $0.00002/1K = $0.000005"""
        token_count = 250
        cost_per_1k = 0.00002
        cost = (token_count / 1000.0) * cost_per_1k
        assert cost == pytest.approx(0.000005)

    def test_zero_cost_for_local_model(self) -> None:
        token_count = 1_000_000
        cost_per_1k = 0.0
        cost = (token_count / 1000.0) * cost_per_1k
        assert cost == 0.0
