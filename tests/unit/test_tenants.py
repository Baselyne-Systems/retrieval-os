"""Unit tests for the Tenants domain (ORM, schemas, key helpers)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from retrieval_os.tenants.models import ApiKey, Tenant
from retrieval_os.tenants.schemas import (
    ApiKeyCreatedResponse,
    CreateApiKeyRequest,
    CreateTenantRequest,
)
from retrieval_os.tenants.service import generate_api_key, hash_api_key

# ── Tenant ORM ────────────────────────────────────────────────────────────────


class TestTenantModel:
    def test_constructor(self) -> None:
        now = datetime.now(UTC)
        t = Tenant(
            id="t-001",
            name="acme",
            is_active=True,
            max_requests_per_minute=120,
            max_plans=5,
            created_at=now,
            updated_at=now,
        )
        assert t.name == "acme"
        assert t.max_requests_per_minute == 120
        assert t.max_plans == 5
        assert t.is_active is True

    def test_explicit_is_active_false(self) -> None:
        now = datetime.now(UTC)
        t = Tenant(id="t-002", name="beta", is_active=False, created_at=now, updated_at=now)
        assert t.is_active is False


# ── ApiKey ORM ────────────────────────────────────────────────────────────────


class TestApiKeyModel:
    def test_constructor(self) -> None:
        now = datetime.now(UTC)
        k = ApiKey(
            id="k-001",
            tenant_id="t-001",
            name="prod-key",
            key_prefix="ros_a1b2c3d4",
            key_hash="abc123",
            is_active=True,
            created_at=now,
            expires_at=None,
        )
        assert k.key_prefix == "ros_a1b2c3d4"
        assert k.is_active is True
        assert k.expires_at is None

    def test_with_expiry(self) -> None:
        now = datetime.now(UTC)
        k = ApiKey(
            id="k-002",
            tenant_id="t-001",
            name="temp-key",
            key_prefix="ros_xxxxxxxx",
            key_hash="hash",
            is_active=True,
            created_at=now,
            expires_at=now + timedelta(days=30),
        )
        assert k.expires_at is not None
        assert k.expires_at > now


# ── Schemas ───────────────────────────────────────────────────────────────────


class TestCreateTenantRequest:
    def test_valid_slug(self) -> None:
        req = CreateTenantRequest(name="my-tenant")
        assert req.name == "my-tenant"

    def test_valid_with_numbers(self) -> None:
        req = CreateTenantRequest(name="tenant-42")
        assert req.name == "tenant-42"

    def test_invalid_starts_with_hyphen(self) -> None:
        with pytest.raises(Exception):
            CreateTenantRequest(name="-bad")

    def test_invalid_ends_with_hyphen(self) -> None:
        with pytest.raises(Exception):
            CreateTenantRequest(name="bad-")

    def test_invalid_uppercase(self) -> None:
        with pytest.raises(Exception):
            CreateTenantRequest(name="MyTenant")

    def test_max_rpm_defaults_to_60(self) -> None:
        req = CreateTenantRequest(name="default-tenant")
        assert req.max_requests_per_minute == 60

    def test_max_rpm_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            CreateTenantRequest(name="t", max_requests_per_minute=0)

    def test_max_plans_defaults_to_10(self) -> None:
        req = CreateTenantRequest(name="default-tenant")
        assert req.max_plans == 10


class TestCreateApiKeyRequest:
    def test_valid(self) -> None:
        req = CreateApiKeyRequest(name="my-key")
        assert req.name == "my-key"
        assert req.expires_at is None

    def test_with_expiry(self) -> None:
        exp = datetime.now(UTC) + timedelta(days=365)
        req = CreateApiKeyRequest(name="expiring-key", expires_at=exp)
        assert req.expires_at == exp

    def test_empty_name_raises(self) -> None:
        with pytest.raises(Exception):
            CreateApiKeyRequest(name="")


class TestApiKeyCreatedResponse:
    def test_includes_raw_key(self) -> None:
        now = datetime.now(UTC)
        resp = ApiKeyCreatedResponse(
            id="k-001",
            tenant_id="t-001",
            name="my-key",
            key_prefix="ros_a1b2c3d4",
            is_active=True,
            created_at=now,
            expires_at=None,
            key="ros_a1b2c3d4_somesecretvalue",
        )
        assert resp.key == "ros_a1b2c3d4_somesecretvalue"


# ── Key generation helpers ────────────────────────────────────────────────────


class TestGenerateApiKey:
    def test_returns_three_parts(self) -> None:
        full_key, prefix, hash_hex = generate_api_key()
        assert full_key
        assert prefix
        assert hash_hex

    def test_key_starts_with_ros(self) -> None:
        full_key, _, _ = generate_api_key()
        assert full_key.startswith("ros_")

    def test_prefix_format(self) -> None:
        _, prefix, _ = generate_api_key()
        assert prefix.startswith("ros_")
        assert len(prefix) == 12  # "ros_" + 8 hex chars

    def test_prefix_is_start_of_full_key(self) -> None:
        full_key, prefix, _ = generate_api_key()
        assert full_key.startswith(prefix)

    def test_hash_is_sha256_of_key(self) -> None:
        full_key, _, hash_hex = generate_api_key()
        expected = hashlib.sha256(full_key.encode()).hexdigest()
        assert hash_hex == expected

    def test_hash_is_64_chars(self) -> None:
        _, _, hash_hex = generate_api_key()
        assert len(hash_hex) == 64

    def test_unique_keys_each_call(self) -> None:
        key1, _, _ = generate_api_key()
        key2, _, _ = generate_api_key()
        assert key1 != key2

    def test_unique_hashes_each_call(self) -> None:
        _, _, hash1 = generate_api_key()
        _, _, hash2 = generate_api_key()
        assert hash1 != hash2


class TestHashApiKey:
    def test_sha256_of_key(self) -> None:
        key = "ros_a1b2c3d4_some_secret"
        expected = hashlib.sha256(key.encode()).hexdigest()
        assert hash_api_key(key) == expected

    def test_deterministic(self) -> None:
        key = "ros_stable_key"
        assert hash_api_key(key) == hash_api_key(key)

    def test_different_keys_produce_different_hashes(self) -> None:
        assert hash_api_key("key1") != hash_api_key("key2")


# ── Auth middleware helpers ───────────────────────────────────────────────────


class TestExtractPrefix:
    def test_valid_key(self) -> None:
        from retrieval_os.api.middleware.auth import _extract_prefix

        key = "ros_a1b2c3d4_thisistherest"
        assert _extract_prefix(key) == "ros_a1b2c3d4"

    def test_missing_ros_prefix(self) -> None:
        from retrieval_os.api.middleware.auth import _extract_prefix

        assert _extract_prefix("sk-openai-key") is None

    def test_too_few_segments(self) -> None:
        from retrieval_os.api.middleware.auth import _extract_prefix

        assert _extract_prefix("ros_onlyone") is None

    def test_empty_string(self) -> None:
        from retrieval_os.api.middleware.auth import _extract_prefix

        assert _extract_prefix("") is None
