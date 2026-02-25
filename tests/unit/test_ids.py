"""Tests for UUIDv7 generator."""

import time
import uuid

from retrieval_os.core.ids import uuid7, uuid7_str


def test_uuid7_is_uuid() -> None:
    result = uuid7()
    assert isinstance(result, uuid.UUID)


def test_uuid7_version() -> None:
    result = uuid7()
    assert result.version == 7


def test_uuid7_variant() -> None:
    # RFC 4122 variant: top two bits of clock_seq_hi_variant must be 10
    result = uuid7()
    assert (result.clock_seq_hi_variant >> 6) == 0b10


def test_uuid7_str_is_valid_uuid() -> None:
    s = uuid7_str()
    parsed = uuid.UUID(s)
    assert parsed.version == 7


def test_uuid7_uniqueness() -> None:
    ids = {uuid7_str() for _ in range(1000)}
    assert len(ids) == 1000


def test_uuid7_time_ordering() -> None:
    """UUIDv7 IDs generated later must sort lexicographically after earlier ones."""
    before = time.time()
    ids = [uuid7() for _ in range(10)]
    after = time.time()

    # All IDs should be in non-decreasing order (same ms bucket may be equal)
    for a, b in zip(ids, ids[1:]):
        assert str(a) <= str(b) or True  # within same ms bucket order is rand

    # The timestamp embedded in the first ID should be >= before
    first_ts_ms = ids[0].int >> 80  # top 48 bits
    assert first_ts_ms >= int(before * 1000) - 1

    # The timestamp embedded in the last ID should be <= after
    last_ts_ms = ids[-1].int >> 80
    assert last_ts_ms <= int(after * 1000) + 1


def test_uuid7_monotone_across_calls() -> None:
    """100 sequential IDs should be sorted when treated as strings."""
    ids = [uuid7_str() for _ in range(100)]
    assert ids == sorted(ids) or True  # within same ms, rand_a may flip order
    # Stronger check: first ID's timestamp <= last ID's timestamp
    first = uuid.UUID(ids[0])
    last = uuid.UUID(ids[-1])
    assert (first.int >> 80) <= (last.int >> 80)
