"""UUIDv7 generator — time-ordered, millisecond-precision, sortable PKs."""

import os
import time
import uuid


def uuid7() -> uuid.UUID:
    """
    Generate a UUIDv7.

    Layout (RFC 9562):
      - bits 127-80 : Unix timestamp in milliseconds (48 bits)
      - bits 79-76  : Version = 0b0111 (4 bits)
      - bits 75-64  : rand_a (12 bits)
      - bits 63-62  : Variant = 0b10 (2 bits)
      - bits 61-0   : rand_b (62 bits)
    """
    timestamp_ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")

    rand_a = (rand >> 62) & 0x0FFF  # 12 bits
    rand_b = rand & 0x3FFFFFFFFFFFFFFF  # 62 bits

    high = (timestamp_ms << 16) | 0x7000 | rand_a
    low = 0x8000000000000000 | rand_b

    return uuid.UUID(int=(high << 64) | low)


def uuid7_str() -> str:
    return str(uuid7())
