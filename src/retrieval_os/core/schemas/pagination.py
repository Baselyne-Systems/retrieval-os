"""Cursor-based pagination schema used across all list endpoints."""

from pydantic import BaseModel


class CursorPage[T](BaseModel):
    """
    Cursor-paginated response.

    The `cursor` field is an opaque token (base64-encoded UUIDv7) for the next
    page. Clients pass it back as ?cursor= on the next request. None means
    there are no more pages.
    """

    items: list[T]
    total: int
    cursor: str | None = None
    has_more: bool
