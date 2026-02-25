"""S3-compatible object storage client (boto3).

Uses asyncio.to_thread to avoid blocking the event loop on boto3 sync calls.
Swap S3_ENDPOINT_URL to AWS/GCS in production — no code changes needed.
"""

import asyncio
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from retrieval_os.core.config import settings


def _make_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        region_name=settings.s3_region,
        config=Config(signature_version="s3v4"),
    )


def get_s3_client() -> Any:
    """Returns a new boto3 S3 client (cheap to construct, not pooled)."""
    return _make_client()


async def check_s3_connection() -> bool:
    """Returns True if the bucket is reachable."""
    def _check() -> bool:
        try:
            _make_client().head_bucket(Bucket=settings.s3_bucket_name)
            return True
        except Exception:
            return False

    return await asyncio.to_thread(_check)


async def ensure_bucket_exists() -> None:
    """Creates the default bucket if it does not exist."""
    def _ensure() -> None:
        client = _make_client()
        try:
            client.head_bucket(Bucket=settings.s3_bucket_name)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
                client.create_bucket(Bucket=settings.s3_bucket_name)
            else:
                raise

    await asyncio.to_thread(_ensure)


async def object_exists(key: str) -> bool:
    """Returns True if the object exists in the default bucket."""
    def _check() -> bool:
        try:
            _make_client().head_object(Bucket=settings.s3_bucket_name, Key=key)
            return True
        except ClientError:
            return False

    return await asyncio.to_thread(_check)


async def get_object_metadata(key: str) -> dict[str, Any]:
    """Returns ContentLength and ETag for an object."""
    def _head() -> dict[str, Any]:
        resp = _make_client().head_object(Bucket=settings.s3_bucket_name, Key=key)
        return {
            "size_bytes": resp["ContentLength"],
            "etag": resp["ETag"].strip('"'),
        }

    return await asyncio.to_thread(_head)
