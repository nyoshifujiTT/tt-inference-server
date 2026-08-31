# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""S3 client for the pyannoteAI two-step upload (``media://`` object keys).

The official flow (https://docs.pyannote.ai/api-reference/upload-media) is:

  1. ``POST /v1/media/input`` with ``{"url": "media://<key>"}`` declares a key
     and answers with a **pre-signed URL on the storage service**;
  2. the client ``PUT``\\s the bytes straight to that URL, never through the
     inference API;
  3. ``media://<key>`` is then passed as ``DiarizeRequest.url``.

This module is the storage *client* for that flow. The server signs URLs and
reads objects back; it never receives, stores or serves the bytes itself. That
is the whole point of step 2: a multi-hundred-megabyte upload must not occupy
the process that is scheduling device work. Serving the PUT here — which is
what this server used to do — also put an upload path inside the official
``/v1/media`` namespace that no pyannoteAI client would ever call.

Any S3-compatible endpoint works because the only thing used is SigV4
pre-signing. For a self-hosted deployment RustFS is the one to reach for: it is
Apache-2.0 (MinIO's community edition is AGPL and archived, Garage is AGPL), a
single container configured by two environment variables, and it implements S3
lifecycle natively, so the "at least 24 hours" retention the official API
promises is a bucket policy rather than a sweeper thread in here.

With no storage configured the endpoint answers 501 and points at the two
inputs that need no storage at all: an ``http(s)://`` URL, or inline base64.
"""

from __future__ import annotations

import re
import threading
from typing import Optional
from urllib.parse import urlsplit

from config.settings import settings

MEDIA_SCHEME = "media://"
# The object-key charset of the official GetMediaUploadUrl schema
# (media://[a-zA-Z0-9-_./]+). Traversal is outside it by construction; the
# explicit ".." check below is there so a future widening of the charset does
# not quietly become a traversal.
_OBJECT_KEY_RE = re.compile(r"^[A-Za-z0-9\-_./]+$")

_client = None
_client_lock = threading.Lock()


class MediaStorageError(ValueError):
    """Malformed ``media://`` url, or a bad object key."""


class MediaStorageNotConfigured(RuntimeError):
    """No object storage is configured for this deployment."""


def parse_media_key(url: str) -> str:
    """Extract and validate the object key of a ``media://<key>`` URL."""
    if not isinstance(url, str) or not url.startswith(MEDIA_SCHEME):
        raise MediaStorageError(f"not a media:// url: {url!r}")
    key = url[len(MEDIA_SCHEME) :]
    if not key or not _OBJECT_KEY_RE.match(key):
        raise MediaStorageError(f"invalid media object-key: {key!r}")
    if ".." in key.split("/"):
        raise MediaStorageError(f"invalid media object-key (traversal): {key!r}")
    return key


def is_configured() -> bool:
    """True when this deployment has an object-storage endpoint and bucket."""
    return bool(settings.media_storage_endpoint and settings.media_storage_bucket)


def storage_hostname() -> Optional[str]:
    """Hostname of the configured endpoint, for the download allowlist."""
    if not settings.media_storage_endpoint:
        return None
    return urlsplit(settings.media_storage_endpoint).hostname


def _require_configured() -> None:
    if not is_configured():
        raise MediaStorageNotConfigured(
            "This server has no object storage configured, so it cannot hand "
            "out an upload URL. Set MEDIA_STORAGE_ENDPOINT and "
            "MEDIA_STORAGE_BUCKET (any S3-compatible service; RustFS runs as a "
            "single Apache-2.0 container), or send the audio without staging "
            "it: pass an http(s):// url, or the audio itself as inline base64."
        )


def get_client():
    """Process-wide boto3 S3 client for the configured endpoint.

    Built lazily and under a lock: worker startup touches this from more than
    one thread, and botocore session construction is not thread-safe.
    """
    global _client
    _require_configured()
    if _client is None:
        with _client_lock:
            if _client is None:
                import boto3
                from botocore.config import Config

                _client = boto3.client(
                    "s3",
                    endpoint_url=settings.media_storage_endpoint,
                    aws_access_key_id=settings.media_storage_access_key or None,
                    aws_secret_access_key=settings.media_storage_secret_key or None,
                    region_name=settings.media_storage_region,
                    # SigV4 and path-style: self-hosted endpoints are reached by
                    # host or IP, where the virtual-host style boto3 prefers
                    # would put the bucket in a hostname that does not resolve.
                    config=Config(
                        signature_version="s3v4",
                        s3={"addressing_style": "path"},
                    ),
                )
    return _client


def reset_client() -> None:
    """Drop the cached client (settings changed; used by tests)."""
    global _client
    with _client_lock:
        _client = None


def presigned_put_url(url: str) -> str:
    """Sign a PUT URL for a ``media://<key>`` declaration.

    The signature is the only credential the client needs, so the upload works
    from plain ``curl -T`` with no SDK — the same property the official API's
    pre-signed URL has.
    """
    key = parse_media_key(url)
    return get_client().generate_presigned_url(
        "put_object",
        Params={"Bucket": settings.media_storage_bucket, "Key": key},
        ExpiresIn=settings.media_storage_presign_expiry_seconds,
    )


def presigned_get_url(url: str) -> str:
    """Sign a GET URL for a staged ``media://<key>``.

    Signed rather than assembled as a public object URL so the bucket can stay
    private: a deployment should not have to make every staged recording world
    readable for this server to read one back.
    """
    key = parse_media_key(url)
    return get_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.media_storage_bucket, "Key": key},
        ExpiresIn=settings.media_storage_presign_expiry_seconds,
    )
