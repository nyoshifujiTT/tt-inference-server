# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Resolve a pyannoteAI ``DiarizeRequest.url`` to raw audio bytes.

pyannoteAI accepts two url forms (https://docs.pyannote.ai/openapi.json):
  - a public ``http(s)://`` URL of the audio file, and
  - a ``media://<object-key>`` reference to a file staged via the media API.

This resolver fetches either into memory. http(s) downloads are capped by a
max-size to avoid unbounded memory use; unsupported schemes raise
``AudioUrlError`` (surfaced by the router as HTTP 400).
"""

from __future__ import annotations

import urllib.request
from typing import Optional

from utils.media_storage import (
    MEDIA_SCHEME,
    MediaStorage,
    MediaStorageError,
    get_media_storage,
)

_HTTP_SCHEMES = ("http://", "https://")
DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50 MB, matches MAX_AUDIO_SIZE_BYTES default


class AudioUrlError(ValueError):
    """Raised for an unsupported or unfetchable audio url."""


def resolve_audio_url(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = 30.0,
    storage: Optional[MediaStorage] = None,
) -> bytes:
    """Return the audio bytes referenced by ``url`` (http(s) or media://)."""
    if not isinstance(url, str) or not url:
        raise AudioUrlError("audio url must be a non-empty string")

    if url.startswith(MEDIA_SCHEME):
        st = storage or get_media_storage()
        try:
            return st.get(url)
        except MediaStorageError as e:
            raise AudioUrlError(str(e)) from e

    if url.startswith(_HTTP_SCHEMES):
        return _download_http(url, max_bytes=max_bytes, timeout=timeout)

    raise AudioUrlError(
        f"unsupported audio url scheme: {url!r}; use http(s):// or media://"
    )


def _download_http(url: str, *, max_bytes: int, timeout: float) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            if status != 200:
                raise AudioUrlError(f"audio url returned HTTP {status}: {url!r}")
            # read one byte past the cap to detect oversize without loading it all
            data = resp.read(max_bytes + 1)
    except AudioUrlError:
        raise
    except Exception as e:  # noqa: BLE001 - network/DNS/timeout -> 400
        raise AudioUrlError(f"failed to fetch audio url {url!r}: {e}") from e
    if len(data) > max_bytes:
        raise AudioUrlError(
            f"audio at {url!r} exceeds the maximum size of {max_bytes} bytes"
        )
    if not data:
        raise AudioUrlError(f"audio url returned no data: {url!r}")
    return data
