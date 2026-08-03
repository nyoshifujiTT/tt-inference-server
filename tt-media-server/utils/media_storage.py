# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Local temporary media storage for pyannoteAI-style ``media://`` inputs.

The pyannoteAI cloud API lets a client stage a private file in temporary storage
before diarizing it (https://docs.pyannote.ai/api-reference/upload-media):

  1. ``POST /v1/media/input`` with ``{"url": "media://<object-key>"}`` declares an
     object key and returns a pre-signed URL to ``PUT`` the bytes to.
  2. The client ``PUT``\\s the file to that URL.
  3. The ``media://<object-key>`` URL is then passed as ``DiarizeRequest.url``.

This module is the self-hosted equivalent: it maps ``media://<key>`` to a file
under a local storage directory, hands out a ``PUT`` URL served by this same
server, and expires objects after a retention window (default 24h, matching the
pyannoteAI "at least 24 hours" guarantee).
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional

MEDIA_SCHEME = "media://"
# object-key charset per the pyannoteAI GetMediaUploadUrl pattern
# (media://[a-zA-Z0-9-_./]+); we forbid any path traversal.
_OBJECT_KEY_RE = re.compile(r"^[A-Za-z0-9\-_./]+$")


class MediaStorageError(ValueError):
    """Raised for malformed media:// urls or missing/expired objects."""


def parse_media_key(url: str) -> str:
    """Extract and validate the object-key from a ``media://<key>`` URL."""
    if not isinstance(url, str) or not url.startswith(MEDIA_SCHEME):
        raise MediaStorageError(f"not a media:// url: {url!r}")
    key = url[len(MEDIA_SCHEME):]
    if not key or not _OBJECT_KEY_RE.match(key):
        raise MediaStorageError(f"invalid media object-key: {key!r}")
    # defense in depth against traversal even though the charset forbids '..'
    if ".." in key.split("/"):
        raise MediaStorageError(f"invalid media object-key (traversal): {key!r}")
    return key


@dataclass
class MediaStorage:
    """Filesystem-backed store for ``media://`` objects.

    Not a general blob store: keys are namespaced under ``root`` and objects
    older than ``retention_seconds`` are treated as expired (and removed on
    access / sweep).
    """

    root: str
    retention_seconds: int = 86400

    def __post_init__(self) -> None:
        os.makedirs(self.root, exist_ok=True)

    def _path_for(self, key: str) -> str:
        path = os.path.normpath(os.path.join(self.root, key))
        root_abs = os.path.abspath(self.root)
        if os.path.commonpath([root_abs, os.path.abspath(path)]) != root_abs:
            raise MediaStorageError(f"object-key escapes storage root: {key!r}")
        return path

    def declare(self, url: str) -> str:
        """Register an object-key (from a media:// url); return the object-key.

        Does not create the file yet — the client PUTs bytes next. We only make
        the parent directory so the later PUT can write.
        """
        key = parse_media_key(url)
        path = self._path_for(key)
        os.makedirs(os.path.dirname(path) or self.root, exist_ok=True)
        return key

    def put(self, key: str, data: bytes) -> None:
        """Store bytes for a previously-declared (or new) object-key."""
        if not _OBJECT_KEY_RE.match(key):
            raise MediaStorageError(f"invalid media object-key: {key!r}")
        path = self._path_for(key)
        os.makedirs(os.path.dirname(path) or self.root, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def get(self, url_or_key: str) -> bytes:
        """Read bytes for a ``media://<key>`` url (or bare key). Expiry-aware."""
        key = (
            parse_media_key(url_or_key)
            if url_or_key.startswith(MEDIA_SCHEME)
            else url_or_key
        )
        path = self._path_for(key)
        if not os.path.exists(path):
            raise MediaStorageError(f"media object not found: {key!r}")
        if self._is_expired(path):
            self._safe_unlink(path)
            raise MediaStorageError(f"media object expired: {key!r}")
        with open(path, "rb") as f:
            return f.read()

    def _is_expired(self, path: str) -> bool:
        if self.retention_seconds <= 0:
            return False
        return (time.time() - os.path.getmtime(path)) > self.retention_seconds

    @staticmethod
    def _safe_unlink(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

    def sweep(self) -> int:
        """Delete all expired objects; return how many were removed."""
        removed = 0
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                p = os.path.join(dirpath, name)
                if self._is_expired(p):
                    self._safe_unlink(p)
                    removed += 1
        return removed


_STORAGE: Optional[MediaStorage] = None


def get_media_storage() -> MediaStorage:
    """Process-wide MediaStorage singleton.

    Root and retention come from settings/env so deployments can point it at a
    writable tmpfs; defaults to ``/tmp/tt_media_input`` with 24h retention.
    """
    global _STORAGE
    if _STORAGE is None:
        root = os.environ.get("MEDIA_INPUT_DIR", "/tmp/tt_media_input")
        retention = int(os.environ.get("MEDIA_INPUT_RETENTION_SECONDS", "86400"))
        _STORAGE = MediaStorage(root=root, retention_seconds=retention)
    return _STORAGE
