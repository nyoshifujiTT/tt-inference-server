# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Lightweight async diarization job store (pyannoteAI job model).

The pyannoteAI cloud API is asynchronous: ``POST /v1/diarize`` returns a
``JobCreated`` (jobId + status) and the client polls ``GET /v1/jobs/{jobId}``
for a ``DiarizationJob`` (jobId, status, createdAt, updatedAt, output). See
https://docs.pyannote.ai/openapi.json.

This store is the self-hosted equivalent. It is intentionally *not* the generic
``job_manager`` used by video: that stores ``request.model_dump()`` (which would
serialize the entire audio payload) and persists file-path results, neither of
which fits a JSON-output diarization job. This store keeps only a small job
record and the JSON output in memory, with TTL expiry.

Status values are exactly the pyannoteAI ``JobStatus`` enum:
``pending|created|succeeded|canceled|failed|running``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Optional

# pyannoteAI JobStatus enum values (https://docs.pyannote.ai/openapi.json)
STATUS_CREATED = "created"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELED = "canceled"
TERMINAL_STATUSES = frozenset({STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELED})


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class DiarizationJob:
    job_id: str
    status: str = STATUS_CREATED
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    output: Optional[dict] = None
    warning: Optional[str] = None
    _created_monotonic: float = field(default_factory=time.time)

    def touch(self, status: str) -> None:
        self.status = status
        self.updated_at = _now_iso()

    def created_dict(self) -> dict:
        """The pyannoteAI ``JobCreated`` shape (jobId + status [+ warning])."""
        d = {"jobId": self.job_id, "status": self.status}
        if self.warning:
            d["warning"] = self.warning
        return d

    def job_dict(self) -> dict:
        """The pyannoteAI ``DiarizationJob`` shape returned by GET /v1/jobs/{id}."""
        d = {
            "jobId": self.job_id,
            "status": self.status,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
        if self.output is not None:
            d["output"] = self.output
        return d


class DiarizationJobStore:
    """In-memory diarization job store with TTL expiry (default 24h)."""

    def __init__(self, retention_seconds: int = 86400):
        self._jobs: Dict[str, DiarizationJob] = {}
        self._lock = Lock()
        self._retention = retention_seconds

    def create(self) -> DiarizationJob:
        self._sweep_locked_wrapper()
        job = DiarizationJob(job_id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[DiarizationJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if self._is_expired(job):
                del self._jobs[job_id]
                return None
            return job

    def set_running(self, job_id: str) -> None:
        with self._lock:
            j = self._jobs.get(job_id)
            if j:
                j.touch(STATUS_RUNNING)

    def set_succeeded(
        self, job_id: str, output: dict, warning: Optional[str] = None
    ) -> None:
        with self._lock:
            j = self._jobs.get(job_id)
            if j:
                j.output = output
                j.warning = warning
                j.touch(STATUS_SUCCEEDED)

    def set_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            j = self._jobs.get(job_id)
            if j:
                j.output = {"error": error}
                j.touch(STATUS_FAILED)

    def _is_expired(self, job: DiarizationJob) -> bool:
        if self._retention <= 0:
            return False
        return (time.time() - job._created_monotonic) > self._retention

    def _sweep_locked_wrapper(self) -> None:
        with self._lock:
            expired = [jid for jid, j in self._jobs.items() if self._is_expired(j)]
            for jid in expired:
                del self._jobs[jid]


_STORE: Optional[DiarizationJobStore] = None


def get_job_store() -> DiarizationJobStore:
    global _STORE
    if _STORE is None:
        import os

        _STORE = DiarizationJobStore(
            retention_seconds=int(
                os.environ.get("DIARIZATION_JOB_RETENTION_SECONDS", "86400")
            )
        )
    return _STORE


def post_webhook(url: str, payload: dict, timeout: float = 10.0) -> bool:
    """Best-effort webhook POST of a job payload. Returns True on 2xx."""
    import json
    import urllib.request

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:  # noqa: BLE001 - webhook delivery is best-effort
        return False
