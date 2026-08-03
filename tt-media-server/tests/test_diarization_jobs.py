# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import time

from utils.diarization_jobs import (
    STATUS_CREATED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    DiarizationJobStore,
)


def test_create_returns_created_job():
    store = DiarizationJobStore()
    job = store.create()
    assert job.status == STATUS_CREATED
    assert job.job_id
    cd = job.created_dict()
    assert cd["jobId"] == job.job_id and cd["status"] == STATUS_CREATED


def test_lifecycle_running_then_succeeded():
    store = DiarizationJobStore()
    job = store.create()
    store.set_running(job.job_id)
    assert store.get(job.job_id).status == STATUS_RUNNING
    store.set_succeeded(job.job_id, {"diarization": []}, warning="w")
    j = store.get(job.job_id)
    assert j.status == STATUS_SUCCEEDED
    d = j.job_dict()
    assert d["jobId"] == job.job_id
    assert d["status"] == STATUS_SUCCEEDED
    assert d["output"] == {"diarization": []}
    assert "createdAt" in d and "updatedAt" in d


def test_failed_sets_error_output():
    store = DiarizationJobStore()
    job = store.create()
    store.set_failed(job.job_id, "boom")
    j = store.get(job.job_id)
    assert j.status == STATUS_FAILED
    assert j.job_dict()["output"] == {"error": "boom"}


def test_get_unknown_returns_none():
    assert DiarizationJobStore().get("nope") is None


def test_expiry_removes_job():
    store = DiarizationJobStore(retention_seconds=1)
    job = store.create()
    # backdate creation beyond retention
    store._jobs[job.job_id]._created_monotonic = time.time() - 10
    assert store.get(job.job_id) is None
