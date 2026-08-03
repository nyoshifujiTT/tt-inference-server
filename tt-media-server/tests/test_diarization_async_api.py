# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os
import time

os.environ["NO_AUTH"] = "1"

import pytest
from domain.diarization_response import DiarizationResponse, DiarizationSegment
from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_ai_api import diarization
from resolver.service_resolver import service_resolver


class _FakeService:
    async def process_request(self, request):
        return DiarizationResponse(
            segments=[DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0)],
            exclusiveDiarization=None,
        )


def _make_client(fake):
    app = FastAPI()
    app.include_router(diarization.async_router, prefix="/v1")
    app.dependency_overrides[service_resolver] = lambda: fake
    return TestClient(app)


@pytest.fixture(autouse=True)
def _media_and_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_INPUT_DIR", str(tmp_path))
    import utils.media_storage as ms
    import utils.diarization_jobs as dj

    ms._STORAGE = None
    ms.get_media_storage().put("audio.wav", b"RIFFxxxxWAVE")
    dj._STORE = None
    yield
    ms._STORAGE = None
    dj._STORE = None


def _poll(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/v1/jobs/{job_id}")
        assert r.status_code == 200, r.text
        if r.json()["status"] in ("succeeded", "failed", "canceled"):
            return r.json()
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def test_create_job_returns_jobcreated():
    resp = _make_client(_FakeService()).post(
        "/v1/diarize", json={"url": "media://audio.wav"}
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["jobId"]
    assert body["status"] in ("created", "running", "succeeded")


def test_job_completes_with_output():
    client = _make_client(_FakeService())
    job_id = client.post("/v1/diarize", json={"url": "media://audio.wav"}).json()["jobId"]
    final = _poll(client, job_id)
    assert final["status"] == "succeeded"
    assert final["jobId"] == job_id
    assert "createdAt" in final and "updatedAt" in final
    assert final["output"]["diarization"][0]["speaker"] == "SPEAKER_00"


def test_get_unknown_job_404():
    r = _make_client(_FakeService()).get("/v1/jobs/does-not-exist")
    assert r.status_code == 404


def test_create_job_requires_url():
    r = _make_client(_FakeService()).post("/v1/diarize", json={"numSpeakers": 2})
    assert r.status_code == 400


def test_create_job_rejects_precision2_option():
    r = _make_client(_FakeService()).post(
        "/v1/diarize", json={"url": "media://audio.wav", "transcription": True}
    )
    assert r.status_code == 400
    assert "precision-2" in r.json()["detail"]


def test_webhook_is_called(monkeypatch):
    calls = {}

    def fake_post(url, payload, timeout=10.0):
        calls["url"] = url
        calls["payload"] = payload
        return True

    import utils.diarization_jobs as dj

    monkeypatch.setattr(dj, "post_webhook", fake_post)

    client = _make_client(_FakeService())
    job_id = client.post(
        "/v1/diarize",
        json={"url": "media://audio.wav", "webhook": "http://example/hook"},
    ).json()["jobId"]
    _poll(client, job_id)
    # give the background webhook thread a moment
    deadline = time.time() + 2.0
    while time.time() < deadline and "url" not in calls:
        time.sleep(0.02)
    assert calls.get("url") == "http://example/hook"
    assert calls["payload"]["jobId"] == job_id
    assert calls["payload"]["status"] == "succeeded"
