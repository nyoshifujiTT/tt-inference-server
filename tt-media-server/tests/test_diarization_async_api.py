# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os
import time

os.environ["NO_AUTH"] = "1"

import contextlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import utils.media_downloader as _media_downloader
from domain.diarization_response import DiarizationResponse, DiarizationSegment
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.diarization_auth import auth_headers
from open_ai_api import diarization, media
from resolver.service_resolver import service_resolver


# Captured before the autouse fixture stubs it out, so the one test that wants
# a real fetch can put it back.
_REAL_DOWNLOAD = _media_downloader.download_media_url


class _FakeService:
    async def process_request(self, request):
        return DiarizationResponse(
            segments=[DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0)],
            exclusiveDiarization=None,
        )


class _RecordingService(_FakeService):
    """Keeps the request so a test can assert on the audio that arrived."""

    def __init__(self):
        self.last = None

    async def process_request(self, request):
        self.last = request
        return await super().process_request(request)


@contextlib.contextmanager
def _object_store():
    """A minimal S3-shaped store: PUT /bucket/key stores, GET returns.

    Not an S3 implementation and not trying to be — it exists so the staged
    upload really leaves this process over HTTP and really comes back, which a
    stub cannot show. Pre-signed urls are signed locally by boto3, so the
    signature is exercised on the way out even though this handler does not
    verify it.
    """
    objects: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self):  # noqa: N802 - BaseHTTPRequestHandler's naming
            length = int(self.headers.get("content-length", 0))
            objects[self.path.split("?", 1)[0]] = self.rfile.read(length)
            self.send_response(200)
            self.end_headers()

        def do_GET(self):  # noqa: N802
            body = objects.get(self.path.split("?", 1)[0])
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", objects
    finally:
        server.shutdown()
        server.server_close()


def _make_client(fake):
    app = FastAPI()
    app.include_router(diarization.async_router, prefix="/v1")
    app.dependency_overrides[service_resolver] = lambda: fake
    # Authenticate every request: NO_AUTH is only honoured when this module wins
    # the import race against security.api_key_checker (see diarization_auth).
    return TestClient(app, headers=auth_headers())


@pytest.fixture(autouse=True)
def _media_and_store(monkeypatch):
    """Configure object storage and stub the fetch of the staged object.

    media:// keys are signed into a GET url on the storage service and read
    back through media_downloader, so there is no local file to seed; these
    cases are about the job API, not about the network.
    """
    from unittest.mock import AsyncMock

    import utils.diarization_jobs as dj
    import utils.media_downloader as md
    import utils.media_object_storage as mos

    # Patch through the modules under test, not ``config.settings``: other test
    # modules replace ``sys.modules["config.settings"]`` with a Mock at import
    # time and never restore it, so patching there reaches the Mock while the
    # code keeps its reference to the real settings object (the same reason
    # tests/test_video_api.py patches ``open_ai_api.video.settings``).
    settings = mos.settings

    monkeypatch.setattr(settings, "media_storage_endpoint", "https://s3.test", False)
    monkeypatch.setattr(settings, "media_storage_bucket", "media", False)
    monkeypatch.setattr(settings, "media_storage_access_key", "key", False)
    monkeypatch.setattr(settings, "media_storage_secret_key", "secret", False)
    mos.reset_client()
    monkeypatch.setattr(
        md, "download_media_url", AsyncMock(return_value=b"RIFFxxxxWAVE")
    )
    dj._STORE = None
    yield
    mos.reset_client()
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
    job_id = client.post("/v1/diarize", json={"url": "media://audio.wav"}).json()[
        "jobId"
    ]
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


def test_media_staged_audio_flows_through_the_job_api(monkeypatch):
    """The documented flow, end to end against a real object store.

    Stand up an S3-shaped HTTP endpoint, declare a key, PUT the bytes to the
    url the server signs, then create a job against the media:// url and let
    the server read the object back. Nothing is stubbed between the declaration
    and the audio arriving at the service, which is the whole point: the bytes
    have to reach the model without ever passing through this API.

    media_url_allowed_domains is deliberately left empty: the storage host the
    server signed the url against is allowed on its own, so the documented flow
    works without an operator widening the allowlist by hand.
    """
    import utils.diarization_jobs as dj
    import utils.media_downloader as md
    import utils.media_object_storage as mos

    # Patch through the modules under test, not ``config.settings``: other test
    # modules replace ``sys.modules["config.settings"]`` with a Mock at import
    # time and never restore it, so patching there reaches the Mock while the
    # code keeps its reference to the real settings object (the same reason
    # tests/test_video_api.py patches ``open_ai_api.video.settings``).
    settings = mos.settings

    with _object_store() as (endpoint, objects):
        # undo the autouse stub: this case is exactly about the real fetch
        monkeypatch.setattr(md, "download_media_url", _REAL_DOWNLOAD, False)
        monkeypatch.setattr(settings, "media_storage_endpoint", endpoint, False)
        monkeypatch.setattr(settings, "media_storage_bucket", "media", False)
        monkeypatch.setattr(settings, "media_storage_access_key", "key", False)
        monkeypatch.setattr(settings, "media_storage_secret_key", "secret", False)
        mos.reset_client()
        dj._STORE = None

        app = FastAPI()
        app.include_router(media.router, prefix="/v1/media")
        app.include_router(diarization.async_router, prefix="/v1")
        service = _RecordingService()
        app.dependency_overrides[service_resolver] = lambda: service
        client = TestClient(app, headers=auth_headers())

        declared = client.post(
            "/v1/media/input", json={"url": "media://sess/staged.wav"}
        )
        assert declared.status_code == 201, declared.text
        put_url = declared.json()["url"]
        # the upload goes to the storage service, not back into this API
        assert put_url.startswith(endpoint)

        assert httpx.put(put_url, content=b"RIFFxxxxWAVE").status_code == 200
        assert objects["/media/sess/staged.wav"] == b"RIFFxxxxWAVE"

        created = client.post("/v1/diarize", json={"url": "media://sess/staged.wav"})
        assert created.status_code == 201, created.text

        job = _poll(client, created.json()["jobId"])
        assert job["status"] == "succeeded", job
        assert "diarization" in job["output"], job["output"]
        # the staged bytes are what reached the model
        assert bytes(service.last.file) == b"RIFFxxxxWAVE"

    mos.reset_client()
    dj._STORE = None
