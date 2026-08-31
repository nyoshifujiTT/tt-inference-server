# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os

os.environ["NO_AUTH"] = "1"

import pytest
from domain.diarization_response import DiarizationResponse, DiarizationSegment
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.diarization_auth import auth_headers
from open_ai_api import diarization
from resolver.service_resolver import service_resolver


class _FakeService:
    """Stands in for DiarizationService: records the request, returns a fixed result."""

    def __init__(self):
        self.last = None

    async def process_request(self, request):
        self.last = request
        return DiarizationResponse(
            segments=[DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0)],
            exclusiveDiarization=[
                DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0)
            ],
        )


def _make_client(fake):
    """Client over the official job API.

    There is no synchronous route to test: the official spec has none, so the
    one this service used to publish at /v1/audio/diarize was removed. Request
    validation is the same code either way -- both routes parse the body
    through ``_read_json_body`` + ``_build_request_from_body`` -- so these cases
    exercise it through POST /v1/diarize.
    """
    app = FastAPI()
    app.include_router(diarization.async_router, prefix="/v1")
    app.dependency_overrides[service_resolver] = lambda: fake
    # Authenticate every request: NO_AUTH is only honoured when this module wins
    # the import race against security.api_key_checker (see diarization_auth).
    return TestClient(app, headers=auth_headers())


def _diarize(client, body):
    """POST a diarize body; return the create-job response."""
    return client.post("/v1/diarize", json=body)


@pytest.fixture(autouse=True)
def _media(tmp_path, monkeypatch):
    """Back the media:// resolver with a temp store and stage a dummy audio object."""
    monkeypatch.setenv("MEDIA_INPUT_DIR", str(tmp_path))
    import utils.media_storage as ms

    ms._STORAGE = None
    ms.get_media_storage().put("audio.wav", b"RIFFxxxxWAVE")
    yield
    ms._STORAGE = None


_URL = "media://audio.wav"


def test_diarize_endpoint_returns_pyannoteai_shape():
    fake = _FakeService()
    client = _make_client(fake)
    resp = client.post(
        "/v1/diarize",
        json={"url": _URL, "numSpeakers": 2, "exclusive": True},
    )
    assert resp.status_code == 201, resp.text
    job_id = resp.json()["jobId"]

    body = client.get(f"/v1/jobs/{job_id}").json()["output"]
    assert body["diarization"][0] == {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}
    assert body["exclusiveDiarization"][0]["speaker"] == "SPEAKER_00"
    assert fake.last.num_speakers == 2
    assert fake.last.exclusive is True
    assert isinstance(fake.last.file, (bytes, bytearray))


def test_diarize_endpoint_default_exclusive_and_no_hints():
    fake = _FakeService()
    client = _make_client(fake)
    resp = client.post("/v1/diarize", json={"url": _URL})
    assert resp.status_code == 201, resp.text
    assert fake.last.num_speakers is None
    assert fake.last.exclusive is True


def test_diarize_requires_url():
    fake = _FakeService()
    resp = _make_client(fake).post("/v1/diarize", json={"numSpeakers": 2})
    assert resp.status_code == 400, resp.text
    assert "url" in resp.json()["detail"]
    assert fake.last is None


# The official API documents 400, 401 and 429 -- never 415. A hand-rolled
# Content-Type check here used to answer 415, which reads as "that format is not
# supported yet" and sent people hunting for a multipart upload that the
# official API does not offer either. Letting FastAPI validate the body gives
# its own 422 for anything that is not a JSON object.
_OFFICIAL_ERROR_STATUSES = {400, 401, 429, 422}


def test_a_non_json_body_is_rejected_without_inventing_a_status():
    fake = _FakeService()
    resp = _make_client(fake).post(
        "/v1/diarize",
        content=b"not json",
        headers={"content-type": "text/plain"},
    )
    assert resp.status_code == 422, resp.text
    assert fake.last is None


def test_a_multipart_upload_is_rejected_without_inventing_a_status():
    """There is no multipart path to invent a bespoke 415 for.

    The official API takes audio by url only: either http(s) or a media:// key
    staged through POST /v1/media/input. A 415 here implied a format this
    server had merely not implemented yet.
    """
    fake = _FakeService()
    resp = _make_client(fake).post(
        "/v1/diarize",
        files={"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")},
    )
    assert resp.status_code == 422, resp.text
    assert resp.status_code in _OFFICIAL_ERROR_STATUSES
    assert fake.last is None


def test_diarize_bad_media_url():
    fake = _FakeService()
    resp = _make_client(fake).post(
        "/v1/diarize", json={"url": "media://does-not-exist.wav"}
    )
    assert resp.status_code == 400, resp.text
    assert fake.last is None


def test_diarize_unsupported_scheme():
    fake = _FakeService()
    resp = _make_client(fake).post("/v1/diarize", json={"url": "ftp://h/a.wav"})
    assert resp.status_code == 400, resp.text
    assert fake.last is None


def test_diarize_accepts_served_community_1_model():
    fake = _FakeService()
    client = _make_client(fake)
    resp = client.post("/v1/diarize", json={"url": _URL, "model": "community-1"})
    assert resp.status_code == 201, resp.text
    output = client.get(f"/v1/jobs/{resp.json()['jobId']}").json()["output"]
    assert output["diarization"][0]["speaker"] == "SPEAKER_00"


def test_diarize_rejects_precision_2_model():
    fake = _FakeService()
    resp = _make_client(fake).post(
        "/v1/diarize", json={"url": _URL, "model": "precision-2"}
    )
    assert resp.status_code == 400, resp.text
    assert "not served" in resp.json()["detail"]
    assert fake.last is None


def test_diarize_rejects_unknown_model():
    fake = _FakeService()
    resp = _make_client(fake).post(
        "/v1/diarize", json={"url": _URL, "model": "totally-made-up"}
    )
    assert resp.status_code == 400, resp.text
    assert "unknown diarization model" in resp.json()["detail"]
    assert fake.last is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence", True),
        ("turnLevelConfidence", True),
        ("transcription", True),
        ("transcriptionConfig", {"model": "x"}),
    ],
)
def test_diarize_rejects_precision2_only_options(field, value):
    fake = _FakeService()
    resp = _make_client(fake).post("/v1/diarize", json={"url": _URL, field: value})
    assert resp.status_code == 400, resp.text
    assert "precision-2" in resp.json()["detail"]
    assert fake.last is None


def test_diarize_allows_precision2_options_when_false():
    fake = _FakeService()
    resp = _make_client(fake).post(
        "/v1/diarize",
        json={"url": _URL, "confidence": False, "transcription": False},
    )
    assert resp.status_code == 201, resp.text
