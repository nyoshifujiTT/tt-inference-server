# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os

os.environ["NO_AUTH"] = "1"

from domain.diarization_response import DiarizationResponse, DiarizationSegment
from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_ai_api import diarization
from resolver.service_resolver import service_resolver


class _FakeService:
    """Stands in for DiarizationService: no ffmpeg/pyannote, records the request."""

    def __init__(self):
        self.last = None

    async def process_request(self, request):
        self.last = request
        return DiarizationResponse(
            segments=[DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_00")],
            exclusiveDiarization=[
                DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_00")
            ],
        )


def _make_client(fake):
    app = FastAPI()
    app.include_router(diarization.router, prefix="/v1/audio")
    app.dependency_overrides[service_resolver] = lambda: fake
    return TestClient(app)


def test_diarize_endpoint_returns_pyannoteai_shape():
    fake = _FakeService()
    client = _make_client(fake)
    resp = client.post(
        "/v1/audio/diarize",
        files={"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")},
        data={"num_speakers": "2", "exclusive": "true"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["segments"][0] == {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
    assert body["exclusiveDiarization"][0]["speaker"] == "SPEAKER_00"
    # request parsing wired the speaker hint through
    assert fake.last.num_speakers == 2
    assert fake.last.exclusive is True
    assert isinstance(fake.last.file, (bytes, bytearray))


def test_diarize_endpoint_default_exclusive_and_no_hints():
    fake = _FakeService()
    client = _make_client(fake)
    resp = client.post(
        "/v1/audio/diarize",
        files={"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    assert fake.last.num_speakers is None
    assert fake.last.exclusive is True
