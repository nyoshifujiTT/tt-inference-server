# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os

os.environ["NO_AUTH"] = "1"

from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_ai_api import diarization
from resolver.service_resolver import service_resolver


class _FakeService:
    def __init__(self):
        self.last = None

    async def diarized_transcription(self, request, model, language=None, prompt=None):
        self.last = {
            "model": model,
            "language": language,
            "prompt": prompt,
            "num_speakers": request.num_speakers,
        }
        return {
            "task": "transcribe",
            "text": "hello world",
            "duration": 2.0,
            "segments": [
                {
                    "id": 0,
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                },
                {
                    "id": 1,
                    "speaker": "SPEAKER_01",
                    "start": 1.0,
                    "end": 2.0,
                    "text": "world",
                },
            ],
        }


def _client(fake):
    app = FastAPI()
    app.include_router(diarization.router, prefix="/v1/audio")
    app.dependency_overrides[service_resolver] = lambda: fake
    return TestClient(app)


def test_diarized_transcriptions_returns_diarized_json():
    fake = _FakeService()
    resp = _client(fake).post(
        "/v1/audio/diarized-transcriptions",
        files={"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")},
        data={
            "model": "neosophie/Qwen3-ASR-1.7B-JA+pyannote/speaker-diarization-community-1",
            "numSpeakers": "2",
            "language": "ja",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text"] == "hello world"
    assert [s["speaker"] for s in body["segments"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert body["segments"][0]["id"] == 0
    # composite model + hints threaded through
    assert fake.last["model"].endswith("community-1")
    assert fake.last["language"] == "ja"
    assert fake.last["num_speakers"] == 2
