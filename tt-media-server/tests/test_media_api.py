# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os

os.environ["NO_AUTH"] = "1"

import importlib

import pytest
from tests.diarization_auth import auth_headers
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_INPUT_DIR", str(tmp_path))
    # reset the storage singleton so it picks up the tmp dir
    import utils.media_storage as ms

    ms._STORAGE = None
    from open_ai_api import media

    importlib.reload(media)
    app = FastAPI()
    app.include_router(media.router, prefix="/v1/media")
    yield TestClient(app)
    ms._STORAGE = None


def test_declare_then_put_roundtrip(client):
    # 1. declare media://key -> returns a PUT url
    r = client.post(
        "/v1/media/input", json={"url": "media://sess/a.wav"}, headers=auth_headers()
    )
    assert r.status_code == 201, r.text
    put_url = r.json()["url"]
    assert put_url.endswith("/v1/media/input/sess/a.wav")

    # 2. PUT bytes to that url (use just the path for the test client)
    path = "/v1/media/input/sess/a.wav"
    r2 = client.put(path, content=b"RIFFDATA", headers=auth_headers())
    assert r2.status_code == 200, r2.text
    assert r2.json()["url"] == "media://sess/a.wav"

    # 3. the staged bytes come back for the media:// url the diarize body uses
    from utils.media_storage import get_media_storage

    assert get_media_storage().get("media://sess/a.wav") == b"RIFFDATA"


def test_declare_requires_url(client):
    r = client.post("/v1/media/input", json={}, headers=auth_headers())
    assert r.status_code == 400


def test_declare_rejects_bad_key(client):
    r = client.post(
        "/v1/media/input", json={"url": "media://../etc/passwd"}, headers=auth_headers()
    )
    assert r.status_code == 400


def test_put_empty_body_rejected(client):
    client.post(
        "/v1/media/input", json={"url": "media://k.wav"}, headers=auth_headers()
    )
    r = client.put("/v1/media/input/k.wav", content=b"", headers=auth_headers())
    assert r.status_code == 400
