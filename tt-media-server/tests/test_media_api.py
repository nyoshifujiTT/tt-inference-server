# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""POST /v1/media/input hands back a pre-signed url on the storage service.

That is what the official API returns (https://docs.pyannote.ai/api-reference/
upload-media) and what its clients expect: the upload goes straight to storage,
never through the process scheduling device work. This server used to serve the
PUT itself, which put a receive path inside the official /v1/media namespace
that no pyannoteAI client would ever call.

Signing is a local computation -- boto3 never contacts the endpoint to produce
a pre-signed url -- so these tests exercise the real client against a real
signer and assert on the url it produces.
"""

import os

os.environ["NO_AUTH"] = "1"

from urllib.parse import parse_qs, urlsplit

import pytest
import utils.media_object_storage as mos
from config.settings import settings
from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_ai_api import media
from tests.diarization_auth import auth_headers

_ENDPOINT = "https://storage.example.com:9000"


@pytest.fixture()
def configured(monkeypatch):
    monkeypatch.setattr(settings, "media_storage_endpoint", _ENDPOINT, False)
    monkeypatch.setattr(settings, "media_storage_bucket", "media", False)
    monkeypatch.setattr(settings, "media_storage_access_key", "AKIATEST", False)
    monkeypatch.setattr(settings, "media_storage_secret_key", "secret", False)
    mos.reset_client()
    yield
    mos.reset_client()


@pytest.fixture()
def unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "media_storage_endpoint", "", False)
    monkeypatch.setattr(settings, "media_storage_bucket", "", False)
    mos.reset_client()
    yield
    mos.reset_client()


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(media.router, prefix="/v1/media")
    return TestClient(app, headers=auth_headers())


def test_the_put_url_points_at_the_storage_service_not_at_this_server(
    client, configured
):
    resp = client.post("/v1/media/input", json={"url": "media://sess/a.wav"})
    assert resp.status_code == 201, resp.text
    parts = urlsplit(resp.json()["url"])
    assert f"{parts.scheme}://{parts.netloc}" == _ENDPOINT
    # path-style addressing: a self-hosted endpoint is reached by host or IP,
    # where boto3's preferred virtual-host style would invent a hostname.
    assert parts.path == "/media/sess/a.wav"


def test_the_put_url_carries_a_sigv4_signature_and_an_expiry(client, configured):
    resp = client.post("/v1/media/input", json={"url": "media://sess/a.wav"})
    query = parse_qs(urlsplit(resp.json()["url"]).query)
    # SigV4, so the client needs nothing but the url -- plain `curl -T` works.
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert query["X-Amz-Signature"]
    assert query["X-Amz-Expires"] == [
        str(settings.media_storage_presign_expiry_seconds)
    ]
    assert query["X-Amz-Credential"][0].startswith("AKIATEST/")


def test_no_upload_route_is_served_here(client, configured):
    """The bytes must not come back through the inference API."""
    resp = client.put("/v1/media/input/sess/a.wav", content=b"RIFFDATA")
    assert resp.status_code == 404, resp.text
    # and not even on the declaration path itself
    assert client.put("/v1/media/input", content=b"RIFFDATA").status_code == 405


def test_declare_requires_url(client, configured):
    assert client.post("/v1/media/input", json={}).status_code == 400


@pytest.mark.parametrize(
    "bad",
    [
        "https://x/y.wav",  # not a media:// declaration
        "media://",  # empty key
        "media://../etc/passwd",  # traversal
        "media://a b",  # space is outside the official key charset
    ],
)
def test_declare_rejects_bad_keys(client, configured, bad):
    assert client.post("/v1/media/input", json={"url": bad}).status_code == 400


def test_without_storage_the_endpoint_says_so_and_names_the_alternatives(
    client, unconfigured
):
    """501, not 500: the server works as configured, this optional capability
    was simply not turned on -- and the caller needs to know what does work."""
    resp = client.post("/v1/media/input", json={"url": "media://sess/a.wav"})
    assert resp.status_code == 501, resp.text
    detail = resp.json()["detail"]
    assert "MEDIA_STORAGE_ENDPOINT" in detail
    assert "base64" in detail
    assert "http(s)" in detail
