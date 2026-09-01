# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""``DiarizeRequest.url`` also carries inline base64 audio.

A deployment with no object storage reachable from the server had no way to
send audio at all: the official request has one audio field and it is a url.
The official schema types that field as a bare ``{"type": "string"}`` with no
pattern, so putting the bytes in it is not a schema violation, and it is the
same "url or base64 in one field" shape ``domain/video_i2v_generate_request``
already uses for images. These tests pin that a base64 body diarizes, that it
is never confused with a url, and that it obeys the same byte cap as a
download -- how the bytes arrive must not change how many the server holds.
"""

import base64
import os

os.environ["NO_AUTH"] = "1"

from unittest.mock import AsyncMock, patch

import pytest
import utils.media_downloader as media_downloader
from config.settings import settings
from domain.diarization_response import DiarizationResponse, DiarizationSegment
from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_ai_api import diarization
from resolver.service_resolver import service_resolver
from tests.diarization_auth import auth_headers

_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt "
_WAV_B64 = base64.b64encode(_WAV).decode("ascii")


class _FakeService:
    def __init__(self):
        self.last = None

    async def process_request(self, request):
        self.last = request
        return DiarizationResponse(
            segments=[DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0)],
            exclusiveDiarization=[],
        )


@pytest.fixture()
def fake():
    return _FakeService()


@pytest.fixture()
def client(fake):
    app = FastAPI()
    app.include_router(diarization.async_router, prefix="/v1")
    app.dependency_overrides[service_resolver] = lambda: fake
    return TestClient(app, headers=auth_headers())


def test_base64_audio_reaches_the_service_verbatim(client, fake):
    resp = client.post("/v1/diarize", json={"url": _WAV_B64})
    assert resp.status_code == 201, resp.text
    assert bytes(fake.last.file) == _WAV


def test_base64_audio_is_never_sent_to_the_downloader(client, fake):
    """Base64 must not be mistaken for something to fetch: the downloader is
    the network, and a payload is not a destination."""
    download = AsyncMock()
    with patch.object(media_downloader, "download_media_url", download):
        resp = client.post("/v1/diarize", json={"url": _WAV_B64})
    assert resp.status_code == 201, resp.text
    download.assert_not_awaited()


def test_a_string_that_is_neither_a_url_nor_base64_is_rejected(client, fake):
    resp = client.post("/v1/diarize", json={"url": "not base64!!"})
    assert resp.status_code == 400, resp.text
    assert "base64" in resp.json()["detail"]
    assert fake.last is None


@pytest.mark.parametrize(
    "body",
    [
        {},  # url missing entirely
        {"url": "not base64!!"},  # unparseable
        {"url": "ftp://h/a.wav"},  # a scheme we do not serve
    ],
)
def test_the_errors_that_offer_base64_say_it_is_non_standard(client, body):
    """Whoever reads these is being told base64 is an option, and must be told
    in the same breath that pyannoteAI's own service will not take it.

    Otherwise the error is an invitation to write a client that works here and
    fails the moment it is pointed at the cloud API -- the exact portability
    this endpoint exists to preserve."""
    detail = client.post("/v1/diarize", json=body).json()["detail"]
    assert "base64" in detail
    assert "non-standard" in detail


def test_an_unserved_scheme_is_named_rather_than_read_as_base64(client, fake):
    """``ftp://...`` is a url the server does not serve. Reporting it as
    malformed base64 would send the caller looking in the wrong place."""
    resp = client.post("/v1/diarize", json={"url": "ftp://h/a.wav"})
    assert resp.status_code == 400, resp.text
    assert "scheme" in resp.json()["detail"]
    assert fake.last is None


def test_empty_base64_is_rejected(client, fake):
    resp = client.post("/v1/diarize", json={"url": base64.b64encode(b"").decode()})
    # an empty string is a missing url, not an empty payload
    assert resp.status_code == 400, resp.text
    assert fake.last is None


def test_base64_obeys_the_same_byte_cap_as_a_download(client, fake, monkeypatch):
    monkeypatch.setattr(settings, "media_url_max_bytes", 16, raising=False)
    oversize = base64.b64encode(b"x" * 64).decode("ascii")
    resp = client.post("/v1/diarize", json={"url": oversize})
    assert resp.status_code == 413, resp.text
    assert fake.last is None


def test_the_cap_is_checked_before_the_payload_is_decoded(client, fake, monkeypatch):
    """The pre-decode length check has to actually fire, otherwise an oversize
    body is materialised in memory first and the cap protects nothing."""
    monkeypatch.setattr(settings, "media_url_max_bytes", 16, raising=False)
    with patch.object(
        base64, "b64decode", side_effect=AssertionError("decoded an oversize payload")
    ):
        resp = client.post(
            "/v1/diarize", json={"url": base64.b64encode(b"y" * 4096).decode("ascii")}
        )
    assert resp.status_code == 413, resp.text
