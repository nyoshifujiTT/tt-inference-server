# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""An http(s) diarize ``url`` is fetched through the server's media downloader.

The endpoint used to reach the network with a private ``urllib.request`` helper
that had no hostname allowlist, no redirect re-validation and no shared
deadline -- a second, unhardened way out of the process, next to the one
``utils/media_downloader.py`` exists to be. These tests pin the routing (the
downloader is called, with the caller's URL) and the status taxonomy
``open_ai_api/video.py`` already uses for the same errors.
"""

import os

os.environ["NO_AUTH"] = "1"

from unittest.mock import AsyncMock, patch

import pytest
import utils.media_downloader as media_downloader
from domain.diarization_response import DiarizationResponse, DiarizationSegment
from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_ai_api import diarization
from resolver.service_resolver import service_resolver
from tests.diarization_auth import auth_headers

_URL = "https://storage.example.com/bucket/audio.wav?X-Amz-Signature=deadbeef"


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


def test_an_http_url_is_downloaded_through_the_shared_downloader(client, fake):
    download = AsyncMock(return_value=b"RIFFxxxxWAVE")
    with patch.object(media_downloader, "download_media_url", download):
        resp = client.post("/v1/diarize", json={"url": _URL})
    assert resp.status_code == 201, resp.text
    download.assert_awaited_once_with(_URL)
    assert bytes(fake.last.file) == b"RIFFxxxxWAVE"


@pytest.mark.parametrize(
    "error,status",
    [
        (media_downloader.MediaDownloadPolicyError("not allowed"), 400),
        (media_downloader.MediaDownloadTooLargeError("too big"), 413),
        (media_downloader.MediaDownloadFetchError("origin said 403"), 422),
    ],
)
def test_download_failures_map_to_the_same_statuses_as_the_video_path(
    client, fake, error, status
):
    download = AsyncMock(side_effect=error)
    with patch.object(media_downloader, "download_media_url", download):
        resp = client.post("/v1/diarize", json={"url": _URL})
    assert resp.status_code == status, resp.text
    assert fake.last is None


def test_a_misconfigured_server_is_not_reported_as_a_client_error(client, fake):
    """``MediaDownloadError`` itself means operator error, so it must not be
    caught into a 4xx that blames the caller -- the downloader's own docstring
    makes that distinction and the video path honours it."""
    download = AsyncMock(side_effect=media_downloader.MediaDownloadError("bad config"))
    with patch.object(media_downloader, "download_media_url", download):
        with pytest.raises(media_downloader.MediaDownloadError):
            client.post("/v1/diarize", json={"url": _URL})
    assert fake.last is None


@pytest.mark.parametrize("url", ["ftp://h/a.wav", "file:///etc/passwd", "/tmp/a.wav"])
def test_non_http_non_media_urls_are_refused_before_any_fetch(client, fake, url):
    download = AsyncMock()
    with patch.object(media_downloader, "download_media_url", download):
        resp = client.post("/v1/diarize", json={"url": url})
    assert resp.status_code == 400, resp.text
    download.assert_not_awaited()
    assert fake.last is None
