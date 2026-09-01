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


def test_base64_is_capped(client, fake, monkeypatch):
    monkeypatch.setattr(settings, "media_inline_max_bytes", 16, raising=False)
    oversize = base64.b64encode(b"x" * 64).decode("ascii")
    resp = client.post("/v1/diarize", json={"url": oversize})
    assert resp.status_code == 413, resp.text
    assert fake.last is None


def test_the_oversize_error_does_not_promise_a_url_would_get_through(
    client, fake, monkeypatch
):
    """``media_url_max_bytes`` caps the audio, not the transport.

    This message used to read "upload it and pass a url instead", which is
    advice that cannot work: the downloader enforces the same cap, so the
    caller stages a file, retries, and gets 413 a second time from a different
    code path. Point at the two things that do change the outcome.
    """
    monkeypatch.setattr(settings, "media_inline_max_bytes", 16, raising=False)
    detail = client.post(
        "/v1/diarize", json={"url": base64.b64encode(b"x" * 64).decode("ascii")}
    ).json()["detail"]
    assert "does not raise it" in detail
    assert "inline audio limit" in detail


def test_the_inline_cap_is_what_admits_or_refuses_a_base64_body(
    client, fake, monkeypatch
):
    """Moving media_inline_max_bytes moves the inline limit, both ways.

    Sizes straddle a cap of 32 so the same payload is refused before and
    admitted after, without either assertion depending on the default value.
    """
    payload = b"x" * 48
    encoded = base64.b64encode(payload).decode("ascii")

    monkeypatch.setattr(settings, "media_inline_max_bytes", 32, raising=False)
    assert client.post("/v1/diarize", json={"url": encoded}).status_code == 413

    # the endpoint and the downloader share one settings object, so neither
    # route can quietly acquire a limit the other cannot see
    from utils import media_downloader

    assert media_downloader.settings is settings

    monkeypatch.setattr(settings, "media_inline_max_bytes", 4096, raising=False)
    assert client.post("/v1/diarize", json={"url": encoded}).status_code == 201
    assert bytes(fake.last.file) == payload


def test_the_cap_is_checked_before_the_payload_is_decoded(client, fake, monkeypatch):
    """The pre-decode length check has to actually fire, otherwise an oversize
    body is materialised in memory first and the cap protects nothing."""
    monkeypatch.setattr(settings, "media_inline_max_bytes", 16, raising=False)
    with patch.object(
        base64, "b64decode", side_effect=AssertionError("decoded an oversize payload")
    ):
        resp = client.post(
            "/v1/diarize", json={"url": base64.b64encode(b"y" * 4096).decode("ascii")}
        )
    assert resp.status_code == 413, resp.text


def test_max_audio_size_bytes_does_not_apply_to_diarization(client, fake, monkeypatch):
    """The 50 MiB audio limit is the transcription service's, not ours.

    ``settings.max_audio_size_bytes`` is enforced by ``AudioManager``, which is
    reached only through ``model_services/audio_service.py``.
    ``DiarizationService.pre_process`` decodes with ``decode_to_wav`` directly
    and never enters AudioManager, so that setting has no effect here.

    Worth pinning because the two names sit next to each other in settings and
    both look like "how much audio is allowed". Someone tightening
    max_audio_size_bytes to bound diarization would change nothing and believe
    otherwise; this test says so out loud. Squeezing it to a single byte must
    not affect a request that the diarization cap admits.
    """
    monkeypatch.setattr(settings, "max_audio_size_bytes", 1, raising=False)
    monkeypatch.setattr(settings, "media_url_max_bytes", 4096, raising=False)

    payload = b"x" * 1024
    resp = client.post(
        "/v1/diarize", json={"url": base64.b64encode(payload).decode("ascii")}
    )
    assert resp.status_code == 201, resp.text
    assert bytes(fake.last.file) == payload


def test_inline_audio_has_its_own_setting_separate_from_the_download_cap(
    client, fake, monkeypatch
):
    """Nothing is downloaded for a base64 body, so the download cap should not
    be what governs it.

    ``media_url_max_bytes`` arrived with the presigned-URL work (#4983) and is
    read by ``download_media_url`` alone; upstream bounds an *inline* image
    with ``MAX_BASE64_IMAGE_LEN`` on the pydantic field instead. Making a
    base64 body answer to the download setting meant an operator reading its
    name or docstring -- both about URL fetches -- would not have guessed it
    governed inline audio too.
    """
    monkeypatch.setattr(settings, "media_url_max_bytes", 4096, raising=False)
    monkeypatch.setattr(settings, "media_inline_max_bytes", 32, raising=False)

    payload = b"x" * 64  # inside the download cap, past the inline one
    resp = client.post(
        "/v1/diarize", json={"url": base64.b64encode(payload).decode("ascii")}
    )
    assert resp.status_code == 413, resp.text
    assert fake.last is None


def test_zero_makes_inline_follow_the_download_cap(client, fake, monkeypatch):
    """0 is the opt-out: one ceiling for every route, for a deployment that
    would rather tune a single number."""
    monkeypatch.setattr(settings, "media_inline_max_bytes", 0, raising=False)

    monkeypatch.setattr(settings, "media_url_max_bytes", 32, raising=False)
    over = base64.b64encode(b"x" * 64).decode("ascii")
    assert client.post("/v1/diarize", json={"url": over}).status_code == 413

    monkeypatch.setattr(settings, "media_url_max_bytes", 4096, raising=False)
    assert client.post("/v1/diarize", json={"url": over}).status_code == 201


def test_inline_is_capped_lower_than_fetched_audio_by_default():
    """The default is deliberately not the download cap.

    An inline body is read whole by the ASGI layer, parsed into a JSON string
    and only then decoded, so the encoded form -- already 1.333x the audio --
    exists several times over before the audio does. A fetched object is
    streamed into one bytearray. Measured on a p150 with a 60 MiB recording:
    +175 MiB RSS inline against +119 MiB by url. Matching the two numbers
    would price the cheaper path as if it cost the same.
    """
    # Read the declarations out of the source. Importing config.settings here
    # would hand back whatever Mock another test module left in sys.modules,
    # and a Mock compares however you like -- the assertion would pass while
    # measuring nothing.
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "config" / "settings.py"
    ).read_text()
    declared = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                # compile/eval on the isolated expression: defaults are
                # written as arithmetic (16 * 1024 * 1024), which
                # literal_eval refuses.
                declared[node.target.id] = eval(  # noqa: S307 - our own source
                    compile(ast.Expression(node.value), "<settings>", "eval"),
                    {"__builtins__": {}},
                    {},
                )
            except Exception:
                continue

    inline = declared["media_inline_max_bytes"]
    assert inline, "0 would silently restore the download cap as the default"

    # Against the cap this model actually runs with: the class default
    # (7.5 MB) is sized for one video input image, and the diarization entry
    # raises it to fit a recording.
    from config.constants import ModelConfigs, ModelRunners

    entry = next(
        cfg
        for (runner, _device), cfg in ModelConfigs.items()
        if runner is ModelRunners.TT_PYANNOTE_DIARIZATION
    )
    assert inline < entry["media_url_max_bytes"]
