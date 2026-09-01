# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for the speaker-diarization benchmark runner."""

from __future__ import annotations

import asyncio
import base64
import io
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from test_module.benchmark_tests import diarization_benchmark_tests as mod
from test_module.test_status import DiarizationTestStatus


class MockAsyncResponse:
    def __init__(self, status=200, json_data=None):
        self.status = status
        self._json_data = json_data or {}

    async def json(self):
        return self._json_data

    async def text(self):
        return "Error text"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockAsyncSession:
    """Replays the job round trip in order and records the calls made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def _next(self, method, url, kwargs=None):
        self.calls.append((method, url, kwargs or {}))
        return self._responses.pop(0)

    def post(self, url, *args, **kwargs):
        return self._next("POST", url, kwargs)

    def get(self, url, *args, **kwargs):
        return self._next("GET", url, kwargs)

    def put(self, url, *args, **kwargs):
        return self._next("PUT", url, kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _reads(data: bytes):
    """``open()`` stand-in whose read() returns ``data``."""
    handle = MagicMock()
    handle.__enter__.return_value.read.return_value = data
    return MagicMock(return_value=handle)


def _ctx():
    return SimpleNamespace(
        model_spec=SimpleNamespace(model_name="speaker-diarization-community-1"),
        device=SimpleNamespace(name="p150"),
        base_url="http://127.0.0.1:8018",
    )


def _wav_bytes(seconds: float, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return buf.getvalue()


def test_audio_duration_is_read_from_the_wav_header(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(_wav_bytes(2.5))
    assert mod.audio_duration_seconds(str(path)) == pytest.approx(2.5)


def test_unreadable_audio_reports_no_duration():
    assert mod.audio_duration_seconds("/nonexistent/never.wav") is None


_TURNS = [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
    {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
]


def _succeeded(output=None):
    return MockAsyncResponse(
        200, {"status": "succeeded", "output": output or {"diarization": _TURNS}}
    )


def test_diarize_once_creates_a_job_polls_it_and_computes_rtr():
    """The call must drive the official job round trip, in order."""
    session = MockAsyncSession(
        [
            MockAsyncResponse(201, {"jobId": "job-1", "status": "created"}),
            MockAsyncResponse(200, {"status": "running"}),
            _succeeded(),
        ]
    )

    with patch("aiohttp.ClientSession", return_value=session):
        with patch("builtins.open", _reads(b"RIFFDATA")):
            status = asyncio.run(mod.diarize_once(_ctx(), "/tmp/a.wav", 30.0))

    assert status.status is True
    assert status.num_speakers == 2
    assert status.num_turns == 2
    assert status.rtr is not None and status.rtr > 0  # audio secs / wall secs
    assert [method for method, _url, _kw in session.calls] == ["POST", "GET", "GET"]
    assert session.calls[0][1].endswith("/v1/diarize")
    assert session.calls[1][1].endswith("/v1/jobs/job-1")


def test_the_audio_travels_in_the_request_rather_than_through_a_staged_object():
    """Staging would need an object store beside the server; a benchmark that
    cannot run without one measures the deployment, not the model."""
    session = MockAsyncSession(
        [MockAsyncResponse(201, {"jobId": "job-1"}), _succeeded()]
    )

    with patch("aiohttp.ClientSession", return_value=session):
        with patch("builtins.open", _reads(b"RIFFDATA")):
            asyncio.run(mod.diarize_once(_ctx(), "/tmp/a.wav", 30.0))

    urls = [url for _m, url, _kw in session.calls]
    assert not any("/v1/media/input" in url for url in urls)
    assert session.calls[0][2]["json"]["url"] == base64.b64encode(b"RIFFDATA").decode()


def test_a_failed_job_creation_is_recorded_as_a_failed_sample():
    session = MockAsyncSession([MockAsyncResponse(500, {})])

    with patch("aiohttp.ClientSession", return_value=session):
        with patch("builtins.open", _reads(b"RIFFDATA")):
            status = asyncio.run(mod.diarize_once(_ctx(), "/tmp/a.wav", 30.0))

    assert status.status is False


def test_a_job_that_ends_as_failed_is_recorded_as_a_failed_sample():
    """A job the server reports as failed must not be scored as a result."""
    session = MockAsyncSession(
        [
            MockAsyncResponse(201, {"jobId": "job-1"}),
            MockAsyncResponse(200, {"status": "failed", "error": "boom"}),
        ]
    )

    with patch("aiohttp.ClientSession", return_value=session):
        with patch("builtins.open", _reads(b"RIFFDATA")):
            status = asyncio.run(mod.diarize_once(_ctx(), "/tmp/a.wav", 30.0))

    assert status.status is False


def test_a_job_that_never_finishes_fails_the_sample_instead_of_hanging():
    session = MockAsyncSession(
        [MockAsyncResponse(201, {"jobId": "job-1"})]
        + [MockAsyncResponse(200, {"status": "running"}) for _ in range(50)]
    )

    with patch("aiohttp.ClientSession", return_value=session):
        with patch("builtins.open", _reads(b"RIFFDATA")):
            with patch.object(mod, "REQUEST_TIMEOUT_S", 0.05):
                with patch.object(mod, "JOB_POLL_INTERVAL_S", 0.01):
                    status = asyncio.run(mod.diarize_once(_ctx(), "/tmp/a.wav", 30.0))

    assert status.status is False


def test_benchmark_block_reports_rtr_under_the_diarization_task_type():
    status = DiarizationTestStatus(
        True, 5.0, ttft_ms=4900.0, rtr=6.0, num_speakers=2, num_turns=2
    )

    async def _fake(*args, **kwargs):
        return status

    with patch.object(
        mod, "require_health", return_value="diarization-cpu"
    ), patch.object(mod, "sample_audio_path", return_value="/tmp/a.wav"), patch.object(
        mod, "audio_duration_seconds", return_value=30.0
    ), patch.object(mod, "diarize_once", side_effect=_fake), patch.object(
        mod, "run_tiered_check", return_value=({}, "PASS")
    ):
        block = mod.run_diarization_benchmark(_ctx())

    assert block.kind == "benchmarks"
    assert block.task_type == "diarization"
    benchmarks = block.data["Benchmarks"]
    assert benchmarks["num_requests"] == mod.DEFAULT_NUM_CALLS
    assert benchmarks["rtr"] == pytest.approx(6.0)
