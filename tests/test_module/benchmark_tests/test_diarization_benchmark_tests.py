# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for the speaker-diarization benchmark runner."""

from __future__ import annotations

import asyncio
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
    """Replays the staging round trip in order and records the calls made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def _next(self, method, url):
        self.calls.append((method, url))
        return self._responses.pop(0)

    def post(self, url, *args, **kwargs):
        return self._next("POST", url)

    def put(self, url, *args, **kwargs):
        return self._next("PUT", url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


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


def test_diarize_once_stages_then_diarizes_and_computes_rtr():
    """The call must drive the pyannoteAI staging round trip, in order."""
    session = MockAsyncSession(
        [
            MockAsyncResponse(201, {"url": "http://host:8018/v1/media/input/k"}),
            MockAsyncResponse(200, {}),
            MockAsyncResponse(
                200,
                {
                    "diarization": [
                        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
                        {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
                    ]
                },
            ),
        ]
    )

    with patch("aiohttp.ClientSession", return_value=session):
        with patch("builtins.open", MagicMock()):
            status = asyncio.run(mod.diarize_once(_ctx(), "/tmp/a.wav", 30.0))

    assert status.status is True
    assert status.num_speakers == 2
    assert status.num_turns == 2
    assert status.rtr is not None and status.rtr > 0  # audio secs / wall secs
    assert [method for method, _ in session.calls] == ["POST", "PUT", "POST"]
    assert session.calls[0][1].endswith("/v1/media/input")
    assert session.calls[2][1].endswith("/v1/audio/diarize")


def test_a_failed_diarize_is_recorded_as_a_failed_sample():
    session = MockAsyncSession(
        [
            MockAsyncResponse(201, {"url": "http://host:8018/v1/media/input/k"}),
            MockAsyncResponse(200, {}),
            MockAsyncResponse(500, {}),
        ]
    )

    with patch("aiohttp.ClientSession", return_value=session):
        with patch("builtins.open", MagicMock()):
            status = asyncio.run(mod.diarize_once(_ctx(), "/tmp/a.wav", 30.0))

    assert status.status is False


def test_benchmark_block_reports_rtr_under_the_diarization_task_type():
    status = DiarizationTestStatus(
        True, 5.0, ttft_ms=4900.0, rtr=6.0, num_speakers=2, num_turns=2
    )

    async def _fake(*args, **kwargs):
        return status

    with patch.object(mod, "require_health", return_value="diarization-cpu"), patch.object(
        mod, "sample_audio_path", return_value="/tmp/a.wav"
    ), patch.object(mod, "audio_duration_seconds", return_value=30.0), patch.object(
        mod, "diarize_once", side_effect=_fake
    ), patch.object(
        mod, "run_tiered_check", return_value=({}, "PASS")
    ):
        block = mod.run_diarization_benchmark(_ctx())

    assert block.kind == "benchmarks"
    assert block.task_type == "diarization"
    benchmarks = block.data["Benchmarks"]
    assert benchmarks["num_requests"] == mod.DEFAULT_NUM_CALLS
    assert benchmarks["rtr"] == pytest.approx(6.0)
