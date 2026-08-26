# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import io
import unittest
import wave
from unittest.mock import MagicMock, patch

import pytest

from utils.media_clients.diarization_client import (
    DiarizationClientStrategy,
    _audio_duration_seconds,
    _load_reference_annotation,
    _turns_to_annotation,
)
from utils.media_clients.test_status import DiarizationTestStatus
from workflows.workflow_types import ReportCheckTypes


class MockAsyncResponse:
    """Mock async aiohttp response."""

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
    """Mock aiohttp session that replays the staging round trip in order."""

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


def _wav_bytes(seconds: float, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return buf.getvalue()


def _strategy() -> DiarizationClientStrategy:
    model_spec = MagicMock()
    model_spec.model_name = "speaker-diarization-community-1"
    model_spec.model_id = "id_diar"
    device = MagicMock()
    device.name = "P150"
    return DiarizationClientStrategy(
        [], model_spec, device, "/tmp/diar-bench-out", 8018
    )


class TestAudioDuration(unittest.TestCase):
    def test_duration_is_read_from_the_wav_header(self, tmp_path=None):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            handle.write(_wav_bytes(2.5))
            handle.flush()
            assert _audio_duration_seconds(handle.name) == pytest.approx(2.5)

    def test_unreadable_audio_reports_no_duration(self):
        assert _audio_duration_seconds("/nonexistent/never.wav") is None


class TestDiarizeOnce(unittest.TestCase):
    """The benchmark call must drive the pyannoteAI staging round trip."""

    def test_success_stages_then_diarizes_and_computes_rtr(self):
        import asyncio

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

        strategy = _strategy()
        with patch("aiohttp.ClientSession", return_value=session):
            with patch("builtins.open", unittest.mock.mock_open(read_data=b"RIFF")):
                status = asyncio.run(strategy._diarize_once("/tmp/a.wav", 30.0))

        assert status.status is True
        assert status.num_speakers == 2
        assert status.num_turns == 2
        assert status.rtr is not None and status.rtr > 0  # audio secs / wall secs
        methods = [method for method, _ in session.calls]
        assert methods == ["POST", "PUT", "POST"]
        assert session.calls[0][1].endswith("/v1/media/input")
        assert session.calls[2][1].endswith("/v1/audio/diarize")

    def test_failed_diarize_is_recorded_as_a_failed_sample(self):
        import asyncio

        session = MockAsyncSession(
            [
                MockAsyncResponse(201, {"url": "http://host:8018/v1/media/input/k"}),
                MockAsyncResponse(200, {}),
                MockAsyncResponse(500, {}),
            ]
        )

        strategy = _strategy()
        with patch("aiohttp.ClientSession", return_value=session):
            with patch("builtins.open", unittest.mock.mock_open(read_data=b"RIFF")):
                status = asyncio.run(strategy._diarize_once("/tmp/a.wav", 30.0))

        assert status.status is False


class TestReferenceAnnotation(unittest.TestCase):
    """The DER reference is the RTTM shipped beside pyannote's sample."""

    def test_rttm_is_parsed_into_segments_per_speaker(self):
        import tempfile

        rttm = (
            "SPEAKER sample 1 6.690 0.430 <NA> <NA> speaker90 <NA> <NA>\n"
            "SPEAKER sample 1 7.550 0.800 <NA> <NA> speaker91 <NA> <NA>\n"
            "\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".rttm") as handle:
            handle.write(rttm)
            handle.flush()
            annotation = _load_reference_annotation(handle.name)

        assert sorted(annotation.labels()) == ["speaker90", "speaker91"]
        # RTTM stores start + duration; the annotation must hold start + end.
        first = next(iter(annotation.itertracks(yield_label=True)))
        assert first[0].start == pytest.approx(6.690)
        assert first[0].end == pytest.approx(7.120)

    def test_served_turns_become_a_comparable_annotation(self):
        annotation = _turns_to_annotation(
            [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.5},
            ]
        )
        assert sorted(annotation.labels()) == ["SPEAKER_00", "SPEAKER_01"]


class TestAccuracyCheck(unittest.TestCase):
    """DER alone is not enough: a wrong speaker count must fail."""

    def test_low_der_with_matching_speaker_count_passes(self):
        assert (
            _strategy()._calculate_accuracy_check(0.01, True) == ReportCheckTypes.PASS
        )

    def test_high_der_fails(self):
        assert (
            _strategy()._calculate_accuracy_check(0.9, True) == ReportCheckTypes.FAIL
        )

    def test_wrong_speaker_count_fails_even_with_a_low_der(self):
        assert (
            _strategy()._calculate_accuracy_check(0.0, False) == ReportCheckTypes.FAIL
        )


class TestRunEval(unittest.TestCase):
    """run_eval must score a real DER against the reference annotation."""

    def test_eval_writes_a_der_scored_report(self):
        import json
        import pathlib
        import tempfile

        turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
        ]
        status = DiarizationTestStatus(
            True, 5.0, latency=4.9, rtr=6.0, num_speakers=2, num_turns=2, turns=turns
        )

        with tempfile.TemporaryDirectory() as out:
            strategy = _strategy()
            strategy.output_path = out
            strategy.model_spec.hf_model_repo = "pyannote/speaker-diarization-community-1"
            strategy.require_health = MagicMock(return_value="diarization-cpu")
            strategy.get_performance_targets = MagicMock(
                return_value=MagicMock(ttft_ms=None, rtr=None, tolerance=0.1)
            )

            async def _fake(*args, **kwargs):
                return status

            with patch(
                "utils.media_clients.diarization_client._sample_audio_path",
                return_value="/tmp/a.wav",
            ), patch(
                "utils.media_clients.diarization_client._audio_duration_seconds",
                return_value=30.0,
            ), patch(
                "utils.media_clients.diarization_client._load_reference_annotation",
                return_value=_turns_to_annotation(turns),
            ), patch(
                "utils.media_clients.diarization_client._sample_reference_path",
                return_value="/tmp/a.rttm",
            ), patch.object(
                DiarizationClientStrategy, "_diarize_once", side_effect=_fake
            ):
                strategy.run_eval()

            written = list(pathlib.Path(out).rglob("results_*.json"))
            assert len(written) == 1
            report = json.loads(written[0].read_text())[0]

        # Hypothesis equals the reference here, so the DER must be exactly 0.
        assert report["score"] == pytest.approx(0.0)
        assert report["task_name"] == "pyannote_sample_der"
        assert report["speaker_count_matches"] is True
        assert report["accuracy_check"] == ReportCheckTypes.PASS

    def test_a_failed_request_is_not_scored(self):
        strategy = _strategy()
        strategy.require_health = MagicMock(return_value="diarization-cpu")

        async def _fake(*args, **kwargs):
            return DiarizationTestStatus(False, 0.0)

        with patch(
            "utils.media_clients.diarization_client._sample_audio_path",
            return_value="/tmp/a.wav",
        ), patch(
            "utils.media_clients.diarization_client._audio_duration_seconds",
            return_value=30.0,
        ), patch(
            "utils.media_clients.diarization_client._load_reference_annotation",
            return_value=_turns_to_annotation([]),
        ), patch(
            "utils.media_clients.diarization_client._sample_reference_path",
            return_value="/tmp/a.rttm",
        ), patch.object(
            DiarizationClientStrategy, "_diarize_once", side_effect=_fake
        ):
            with pytest.raises(RuntimeError):
                strategy.run_eval()


class TestReport(unittest.TestCase):
    def test_report_records_rtr_and_the_diarization_task_type(self):
        import json
        import tempfile

        status_list = [
            DiarizationTestStatus(True, 5.0, latency=4.9, rtr=6.0, num_speakers=2),
            DiarizationTestStatus(True, 5.0, latency=5.1, rtr=5.0, num_speakers=2),
        ]
        with tempfile.TemporaryDirectory() as out:
            strategy = _strategy()
            strategy.output_path = out
            strategy.get_performance_targets = MagicMock(
                return_value=MagicMock(ttft_ms=None, rtr=None, tolerance=0.1)
            )
            strategy._generate_report(status_list, wall_clock_seconds=10.0)

            import pathlib

            written = list(pathlib.Path(out).glob("benchmark_*.json"))
            assert len(written) == 1
            report = json.loads(written[0].read_text())

        assert report["task_type"] == "diarization"
        assert report["benchmarks"]["num_requests"] == 2
        assert report["benchmarks"]["rtr"] == pytest.approx(5.5)
        assert report["benchmarks"]["latency"] == pytest.approx(5.0)
