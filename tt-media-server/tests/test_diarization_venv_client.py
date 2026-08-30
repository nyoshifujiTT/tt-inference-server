# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The audio-venv worker is how diarization reaches pyannote in the image."""

import json

import numpy as np
import pytest

from utils.diarization_venv_client import DiarizationVenvClient


class _FakeProc:
    """Stand-in for the worker: replays a ready line then canned responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.written = []
        self.stdin = self
        self.stdout = self
        self._lines = [json.dumps({"status": "ready"}) + "\n"]

    def write(self, data):
        self.written.append(data)
        request = json.loads(data)
        response = self._responses.pop(0)
        response.setdefault("id", request["id"])
        self._lines.append(json.dumps(response) + "\n")

    def flush(self):
        pass

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def poll(self):
        return None

    def close(self):
        pass

    def wait(self, timeout=None):
        return 0


def _client(responses, tmp_path, logger, **kwargs):
    proc = _FakeProc(responses)
    script = tmp_path / "diarize_community1.py"
    script.write_text("")
    return (
        DiarizationVenvClient(
            model_path="pyannote/speaker-diarization-community-1",
            logger=logger,
            python_executable=str(tmp_path),
            script_path=script,
            popen_factory=lambda *a, **k: proc,
            **kwargs,
        ),
        proc,
    )


class _Logger:
    def info(self, *_):
        pass

    def warning(self, *_):
        pass


def test_a_result_crosses_the_process_boundary_unchanged(tmp_path):
    expected = {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]}
    client, _ = _client(
        [{"status": "success", "result": expected}], tmp_path, _Logger()
    )
    out = client.diarize({"waveform": np.zeros(16000), "sample_rate": 16000})
    assert out == expected


def test_speaker_hints_are_forwarded_to_the_worker(tmp_path):
    client, proc = _client(
        [{"status": "success", "result": {"segments": []}}], tmp_path, _Logger()
    )
    client.diarize(
        {"waveform": np.zeros(16000), "sample_rate": 16000},
        num_speakers=3,
        exclusive=False,
    )
    sent = json.loads(proc.written[0])
    assert sent["num_speakers"] == 3
    assert sent["exclusive"] is False
    assert sent["sample_rate"] == 16000


def test_a_worker_error_is_raised_rather_than_returned_as_a_result(tmp_path):
    client, _ = _client(
        [{"status": "error", "error": "pipeline failed to build"}], tmp_path, _Logger()
    )
    with pytest.raises(RuntimeError, match="pipeline failed to build"):
        client.diarize({"waveform": np.zeros(16000), "sample_rate": 16000})


def test_the_client_is_unavailable_when_the_audio_venv_is_not_there(tmp_path):
    """Availability is what makes the service fall back to the direct backend."""
    client = DiarizationVenvClient(
        model_path="x",
        logger=_Logger(),
        python_executable=str(tmp_path / "no-such-python"),
        script_path=tmp_path / "no-such-script.py",
    )
    assert client.is_available() is False
