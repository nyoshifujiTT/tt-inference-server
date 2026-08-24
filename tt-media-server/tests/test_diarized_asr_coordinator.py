# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import numpy as np
from utils.diarized_asr_coordinator import DiarizedAsrCoordinator, slice_waveform


def test_slice_waveform_by_sample_index():
    wf = np.arange(16000, dtype=np.float32)  # 1 second @16k
    s = slice_waveform(wf, 16000, 0.25, 0.5)
    assert len(s) == 4000
    assert s[0] == 4000
    # out-of-range / empty
    assert len(slice_waveform(wf, 16000, 0.9, 0.9)) == 0
    assert len(slice_waveform(wf, 16000, 0.5, 10.0)) == 8000  # clamped to end


def test_coordinator_runs_diarize_then_asr_and_builds_diarized_json():
    wf = np.zeros(16000 * 3, dtype=np.float32)  # 3 s

    def fake_diarize(
        audio, num_speakers=None, min_speakers=None, max_speakers=None, exclusive=True
    ):
        assert exclusive is True
        return {
            "segments": [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}],
            "exclusiveDiarization": [
                {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_01"},
            ],
        }

    seen = []

    def fake_asr(chunk, sr):
        seen.append((len(chunk), sr))
        return f"len{len(chunk)}"

    coord = DiarizedAsrCoordinator(fake_diarize, fake_asr, sample_rate=16000)
    out = coord.run("audio.wav", wf, num_speakers=2)

    # used exclusive turns (2), sliced correctly, ASR per turn
    assert seen == [(16000, 16000), (32000, 16000)]
    assert out["task"] == "transcribe"
    assert out["duration"] == 3.0
    assert [s["speaker"] for s in out["segments"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert [s["id"] for s in out["segments"]] == [0, 1]
    assert out["segments"][0]["text"] == "len16000"
    assert out["text"] == "len16000 len32000"


def test_coordinator_falls_back_to_segments_without_exclusive():
    wf = np.zeros(16000, dtype=np.float32)

    def fake_diarize(audio, **kw):
        return {"segments": [{"start": 0.0, "end": 1.0, "speaker": "A"}]}

    coord = DiarizedAsrCoordinator(fake_diarize, lambda c, sr: "t")
    out = coord.run("a.wav", wf)
    assert len(out["segments"]) == 1
    assert out["segments"][0]["speaker"] == "A"
