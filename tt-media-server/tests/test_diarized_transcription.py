# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from utils.diarized_transcription import build_diarized_json


def test_build_diarized_json_basic_shape_and_order():
    turns = [
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
    ]
    texts = {(0.0, 1.0): "hello", (1.0, 2.0): "world"}
    out = build_diarized_json(turns, lambda s, e: texts[(s, e)])
    assert out["task"] == "transcribe"
    assert out["duration"] == 2.0
    assert out["text"] == "hello world"  # ordered by start
    assert out["segments"] == [
        {"id": 0, "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "hello"},
        {"id": 1, "speaker": "SPEAKER_01", "start": 1.0, "end": 2.0, "text": "world"},
    ]


def test_slice_bounds_passed_to_asr():
    seen = []

    def asr(s, e):
        seen.append((s, e))
        return "x"

    build_diarized_json([{"start": 0.5, "end": 3.25, "speaker": "A"}], asr)
    assert seen == [(0.5, 3.25)]


def test_empty_transcripts_are_dropped_and_ids_reindexed():
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 1.0, "end": 2.0, "speaker": "B"},
        {"start": 2.0, "end": 3.0, "speaker": "A"},
    ]

    def asr(s, e):
        return "" if s == 1.0 else "t"

    out = build_diarized_json(turns, asr)
    assert [seg["id"] for seg in out["segments"]] == [0, 1]
    assert [seg["speaker"] for seg in out["segments"]] == ["A", "A"]


def test_keep_empty_when_drop_empty_false():
    out = build_diarized_json(
        [{"start": 0.0, "end": 1.0, "speaker": "A"}],
        lambda s, e: "",
        drop_empty=False,
    )
    assert out["segments"][0]["text"] == ""
    assert out["text"] == ""


def test_explicit_duration_used():
    out = build_diarized_json(
        [{"start": 0.0, "end": 1.0, "speaker": "A"}],
        lambda s, e: "hi",
        duration=42.0,
    )
    assert out["duration"] == 42.0


def test_no_turns_gives_empty_result():
    out = build_diarized_json([], lambda s, e: "x")
    assert out["segments"] == []
    assert out["text"] == ""
    assert out["duration"] == 0.0
