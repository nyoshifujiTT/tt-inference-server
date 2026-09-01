# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from domain.diarization_response import DiarizationResponse, DiarizationSegment
from utils.diarization_warnings import (
    build_speaker_count_warning,
    count_distinct_speakers,
)


def _segs(*speakers):
    return [
        {"speaker": sp, "start": float(i), "end": float(i) + 1.0}
        for i, sp in enumerate(speakers)
    ]


def test_count_distinct_speakers():
    assert count_distinct_speakers(_segs("A", "B", "A")) == 2
    assert count_distinct_speakers([]) == 0


def test_no_warning_when_unconstrained():
    assert build_speaker_count_warning(3) is None


def test_no_warning_when_num_speakers_honored():
    assert build_speaker_count_warning(2, num_speakers=2) is None


def test_warning_when_num_speakers_not_honored():
    w = build_speaker_count_warning(3, num_speakers=2)
    assert w is not None and "numSpeakers=2" in w and "3" in w


def test_warning_below_min_speakers():
    w = build_speaker_count_warning(1, min_speakers=2)
    assert w is not None and "minSpeakers=2" in w


def test_warning_above_max_speakers():
    w = build_speaker_count_warning(5, max_speakers=3)
    assert w is not None and "maxSpeakers=3" in w


def test_response_emits_warning_key_only_when_set():
    with_warn = DiarizationResponse(
        segments=[DiarizationSegment(speaker="A", start=0.0, end=1.0)],
        warning="requested numSpeakers=2 could not be honored; detected 1 speakers",
    )
    d = with_warn.to_dict()
    assert d["warning"].startswith("requested numSpeakers=2")

    without = DiarizationResponse(
        segments=[DiarizationSegment(speaker="A", start=0.0, end=1.0)]
    )
    assert "warning" not in without.to_dict()
