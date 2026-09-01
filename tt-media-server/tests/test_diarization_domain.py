# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import pytest
from domain.diarization_request import DiarizationRequest
from domain.diarization_response import DiarizationResponse, DiarizationSegment


def test_request_defaults_and_exclusive_true_by_default():
    req = DiarizationRequest(file=b"AUDIOBYTES")
    assert req.num_speakers is None
    assert req.min_speakers is None
    assert req.max_speakers is None
    assert req.exclusive is True


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("1", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("", True),
    ],
)
def test_exclusive_coercion(value, expected):
    assert DiarizationRequest(file=b"x", exclusive=value).exclusive is expected


def test_speaker_count_coercion_and_validation():
    req = DiarizationRequest(
        file=b"x", num_speakers="3", min_speakers="", max_speakers=None
    )
    assert req.num_speakers == 3
    assert req.min_speakers is None
    assert req.max_speakers is None
    with pytest.raises(ValueError):
        DiarizationRequest(file=b"x", num_speakers=0)


def test_response_to_dict_pyannoteai_shape():
    resp = DiarizationResponse(
        segments=[DiarizationSegment(start=0.2, end=1.6, speaker="SPEAKER_00")],
        exclusiveDiarization=[
            DiarizationSegment(start=0.2, end=1.6, speaker="SPEAKER_00")
        ],
    )
    d = resp.to_dict()
    assert d["diarization"][0] == {"speaker": "SPEAKER_00", "start": 0.2, "end": 1.6}
    assert d["exclusiveDiarization"][0]["speaker"] == "SPEAKER_00"


def test_response_omits_exclusive_when_none():
    resp = DiarizationResponse(
        segments=[DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_01")]
    )
    d = resp.to_dict()
    assert "exclusiveDiarization" not in d
