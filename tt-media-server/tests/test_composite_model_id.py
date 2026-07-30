# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import pytest
from utils.composite_model_id import (
    CompositeModelIdError,
    ParsedModelId,
    parse_model_id,
)


def test_single_model_is_asr_only():
    p = parse_model_id("neosophie/Qwen3-ASR-1.7B-JA")
    assert p.asr_model == "neosophie/Qwen3-ASR-1.7B-JA"
    assert p.diarization_model is None
    assert p.wants_diarization is False
    assert p.canonical() == "neosophie/Qwen3-ASR-1.7B-JA"


def test_composite_asr_plus_diarization():
    p = parse_model_id(
        "neosophie/Qwen3-ASR-1.7B-JA+pyannote/speaker-diarization-community-1"
    )
    assert p.asr_model == "neosophie/Qwen3-ASR-1.7B-JA"
    assert p.diarization_model == "pyannote/speaker-diarization-community-1"
    assert p.wants_diarization is True
    assert p.canonical() == (
        "neosophie/Qwen3-ASR-1.7B-JA+pyannote/speaker-diarization-community-1"
    )


def test_whitespace_is_trimmed():
    p = parse_model_id("  asr-model  +  diar-model  ")
    assert p.asr_model == "asr-model"
    assert p.diarization_model == "diar-model"


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_empty_model_is_error(bad):
    with pytest.raises(CompositeModelIdError):
        parse_model_id(bad)


@pytest.mark.parametrize("bad", ["asr+", "+diar", "asr++diar", "a+b+c"])
def test_malformed_composite_is_error(bad):
    with pytest.raises(CompositeModelIdError):
        parse_model_id(bad)


def test_parsed_model_id_is_frozen():
    p = ParsedModelId(asr_model="a")
    with pytest.raises(Exception):
        p.asr_model = "b"  # frozen dataclass
