# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from utils.diarization_backend import (
    DiarizationBackend,
    annotation_to_segments,
    build_pipeline_kwargs,
)


class _Turn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeAnnotation:
    """Mimics pyannote.core.Annotation.itertracks(yield_label=True)."""

    def __init__(self, triples):
        self._triples = triples  # list of (start, end, speaker)

    def itertracks(self, yield_label=True):
        for start, end, spk in self._triples:
            yield _Turn(start, end), None, spk


class _FakeOutput:
    def __init__(self, diar, excl):
        self.speaker_diarization = diar
        self.exclusive_speaker_diarization = excl


class _FakePipeline:
    def __init__(self, output):
        self._output = output
        self.calls = []

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return self._output


def test_annotation_to_segments_rounds_and_labels():
    ann = _FakeAnnotation([(0.2000001, 1.6009, "SPEAKER_00"), (1.6009, 2.0, "SPEAKER_01")])
    segs = annotation_to_segments(ann)
    assert segs == [
        {"start": 0.2, "end": 1.601, "speaker": "SPEAKER_00"},
        {"start": 1.601, "end": 2.0, "speaker": "SPEAKER_01"},
    ]


def test_build_pipeline_kwargs_precedence():
    assert build_pipeline_kwargs(3, 2, 5) == {"num_speakers": 3}
    assert build_pipeline_kwargs(None, 2, 5) == {"min_speakers": 2, "max_speakers": 5}
    assert build_pipeline_kwargs(None, None, None) == {}
    assert build_pipeline_kwargs(None, None, 4) == {"max_speakers": 4}


def _make_backend_with_fake(output):
    b = DiarizationBackend(model_path="/does/not/matter")
    b._pipeline = _FakePipeline(output)  # inject to skip real load
    return b


def test_diarize_returns_both_views_when_exclusive():
    out = _FakeOutput(
        _FakeAnnotation([(0.0, 1.0, "SPEAKER_00")]),
        _FakeAnnotation([(0.0, 1.0, "SPEAKER_00")]),
    )
    b = _make_backend_with_fake(out)
    res = b.diarize("audio.wav", num_speakers=2, exclusive=True)
    assert res["segments"] == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    assert res["exclusiveDiarization"] == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    # num_speakers passed through to pipeline
    assert b._pipeline.calls[0][1] == {"num_speakers": 2}


def test_diarize_omits_exclusive_when_disabled():
    out = _FakeOutput(_FakeAnnotation([(0.0, 1.0, "SPEAKER_00")]), _FakeAnnotation([]))
    b = _make_backend_with_fake(out)
    res = b.diarize("audio.wav", exclusive=False)
    assert "exclusiveDiarization" not in res
    assert b._pipeline.calls[0][1] == {}
