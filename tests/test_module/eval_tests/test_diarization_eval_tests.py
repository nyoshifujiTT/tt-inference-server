# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for the speaker-diarization eval runner (DER scoring)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from test_module._test_common import ReportCheckTypes
from test_module.eval_tests import diarization_eval_tests as mod
from test_module.test_status import DiarizationTestStatus


def _ctx():
    return SimpleNamespace(
        model_spec=SimpleNamespace(model_name="speaker-diarization-community-1"),
        device=SimpleNamespace(name="p150"),
        base_url="http://127.0.0.1:8018",
    )


def _fake_accuracy(reference_turns):
    """Stand-in for tt-metal's accuracy module.

    Real scoring is tested in tt-metal; these tests only need the runner to
    route through it, so this keeps them runnable without a tt-metal checkout
    while still exercising the real DER maths via pyannote.metrics.
    """
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def _to_annotation(turns):
        annotation = Annotation()
        for turn in turns:
            annotation[Segment(turn["start"], turn["end"])] = turn["speaker"]
        return annotation

    fake = MagicMock()
    fake.PUBLISHED_DER = 0.17
    fake.PUBLISHED_DER_REF = "https://huggingface.co/pyannote/speaker-diarization-community-1"
    fake.ACCURACY_DER_MAX = 0.15
    fake.CORPUS_DER_TOLERANCE = 0.05
    # No corpus unless a test says otherwise; a MagicMock here would be truthy
    # and silently pick the corpus path.
    fake.corpus_root.return_value = None
    fake.sample_audio_path.return_value = "/tmp/a.wav"
    fake.sample_reference_path.return_value = "/tmp/a.rttm"
    fake.load_rttm.return_value = _to_annotation(reference_turns)
    fake.turns_to_annotation.side_effect = _to_annotation

    def _score(hypothesis, reference):
        return {
            "der": float(DiarizationErrorRate()(reference, hypothesis)),
            "num_speakers": len(hypothesis.labels()),
            "reference_num_speakers": len(reference.labels()),
            "speaker_count_matches": len(hypothesis.labels())
            == len(reference.labels()),
        }

    fake.score_against_reference.side_effect = _score
    return fake


TURNS = [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
    {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
]


def _status(turns=TURNS):
    return DiarizationTestStatus(
        True, 5.0, ttft_ms=4900.0, rtr=6.0, num_speakers=2, num_turns=2, turns=turns
    )


def _run(fake, status=None):
    async def _fake_call(*args, **kwargs):
        return status if status is not None else _status()

    with patch.object(mod, "require_health", return_value="diarization-cpu"), patch.object(
        mod, "_accuracy", return_value=fake
    ), patch.object(mod, "audio_duration_seconds", return_value=30.0), patch.object(
        mod, "diarize_once", side_effect=_fake_call
    ):
        return mod.run_diarization_eval(_ctx())


def test_sample_eval_scores_a_der_against_the_shipped_annotation():
    block = _run(_fake_accuracy(reference_turns=TURNS))

    assert block.kind == "evals"
    assert block.task_type == "diarization"
    # Hypothesis equals the reference here, so the DER must be exactly 0.
    assert block.data["score"] == pytest.approx(0.0)
    assert block.data["task_name"] == "pyannote_sample_der"
    assert block.data["speaker_count_matches"] is True
    assert block.data["accuracy_check"] == ReportCheckTypes.PASS


def test_a_wrong_speaker_count_fails_even_with_a_low_der():
    """A pipeline that merges speakers can still post an acceptable DER."""
    merged = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0}]
    status = DiarizationTestStatus(
        True, 5.0, rtr=6.0, num_speakers=1, num_turns=1, turns=merged
    )

    block = _run(_fake_accuracy(reference_turns=TURNS), status=status)

    assert block.data["speaker_count_matches"] is False
    assert block.data["accuracy_check"] == ReportCheckTypes.FAIL


def test_a_failed_request_is_not_scored():
    fake = _fake_accuracy(reference_turns=TURNS)
    failed = DiarizationTestStatus(False, 0.0)

    with pytest.raises(RuntimeError):
        _run(fake, status=failed)


def test_corpus_result_is_scored_against_the_published_figure():
    fake = _fake_accuracy(reference_turns=[])
    fake.corpus_root.return_value = "/corpus"
    fake.published_corpus_der.return_value = 0.112
    fake.corpus_der.return_value = {
        "der": 0.12,
        "num_recordings": 3,
        "per_recording": {"a": 0.10, "b": 0.11, "c": 0.15},
    }

    block = _run(fake)

    assert block.data["task_name"] == "voxconverse-test_der"
    assert block.data["score"] == pytest.approx(0.12)
    assert block.data["num_recordings"] == 3
    # 0.12 is within 0.112 + 0.05, so this run passes.
    assert block.data["accuracy_check"] == ReportCheckTypes.PASS
    fake.sample_audio_path.assert_not_called()


def test_a_corpus_der_far_above_the_published_figure_fails():
    fake = _fake_accuracy(reference_turns=[])
    fake.corpus_root.return_value = "/corpus"
    fake.published_corpus_der.return_value = 0.112
    fake.corpus_der.return_value = {
        "der": 0.40,  # published 0.112 + tolerance 0.05 = 0.162
        "num_recordings": 3,
        "per_recording": {"a": 0.4},
    }

    block = _run(fake)

    assert block.data["accuracy_check"] == ReportCheckTypes.FAIL


def test_a_split_without_a_published_figure_is_not_scored_as_a_pass():
    """Dev-split runs land under the test-split number; that is not a pass."""
    fake = _fake_accuracy(reference_turns=[])
    fake.corpus_root.return_value = "/corpus"
    fake.published_corpus_der.return_value = None
    fake.corpus_der.return_value = {
        "der": 0.0705,  # what dev actually scores
        "num_recordings": 216,
        "per_recording": {"a": 0.07},
    }

    block = _run(fake)

    assert block.data["score"] == pytest.approx(0.0705)
    assert block.data["published_score"] is None
    assert block.data["accuracy_check"] == ReportCheckTypes.NA
