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


def _plan(corpus_root=None, published_corpus_der=None, corpus_audio=()):
    return {
        "published_der": 0.17,
        "published_der_ref": "https://huggingface.co/pyannote/speaker-diarization-community-1",
        "accuracy_der_max": 0.15,
        "corpus_der_tolerance": 0.05,
        "sample_audio": "/tmp/a.wav",
        "corpus_root": corpus_root,
        "corpus_audio": list(corpus_audio),
        "published_corpus_der": published_corpus_der,
    }


def _fake_scoring(plan, scored):
    """Stand-in for the EVALS_DIARIZATION bridge.

    The DER maths is tt-metal's and runs in its own venv; these tests check
    what the runner sends to it and the verdict it draws from the answer.
    """
    fake = MagicMock()
    fake.plan.return_value = plan
    fake.score.return_value = scored
    return fake


def _sample_score(der, num_speakers=2, reference_num_speakers=2):
    return {
        "der": der,
        "num_speakers": num_speakers,
        "reference_num_speakers": reference_num_speakers,
        "speaker_count_matches": num_speakers == reference_num_speakers,
    }


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
        mod, "scoring", fake
    ), patch.object(mod, "audio_duration_seconds", return_value=30.0), patch.object(
        mod, "diarize_once", side_effect=_fake_call
    ):
        return mod.run_diarization_eval(_ctx())


def test_sample_eval_scores_the_served_turns_against_the_shipped_annotation():
    fake = _fake_scoring(_plan(), _sample_score(0.052))
    block = _run(fake)

    fake.score.assert_called_once_with({"/tmp/a.wav": TURNS})
    assert block.kind == "evals"
    assert block.task_type == "diarization"
    assert block.data["score"] == pytest.approx(0.052)
    assert block.data["published_score"] == pytest.approx(0.17)
    assert block.data["task_name"] == "pyannote_sample_der"
    assert block.data["speaker_count_matches"] is True
    assert block.data["accuracy_check"] == ReportCheckTypes.PASS


def test_a_sample_der_over_the_ceiling_fails():
    block = _run(_fake_scoring(_plan(), _sample_score(0.30)))
    assert block.data["accuracy_check"] == ReportCheckTypes.FAIL


def test_a_wrong_speaker_count_fails_even_with_a_low_der():
    """A pipeline that merges speakers can still post an acceptable DER."""
    block = _run(_fake_scoring(_plan(), _sample_score(0.01, num_speakers=1)))

    assert block.data["speaker_count_matches"] is False
    assert block.data["accuracy_check"] == ReportCheckTypes.FAIL


def test_a_failed_request_is_not_scored():
    fake = _fake_scoring(_plan(), _sample_score(0.0))
    failed = DiarizationTestStatus(False, 0.0)

    with pytest.raises(RuntimeError):
        _run(fake, status=failed)
    fake.score.assert_not_called()


def _corpus_score(der):
    return {"der": der, "num_recordings": 2, "per_recording": {"a": der, "b": der}}


def test_corpus_recordings_are_all_sent_and_scored_together():
    plan = _plan("/corpus", 0.112, corpus_audio=["/corpus/audio/a.wav", "/corpus/audio/b.wav"])
    fake = _fake_scoring(plan, _corpus_score(0.12))

    block = _run(fake)

    fake.score.assert_called_once_with(
        {"/corpus/audio/a.wav": TURNS, "/corpus/audio/b.wav": TURNS},
        corpus_root="/corpus",
        limit=None,
    )
    assert block.data["task_name"] == "voxconverse-test_der"
    assert block.data["score"] == pytest.approx(0.12)
    assert block.data["num_recordings"] == 2
    # 0.12 is within 0.112 + 0.05, so this run passes.
    assert block.data["accuracy_check"] == ReportCheckTypes.PASS


def test_a_corpus_der_far_above_the_published_figure_fails():
    plan = _plan("/corpus", 0.112, corpus_audio=["/corpus/audio/a.wav"])
    block = _run(_fake_scoring(plan, _corpus_score(0.40)))
    assert block.data["accuracy_check"] == ReportCheckTypes.FAIL


def test_a_split_without_a_published_figure_is_not_scored_as_a_pass():
    """Dev-split runs land under the test-split number; that is not a pass."""
    plan = _plan("/corpus", None, corpus_audio=["/corpus/audio/a.wav"])
    block = _run(_fake_scoring(plan, _corpus_score(0.0705)))

    assert block.data["score"] == pytest.approx(0.0705)
    assert block.data["published_score"] is None
    assert block.data["accuracy_check"] == ReportCheckTypes.NA
