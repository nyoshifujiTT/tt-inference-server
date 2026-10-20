# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The bridge to the EVALS_DIARIZATION venv runs the real runner in a subprocess.

The venv python is this interpreter and tt-metal's accuracy module is a small
stand-in on PYTHONPATH, so the runner script, its CLI and the JSON hand-off are
the real ones; only the DER maths (tested in tt-metal) is replaced.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_module import _diarization_scoring as scoring
from workflows.workflow_types import WorkflowVenvType

ACCURACY = textwrap.dedent(
    """
    import os
    PUBLISHED_DER = 0.170
    PUBLISHED_DER_REF = "ref"
    ACCURACY_DER_MAX = 0.15
    CORPUS_DER_TOLERANCE = 0.05
    ROOT = os.path.dirname(__file__)

    def sample_audio_path():
        return os.path.join(ROOT, "sample.wav")

    def sample_reference_path():
        return os.path.join(ROOT, "sample.rttm")

    def load_rttm(path):
        return ["ref", path]

    def turns_to_annotation(turns):
        return turns

    def score_against_reference(hypothesis, reference):
        speakers = {t["speaker"] for t in hypothesis}
        return {"der": 0.25, "num_speakers": len(speakers),
                "reference_num_speakers": 2,
                "speaker_count_matches": len(speakers) == 2}

    def corpus_root(name):
        return os.environ.get("DIARIZATION_CORPUS_DIR")

    def published_corpus_der(name):
        return {"voxconverse-test": 0.112}.get(name)

    def corpus_files(root, limit=None):
        pairs = [(i, os.path.join(root, i + ".wav"), None) for i in ("a", "b", "c")]
        return pairs[:limit] if limit else pairs

    def corpus_der(diarize, root, limit=None):
        per = {i: float(len(diarize(wav))) for i, wav, _ in corpus_files(root, limit)}
        return {"der": sum(per.values()), "num_recordings": len(per), "per_recording": per}
    """
)


@pytest.fixture
def venv(tmp_path, monkeypatch):
    package = tmp_path / "models" / "demos" / "audio" / "pyannote_diarization"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "accuracy.py").write_text(ACCURACY)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setitem(
        scoring.VENV_CONFIGS,
        WorkflowVenvType.EVALS_DIARIZATION,
        SimpleNamespace(venv_python=Path(sys.executable)),
    )
    return package


def test_sample_path_comes_from_the_scoring_venv(venv):
    assert scoring.sample_audio_path() == str(venv / "sample.wav")


def test_plan_without_a_corpus_points_at_the_sample(venv, monkeypatch):
    monkeypatch.delenv("DIARIZATION_CORPUS_DIR", raising=False)
    plan = scoring.plan("voxconverse-test")
    assert plan["corpus_root"] is None
    assert plan["sample_audio"] == str(venv / "sample.wav")
    assert plan["accuracy_der_max"] == 0.15
    assert plan["published_der"] == 0.170


def test_plan_with_a_corpus_lists_its_recordings(venv, monkeypatch, tmp_path):
    monkeypatch.setenv("DIARIZATION_CORPUS_DIR", "/corpus")
    plan = scoring.plan("voxconverse-test", limit=2)
    assert plan["corpus_root"] == "/corpus"
    assert plan["corpus_audio"] == ["/corpus/a.wav", "/corpus/b.wav"]
    assert plan["published_corpus_der"] == 0.112


def test_sample_score_is_computed_from_the_turns_sent(venv):
    turns = [{"speaker": "A", "start": 0, "end": 1}, {"speaker": "B", "start": 1, "end": 2}]
    scored = scoring.score({str(venv / "sample.wav"): turns})
    assert scored == {"der": 0.25, "num_speakers": 2,
                      "reference_num_speakers": 2, "speaker_count_matches": True}


def test_corpus_score_uses_the_turns_of_each_recording(venv):
    turn = {"speaker": "A", "start": 0, "end": 1}
    scored = scoring.score(
        {"/corpus/a.wav": [turn], "/corpus/b.wav": [turn, turn]},
        corpus_root="/corpus",
        limit=2,
    )
    assert scored["per_recording"] == {"a": 1.0, "b": 2.0}
    assert scored["der"] == 3.0


def test_a_runner_failure_is_raised_not_swallowed(venv):
    with pytest.raises(subprocess.CalledProcessError):
        scoring.score({"/not/the/sample.wav": []})


def test_an_unprovisioned_venv_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setitem(
        scoring.VENV_CONFIGS,
        WorkflowVenvType.EVALS_DIARIZATION,
        SimpleNamespace(venv_python=tmp_path / "missing" / "python"),
    )
    with pytest.raises(FileNotFoundError):
        scoring.sample_audio_path()
