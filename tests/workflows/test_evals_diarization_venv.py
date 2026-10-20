# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The EVALS_DIARIZATION venv fetches tt-metal's scoring code at the image's commit."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows import workflow_venvs
from workflows.workflow_types import WorkflowVenvType


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def metal_remote(tmp_path):
    """A tt-metal stand-in with the scoring package and an unrelated tree."""
    repo = tmp_path / "tt-metal"
    package = repo / workflow_venvs.DIARIZATION_ACCURACY_PATH
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "accuracy.py").write_text("PUBLISHED_DER = 0.170\n")
    (repo / "ttnn").mkdir()
    (repo / "ttnn" / "big.py").write_text("x = 1\n")
    _git("init", "-q", cwd=repo)
    _git("config", "uploadpack.allowFilter", "true", cwd=repo)
    _git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=repo)
    _git("add", ".", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c", cwd=repo)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    return f"file://{repo}", commit


def _venv(tmp_path):
    venv = tmp_path / ".venv_evals_diarization"
    (venv / "lib" / "python3.10" / "site-packages").mkdir(parents=True)
    return workflow_venvs.VenvConfig(
        venv_type=WorkflowVenvType.EVALS_DIARIZATION, venv_path=venv
    )


def test_registered_with_its_requirements_and_hook():
    cfg = workflow_venvs.VENV_CONFIGS[WorkflowVenvType.EVALS_DIARIZATION]
    assert cfg.requirements_file == "evals-diarization.txt"
    assert cfg.setup_function is workflow_venvs.setup_evals_diarization
    requirements = (
        Path(workflow_venvs.__file__).resolve().parents[1]
        / "requirements"
        / cfg.requirements_file
    )
    assert "pyannote.audio" in requirements.read_text()


def test_checks_out_only_the_scoring_package_at_the_image_commit(
    tmp_path, metal_remote, monkeypatch
):
    url, commit = metal_remote
    monkeypatch.setattr(workflow_venvs, "TT_METAL_REPO_URL", url)
    cfg = _venv(tmp_path)

    assert workflow_venvs.setup_evals_diarization(
        cfg, SimpleNamespace(tt_metal_commit=commit)
    )

    checkout = workflow_venvs.get_diarization_metal_checkout(cfg)
    assert (checkout / workflow_venvs.DIARIZATION_ACCURACY_PATH / "accuracy.py").is_file()
    assert not (checkout / "ttnn").exists()
    pth = cfg.venv_path / "lib" / "python3.10" / "site-packages" / "tt-metal.pth"
    assert pth.read_text().strip() == str(checkout)

    # Re-running on an existing checkout (the venv is reused across runs) works.
    assert workflow_venvs.setup_evals_diarization(
        cfg, SimpleNamespace(tt_metal_commit=commit)
    )


def test_refuses_to_guess_when_the_image_commit_is_unknown(tmp_path):
    """Scoring with some other commit's accuracy.py would not be the same code."""
    assert not workflow_venvs.setup_evals_diarization(
        _venv(tmp_path), SimpleNamespace(tt_metal_commit=None)
    )


def test_a_commit_the_remote_does_not_have_fails_the_setup(
    tmp_path, metal_remote, monkeypatch
):
    url, _ = metal_remote
    monkeypatch.setattr(workflow_venvs, "TT_METAL_REPO_URL", url)
    assert not workflow_venvs.setup_evals_diarization(
        _venv(tmp_path), SimpleNamespace(tt_metal_commit="0" * 40)
    )
