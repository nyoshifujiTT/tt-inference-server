# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Bridge from the workflow process to the EVALS_DIARIZATION venv.

The diarization eval and benchmark run in WORKFLOW_RUN_SCRIPT, which has
neither pyannote nor tt-metal on its path. Anything that needs them --
locating the reference clip, listing a corpus, computing a DER -- goes through
``eval_tests/_diarization_der_runner.py`` executed by that venv's python.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from workflows.workflow_types import WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger(__name__)

RUNNER = Path(__file__).parent / "eval_tests" / "_diarization_der_runner.py"
RESULT_START = "===DIARIZATION_RESULT_JSON_START==="
RESULT_END = "===DIARIZATION_RESULT_JSON_END==="


def _venv_python() -> Path:
    venv_python = VENV_CONFIGS[WorkflowVenvType.EVALS_DIARIZATION].venv_python
    if not venv_python.is_file():
        raise FileNotFoundError(
            f"EVALS_DIARIZATION venv python not found at {venv_python}; "
            "venv not provisioned."
        )
    return venv_python


def _parse(stdout: str) -> dict:
    if RESULT_START not in stdout or RESULT_END not in stdout:
        raise ValueError("diarization scoring runner produced no result block")
    section = stdout.split(RESULT_START, 1)[1].split(RESULT_END, 1)[0]
    return json.loads(section)


def _run(args: list, stdin: Optional[str] = None) -> dict:
    cmd = [str(_venv_python()), str(RUNNER), *args]
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(
            "diarization scoring runner exited %s\n--- stdout ---\n%s\n"
            "--- stderr ---\n%s",
            proc.returncode,
            proc.stdout,
            proc.stderr,
        )
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )
    return _parse(proc.stdout)


def plan(corpus_name: str, limit: Optional[int] = None) -> dict:
    """What to diarize and the thresholds to judge it by."""
    args = ["plan", "--corpus-name", corpus_name]
    if limit:
        args += ["--limit", str(limit)]
    return _run(args)


def score(turns_by_audio: dict, corpus_root: Optional[str] = None,
          limit: Optional[int] = None) -> dict:
    """DER of the served turns, computed by tt-metal's accuracy module."""
    request = {"turns": turns_by_audio, "corpus_root": corpus_root, "limit": limit}
    return _run(["score"], stdin=json.dumps(request))


def sample_audio_path() -> str:
    """pyannote's bundled 30 s two-speaker sample, located from its venv."""
    return _run(["sample"])["sample_audio"]
