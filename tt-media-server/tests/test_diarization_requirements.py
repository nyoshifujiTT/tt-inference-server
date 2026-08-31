# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The diarization dependencies have to reach the venv the service runs in.

They live in their own requirements file so that what this one model adds is
visible in one place, which only helps if the Dockerfile actually installs it --
and installs it into the main venv, since the service offloads through ttnn and
ttnn exists nowhere else. A file nobody installs is worse than a mixed-in list:
the split reads as done while the dependency is missing at runtime.
"""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_REQUIREMENTS = _ROOT / "diarization-requirements.txt"
_DOCKERFILE = _ROOT / "Dockerfile"


def _packages(path: Path) -> set:
    names = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        for sep in (">=", "==", "<", ">", "["):
            line = line.split(sep)[0]
        names.add(line.strip().lower())
    return names


def test_the_file_holds_what_the_service_needs():
    packages = _packages(_REQUIREMENTS)
    # pyannote builds the pipeline, soundfile decodes, boto3 signs the
    # pre-signed PUT urls POST /v1/media/input hands out.
    assert {"pyannote.audio", "soundfile", "boto3"} <= packages


def test_the_dockerfile_copies_and_installs_it():
    dockerfile = _DOCKERFILE.read_text()
    assert "diarization-requirements.txt" in dockerfile, (
        "the requirements file is not referenced by the Dockerfile at all"
    )
    assert dockerfile.count("diarization-requirements.txt") >= 2, (
        "expected both a COPY and an install line"
    )


def test_it_is_installed_into_the_main_venv_not_the_audio_one():
    """ttnn only exists in the main venv, so this is where it has to land."""
    dockerfile = _DOCKERFILE.read_text()
    install_line = next(
        line
        for line in dockerfile.splitlines()
        if "-r diarization-requirements.txt" in line
    )
    audio_venv_block_start = dockerfile.index("AUDIO_VENV_DIR")
    assert dockerfile.index(install_line) < audio_venv_block_start, (
        "diarization requirements must install before the audio venv is built, "
        "into the main venv"
    )


def test_the_torch_floor_override_is_applied():
    """pyannote 4.x declares torch>=2.8; this image is pinned to 2.7.1."""
    dockerfile = _DOCKERFILE.read_text()
    install_line = next(
        line
        for line in dockerfile.splitlines()
        if "-r diarization-requirements.txt" in line
    )
    assert "uv-overrides-main.txt" in install_line, (
        "without the override the resolve pulls torch to 2.13 and breaks vLLM"
    )


@pytest.mark.parametrize("package", ["pyannote.audio", "soundfile", "boto3"])
def test_the_shared_requirements_no_longer_carry_them(package):
    """Otherwise the split is cosmetic and the two files can drift."""
    assert package not in _packages(_ROOT / "requirements.txt")
