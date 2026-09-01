# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The audio download cap has to be big enough for a recording.

``media_url_max_bytes`` defaults to 7,500,000 across the whole server, a figure
derived from one input image: it is what base64-encodes to the video request's
MAX_BASE64_IMAGE_LEN. Diarization pulls a whole recording through the same
downloader, and 16 kHz mono 16-bit PCM runs 1.92 MB per minute, so the default
would refuse anything past four minutes. Raised on the model's own config entry
rather than globally, so image and video keep theirs.
"""

import subprocess
import sys

from workflows.model_spec import load_templates_from_yaml
from workflows.utils import get_repo_root_path
from workflows.workflow_types import DeviceTypes

# 16 kHz, mono, 16-bit PCM -- what the service decodes uploads to.
_BYTES_PER_MINUTE = 16000 * 2 * 60


def _resolved_setting(name: str) -> int:
    """Read a setting out of a real Settings, resolved from MODEL + DEVICE.

    A subprocess because tt-media-server's conftest replaces config.settings
    with a Mock for the rest of its suite.
    """
    script = (
        "from config.settings import Settings;"
        "s = Settings();"
        f"print(getattr(s, {name!r}))"
    )
    env = {
        "MODEL": "speaker-diarization-community-1",
        "DEVICE": "p150",
        "PYTHONPATH": ".",
        "PATH": "/usr/bin:/bin",
    }
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(get_repo_root_path() / "tt-media-server"),
        check=True,
    )
    return int(out.stdout.strip().splitlines()[-1])


def test_the_cap_is_raised_for_this_model():
    assert _resolved_setting("media_url_max_bytes") > 7_500_000


def test_the_cap_carries_at_least_half_an_hour_of_audio():
    """Long enough for a meeting, which is the workload this model is for."""
    minutes = _resolved_setting("media_url_max_bytes") / _BYTES_PER_MINUTE
    assert minutes >= 30, f"cap only carries {minutes:.1f} minutes of audio"


def test_the_raise_is_scoped_to_this_model():
    """A global raise would let oversized images through the video path too.

    Read from the settings source rather than importing it: config/ lives under
    tt-media-server and is not on this suite's path.
    """
    settings_src = (
        get_repo_root_path() / "tt-media-server" / "config" / "settings.py"
    ).read_text()
    assert "media_url_max_bytes: int = 7_500_000" in settings_src, (
        "the server-wide default moved; the per-model raise may now be redundant "
        "or the image/video paths may have been widened by accident"
    )


def test_the_catalog_entry_is_the_one_carrying_it():
    """Pinned to the spec so the value cannot drift out of the catalog."""
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "audio_tts.yaml"
    )
    diarization = [
        t for t in templates if "pyannote/speaker-diarization-community-1" in t.weights
    ]
    assert diarization, "the diarization template disappeared from the dev catalog"
    devices = {d.device for d in diarization[0].device_model_specs}
    assert DeviceTypes.P150 in devices
