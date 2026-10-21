# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Audio decoding finds ffmpeg the way the video path does.

The media server images install ffmpeg as a system package, but an image built
another way (a tt-model container, say) may only have the binary the
``imageio_ffmpeg`` wheel ships. The video path already falls back to it; audio
must too, or every diarization and TTS request fails on "ffmpeg not found".
"""

import subprocess
import sys
import types
from unittest.mock import patch

import pytest

from utils import ffmpeg_utils


def test_ffmpeg_on_path_wins():
    with patch("utils.ffmpeg_utils.shutil.which", return_value="/usr/bin/ffmpeg"):
        assert ffmpeg_utils.ffmpeg_binary() == "/usr/bin/ffmpeg"


def test_falls_back_to_the_imageio_ffmpeg_binary():
    fake = types.SimpleNamespace(get_ffmpeg_exe=lambda: "/venv/imageio_ffmpeg/ffmpeg")
    with patch("utils.ffmpeg_utils.shutil.which", return_value=None), patch.dict(
        sys.modules, {"imageio_ffmpeg": fake}
    ):
        assert ffmpeg_utils.ffmpeg_binary() == "/venv/imageio_ffmpeg/ffmpeg"


def test_neither_is_an_error_naming_both():
    with patch("utils.ffmpeg_utils.shutil.which", return_value=None), patch.dict(
        sys.modules, {"imageio_ffmpeg": None}
    ):
        with pytest.raises(RuntimeError, match="imageio_ffmpeg"):
            ffmpeg_utils.ffmpeg_binary()


def test_decoding_runs_the_resolved_binary():
    proc = types.SimpleNamespace(
        communicate=lambda input: (b"RIFF", b""), returncode=0
    )
    with patch.object(ffmpeg_utils, "ffmpeg_binary", return_value="/x/ffmpeg"), patch(
        "utils.ffmpeg_utils.subprocess.Popen", return_value=proc
    ) as popen:
        assert ffmpeg_utils.decode_to_wav(b"mp3") == b"RIFF"
    assert popen.call_args.args[0][0] == "/x/ffmpeg"


def test_a_failed_decode_still_raises():
    proc = types.SimpleNamespace(
        communicate=lambda input: (b"", b"bad input"), returncode=1
    )
    with patch.object(ffmpeg_utils, "ffmpeg_binary", return_value="/x/ffmpeg"), patch(
        "utils.ffmpeg_utils.subprocess.Popen", return_value=proc
    ):
        with pytest.raises(subprocess.CalledProcessError):
            ffmpeg_utils.decode_to_wav(b"junk")
