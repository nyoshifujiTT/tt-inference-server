# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
Shared ffmpeg pipe-based conversion (same pattern as audio_manager).
Used by AudioManager (decode to WAV) and TTS post-process (encode WAV to MP3/OGG).
"""

import shutil
import subprocess
from typing import List


def ffmpeg_binary() -> str:
    """ffmpeg from PATH (the media server images install it), else the binary the
    `imageio_ffmpeg` wheel ships -- the same lookup the video path uses
    (``VideoManager._ffmpeg_binary``), so audio does not depend on a system package
    that an image built some other way may not carry."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "ffmpeg not found on PATH and imageio_ffmpeg is not available; "
            "install ffmpeg or `pip install imageio-ffmpeg`"
        ) from e


def run_ffmpeg_stdin_stdout(input_bytes: bytes, ffmpeg_args: List[str]) -> bytes:
    """
    Run ffmpeg with stdin/stdout pipes (same as audio_manager._decode_audio_file).
    Feeds input_bytes to stdin, returns stdout bytes. Raises on failure.
    """
    process = subprocess.Popen(
        [ffmpeg_binary()] + ffmpeg_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output_bytes, error_output = process.communicate(input=input_bytes)
    if process.returncode != 0:
        error_msg = (
            error_output.decode("utf-8") if error_output else "Unknown ffmpeg error"
        )
        raise subprocess.CalledProcessError(process.returncode, "ffmpeg", error_msg)
    return output_bytes


def decode_to_wav(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """
    Convert ffmpeg-supported input (MP3, OGG, etc.) to WAV bytes.
    Same args as audio_manager._decode_audio_file.
    """
    args = [
        "-i",
        "pipe:0",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "wav",
        "-y",
        "pipe:1",
    ]
    return run_ffmpeg_stdin_stdout(audio_bytes, args)


def encode_wav_to(wav_bytes: bytes, output_format: str) -> bytes:
    """
    Convert WAV bytes to MP3 or OGG. Used by TTS post-process.
    """
    if output_format == "mp3":
        args = ["-i", "pipe:0", "-f", "mp3", "-y", "pipe:1"]
    elif output_format == "ogg":
        args = [
            "-i",
            "pipe:0",
            "-acodec",
            "libvorbis",
            "-f",
            "ogg",
            "-y",
            "pipe:1",
        ]
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    return run_ffmpeg_stdin_stdout(wav_bytes, args)
