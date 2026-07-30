# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Minimal OpenAI-compatible ASR HTTP client for per-turn transcription.

Used by the diarized-transcription path to transcribe each speaker turn against
an external ASR server (e.g. the Qwen3-ASR vLLM /v1/audio/transcriptions). Kept
tiny and dependency-injected so the coordinator remains device-independent.
"""

from __future__ import annotations

import io
import wave
from typing import Optional

import requests


def encode_wav_pcm16(samples, sample_rate: int) -> bytes:
    """Encode a float [-1,1] or int16 waveform to a 16-bit PCM mono WAV (bytes)."""
    import numpy as np

    arr = np.asarray(samples)
    if arr.dtype != np.int16:
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(arr.tobytes())
    return buf.getvalue()


def transcribe_wav_bytes(
    asr_url: str,
    model: str,
    wav_bytes: bytes,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    timeout: int = 600,
) -> str:
    """POST WAV to {asr_url}/v1/audio/transcriptions and return the text."""
    url = asr_url.rstrip("/") + "/v1/audio/transcriptions"
    files = {"file": ("turn.wav", wav_bytes, "audio/wav")}
    data = {"model": model, "temperature": 0}
    if language and language.lower() != "auto":
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    r = requests.post(url, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    # OpenAI transcription returns {"text": "..."}; be tolerant of shapes.
    if isinstance(body, dict):
        return (body.get("text") or "").strip()
    return str(body).strip()
