# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import wave
import io

import numpy as np
from utils import asr_http_client


def test_encode_wav_pcm16_roundtrip():
    sig = np.arange(1600, dtype=np.float32) / 1600.0  # 0..~1
    wav = asr_http_client.encode_wav_pcm16(sig, 16000)
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        assert w.getnframes() == 1600


def test_transcribe_wav_bytes_posts_and_returns_text(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"text": "  hello world  "}

    def fake_post(url, files=None, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["files"] = list((files or {}).keys())
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(asr_http_client.requests, "post", fake_post)
    text = asr_http_client.transcribe_wav_bytes(
        "http://asr-host:9011",
        "Qwen3-ASR-1.7B-JA",
        b"WAVBYTES",
        language="ja",
        timeout=42,
    )
    assert text == "hello world"
    assert captured["url"] == "http://asr-host:9011/v1/audio/transcriptions"
    assert captured["data"]["model"] == "Qwen3-ASR-1.7B-JA"
    assert captured["data"]["language"] == "ja"
    assert captured["files"] == ["file"]
    assert captured["timeout"] == 42
