# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""DiarizationService TT-acceleration env toggle (device-independent)."""

import sys


def _load_service(monkeypatch, captured):
    # Stub DiarizationBackend to capture nn_accelerator without needing pyannote.
    import model_services.diarization_service as svc

    class _FakeBackend:
        def __init__(self, model_path, device="cpu", nn_accelerator=None):
            captured["nn_accelerator"] = nn_accelerator

    monkeypatch.setattr(svc, "DiarizationBackend", _FakeBackend)
    return svc


def test_no_tt_env_means_cpu_no_accelerator(monkeypatch):
    monkeypatch.delenv("DIARIZATION_TT_DEVICE_ID", raising=False)
    captured = {}
    svc = _load_service(monkeypatch, captured)
    svc.DiarizationService()
    assert captured["nn_accelerator"] is None


def test_tt_env_but_ttnn_unavailable_falls_back_to_cpu(monkeypatch):
    monkeypatch.setenv("DIARIZATION_TT_DEVICE_ID", "0")
    # Ensure importing ttnn fails -> graceful fallback (nn_accelerator None)
    monkeypatch.setitem(sys.modules, "ttnn", None)  # import ttnn -> ImportError
    captured = {}
    svc = _load_service(monkeypatch, captured)
    svc.DiarizationService()
    assert captured["nn_accelerator"] is None


# --- warmup lifecycle tests (appended) ---


def _load_service_capturing_backend(monkeypatch, calls):
    import model_services.diarization_service as svc

    # In a full-suite run conftest may have left settings.default_sample_rate as
    # a Mock, which makes warmup()'s int() conversion raise and get swallowed by
    # start_workers. Pin it so the warmup path is actually exercised.
    monkeypatch.setattr(svc.settings, "default_sample_rate", 16000, raising=False)

    class _FakeBackend:
        def __init__(self, model_path, device="cpu", nn_accelerator=None):
            self.model_path = model_path

        def diarize(self, path, **kwargs):
            calls.append(path)
            return {"segments": [], "exclusiveDiarization": None}

    monkeypatch.setattr(svc, "DiarizationBackend", _FakeBackend)
    return svc


def test_start_workers_warms_up_with_one_diarize(monkeypatch):
    monkeypatch.delenv("DIARIZATION_TT_DEVICE_ID", raising=False)
    calls = []
    svc = _load_service_capturing_backend(monkeypatch, calls)
    service = svc.DiarizationService()
    assert calls == []  # constructing the service must not run inference
    service.start_workers()
    assert len(calls) == 1  # exactly one warmup diarization
    # Audio is handed to pyannote as an in-memory waveform, not a path, so the
    # pipeline never decodes a file through torchcodec. (torch is stubbed by
    # conftest here, so only the mapping shape is asserted.)
    warmup_audio = calls[0]
    assert isinstance(warmup_audio, dict)
    assert set(warmup_audio) == {"waveform", "sample_rate"}
    assert warmup_audio["sample_rate"] == 16000


def test_start_workers_warmup_failure_is_non_fatal(monkeypatch):
    monkeypatch.delenv("DIARIZATION_TT_DEVICE_ID", raising=False)
    import model_services.diarization_service as svc

    class _BoomBackend:
        def __init__(self, model_path, device="cpu", nn_accelerator=None):
            pass

        def diarize(self, path, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(svc, "DiarizationBackend", _BoomBackend)
    service = svc.DiarizationService()
    assert service.start_workers() is None  # warmup failure must be swallowed


def test_wav_bytes_to_waveform_decodes_without_torchcodec():
    """The helper must decode PCM itself; torchcodec cannot load on this torch pin."""
    import io
    import wave

    import numpy as np

    sample_rate = 16000
    pcm = (np.linspace(-0.5, 0.5, 800) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())

    import model_services.diarization_service as svc

    got = svc._wav_bytes_to_waveform(buf.getvalue())

    assert set(got) == {"waveform", "sample_rate"}
    assert got["sample_rate"] == sample_rate
