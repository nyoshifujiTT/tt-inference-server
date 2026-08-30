# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""DiarizationService device acquisition (device-independent)."""

import sys

import pytest


def _load_service(monkeypatch, captured):
    # Stub DiarizationBackend to capture nn_accelerator without needing pyannote.
    import model_services.diarization_service as svc

    class _FakeBackend:
        def __init__(self, model_path, device="cpu", nn_accelerator=None):
            captured["nn_accelerator"] = nn_accelerator

    monkeypatch.setattr(svc, "DiarizationBackend", _FakeBackend)
    return svc


def test_an_unresolved_device_spec_is_fatal(monkeypatch):
    """Refusing to start beats starting without the accelerator.

    This model is served because it runs on the device. A service that quietly
    drops to CPU still answers every request, so nothing surfaces the problem --
    that is how a broken TT_MESH_GRAPH_DESC_PATH survived a whole bring-up.
    """
    captured = {}
    svc = _load_service(monkeypatch, captured)
    monkeypatch.setattr(svc.settings, "device_ids", object(), raising=False)
    with pytest.raises(RuntimeError, match="no device resolved"):
        svc.DiarizationService()


def test_an_empty_device_spec_is_fatal(monkeypatch):
    """Same for a spec that resolves to nothing at all."""
    captured = {}
    svc = _load_service(monkeypatch, captured)
    monkeypatch.setattr(svc.settings, "device_ids", "", raising=False)
    with pytest.raises(RuntimeError, match="no device resolved"):
        svc.DiarizationService()


def test_device_comes_from_settings_not_a_private_env_var(monkeypatch):
    """The offload target is whatever the catalog resolved, as elsewhere.

    Requiring a diarization-specific env var here would mean the standard
    launch -- run.py passing only MODEL and DEVICE -- silently stayed on CPU.
    """
    captured = {}
    svc = _load_service(monkeypatch, captured)
    monkeypatch.setattr(svc.settings, "device_ids", "(3)", raising=False)

    service = svc.DiarizationService.__new__(svc.DiarizationService)
    service.logger = svc.TTLogger()
    assert service._resolve_device_id() == 3


def test_multi_device_settings_take_the_first(monkeypatch):
    """pyannote is single-device here; a wider mesh must not crash the parse."""
    captured = {}
    svc = _load_service(monkeypatch, captured)
    monkeypatch.setattr(svc.settings, "device_ids", "(0),(1),(2),(3)", raising=False)

    service = svc.DiarizationService.__new__(svc.DiarizationService)
    service.logger = svc.TTLogger()
    assert service._resolve_device_id() == 0


def test_an_unreachable_device_is_fatal(monkeypatch):
    """A device that cannot be opened must stop the service, not degrade it."""
    monkeypatch.setitem(sys.modules, "ttnn", None)  # import ttnn -> ImportError
    captured = {}
    svc = _load_service(monkeypatch, captured)
    monkeypatch.setattr(svc.settings, "device_ids", "(0)", raising=False)
    with pytest.raises(ImportError):
        svc.DiarizationService()


def _stub_device(monkeypatch, svc):
    """Give the service a device so construction reaches the backend.

    The service now refuses to run without one, so tests that are about
    something else still have to supply it.
    """
    import sys
    import types

    monkeypatch.setattr(svc.settings, "device_ids", "(0)", raising=False)
    fake_ttnn = types.ModuleType("ttnn")
    fake_ttnn.open_device = lambda device_id, l1_small_size: object()
    monkeypatch.setitem(sys.modules, "ttnn", fake_ttnn)
    fake_port = types.ModuleType("tt_nn_accelerator")
    fake_port.make_tt_accelerator = lambda device: lambda pipeline: None
    monkeypatch.setitem(sys.modules, "tt_nn_accelerator", fake_port)


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
    _stub_device(monkeypatch, svc)
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


def test_start_workers_warmup_failure_is_fatal(monkeypatch):
    """Warmup is the only thing that exercises the pipeline before traffic.

    Swallowing it lets a server that cannot diarize at all come up and report
    itself ready, with one warning line as the sole evidence.
    """
    monkeypatch.delenv("DIARIZATION_TT_DEVICE_ID", raising=False)
    import model_services.diarization_service as svc

    class _BoomBackend:
        def __init__(self, model_path, device="cpu", nn_accelerator=None):
            pass

        def diarize(self, path, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(svc, "DiarizationBackend", _BoomBackend)
    _stub_device(monkeypatch, svc)
    service = svc.DiarizationService()
    with pytest.raises(RuntimeError, match="boom"):
        service.start_workers()


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


def test_diarized_transcription_also_avoids_file_paths(monkeypatch):
    """The diarize+ASR route must not hand pyannote a path either.

    ``process_request`` was switched to an in-memory waveform but this route
    still wrote a temp file, so it kept hitting the broken torchcodec decode in
    the built image.
    """
    import asyncio

    import model_services.diarization_service as svc
    from domain.diarization_request import DiarizationRequest

    calls = []

    class _FakeBackend:
        def __init__(self, model_path, device="cpu", nn_accelerator=None):
            pass

        def diarize(self, audio, **kwargs):
            calls.append(audio)
            return {"segments": [], "exclusiveDiarization": []}

    monkeypatch.setattr(svc, "DiarizationBackend", _FakeBackend)
    monkeypatch.setattr(svc.settings, "default_sample_rate", 16000, raising=False)
    monkeypatch.setattr(svc.settings, "asr_url", "http://asr.invalid", raising=False)
    monkeypatch.setattr(svc, "decode_to_wav", lambda data, sample_rate=16000: data)
    monkeypatch.setattr(
        svc,
        "_wav_bytes_to_waveform",
        lambda wav_bytes: {"waveform": "tensor", "sample_rate": 16000},
    )
    _stub_device(monkeypatch, svc)

    service = svc.DiarizationService()
    monkeypatch.setattr(service, "_wav_bytes_to_samples", lambda wav_bytes: [])

    request = DiarizationRequest(file=b"RIFFxxxxWAVE")
    asyncio.run(
        service.diarized_transcription(
            request,
            "asr-model+pyannote/speaker-diarization-community-1",
        )
    )

    assert len(calls) == 1
    assert isinstance(calls[0], dict)  # never a filesystem path
    assert set(calls[0]) == {"waveform", "sample_rate"}


def test_readiness_reports_worker_info_for_the_liveness_gate(monkeypatch):
    """/tt-liveness must carry worker_info or the model cannot be benchmarked.

    ``server_tests/test_cases/device_liveness_test.py`` aborts with "No
    worker_info found in response" and then counts entries with ``is_ready``,
    so the benchmark client's health gate fails outright when the field is
    missing. This service has no Scheduler, so it reports its single
    in-process pipeline itself.
    """
    import model_services.diarization_service as svc

    class _FakeBackend:
        def __init__(self, model_path, device="cpu", nn_accelerator=None):
            pass

    monkeypatch.setattr(svc, "DiarizationBackend", _FakeBackend)
    _stub_device(monkeypatch, svc)
    status = svc.DiarizationService().check_is_model_ready()

    assert status["model_ready"] is True
    worker_info = status["worker_info"]
    assert worker_info, "liveness gate rejects an empty worker_info"
    ready = [w for w in worker_info.values() if w.get("is_ready")]
    assert len(ready) == 1  # one in-process pipeline; pyannote is serialized
    entry = ready[0]
    # Same shape Scheduler.get_worker_info() emits, so the gate can read it.
    assert set(entry) >= {"pid", "is_alive", "is_ready", "start_time"}
    assert entry["is_alive"] is True
