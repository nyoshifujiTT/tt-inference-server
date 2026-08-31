# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""DiarizationService is a plain BaseService.

It used to own a pipeline, a device and a warmup of its own, which is how a
mismatched mesh descriptor became a server quietly running on CPU. Everything
about the device now belongs to the Scheduler and TTDiarizationRunner; what is
left here is decoding the request and shaping the response, so that is what
these cover.
"""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest


def _service(monkeypatch):
    import model_services.diarization_service as svc

    monkeypatch.setattr(svc.settings, "default_sample_rate", 16000, raising=False)
    service = svc.DiarizationService.__new__(svc.DiarizationService)
    service.logger = svc.TTLogger() if hasattr(svc, "TTLogger") else MagicMock()
    service.scheduler = MagicMock()
    return svc, service


def test_the_service_is_a_base_service():
    """Being one is what supplies the Scheduler, health and worker lifecycle."""
    from model_services.base_service import BaseService
    from model_services.diarization_service import DiarizationService

    assert issubclass(DiarizationService, BaseService)


@pytest.mark.parametrize(
    "removed",
    [
        "_resolve_device_id",
        "_build_tt_accelerator",
        "warmup",
        "check_is_model_ready",
        "stop_workers",
    ],
)
def test_the_service_no_longer_reimplements_the_standard_machinery(removed):
    """Each of these had a hand-written copy that bypassed the standard path.

    ``check_is_model_ready`` and ``stop_workers`` still exist -- inherited from
    BaseService, which answers them from the Scheduler -- so the check is that
    the class does not define its own.
    """
    from model_services.diarization_service import DiarizationService

    assert removed not in vars(DiarizationService)


def test_preprocessing_decodes_audio_for_the_runner(monkeypatch):
    """The runner receives samples; ffmpeg stays off the device worker."""
    from domain.diarization_request import DiarizationRequest

    svc, service = _service(monkeypatch)
    monkeypatch.setattr(svc, "decode_to_wav", lambda data, sample_rate=16000: data)
    monkeypatch.setattr(
        service, "_wav_bytes_to_samples", lambda wav: np.zeros(4, dtype=np.float32)
    )

    request = asyncio.run(service.pre_process(DiarizationRequest(file=b"RIFFWAVE")))

    assert request._audio_array is not None
    assert len(request._audio_array) == 4


def test_postprocessing_shapes_the_pyannoteai_response(monkeypatch):
    from domain.diarization_request import DiarizationRequest

    _svc, service = _service(monkeypatch)
    result = {
        "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
        "exclusiveDiarization": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
    }

    response = asyncio.run(
        service.post_process(result, DiarizationRequest(file=b"x", num_speakers=1))
    )

    assert [s.speaker for s in response.segments] == ["SPEAKER_00"]
    assert response.exclusiveDiarization is not None


def test_a_speaker_count_mismatch_is_reported_as_a_warning(monkeypatch):
    """The hint is advisory, so it surfaces rather than failing the request."""
    from domain.diarization_request import DiarizationRequest

    _svc, service = _service(monkeypatch)
    result = {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]}

    response = asyncio.run(
        service.post_process(result, DiarizationRequest(file=b"x", num_speakers=3))
    )

    assert response.warning


def test_wav_bytes_to_waveform_decodes_without_torchcodec():
    """torchcodec cannot load against the torch pin this image ships."""
    import io
    import wave

    import model_services.diarization_service as svc

    sample_rate = 16000
    pcm = (np.linspace(-0.5, 0.5, 800) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())

    got = svc._wav_bytes_to_waveform(buf.getvalue())

    assert set(got) == {"waveform", "sample_rate"}
    assert got["sample_rate"] == sample_rate
