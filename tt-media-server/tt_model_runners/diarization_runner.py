# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os

from config.constants import SupportedModels
from domain.diarization_request import DiarizationRequest
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from tt_port.tt_nn_accelerator import make_tt_accelerator
from utils.decorators import log_execution_time
from utils.diarization_backend import DiarizationBackend

# community-1's two nets need a modest L1 scratch reservation; the ttnn convs in
# the WeSpeaker backbone allocate their halo/reader buffers out of it.
DIARIZATION_L1_SMALL_SIZE = 32768

# Long enough to span the segmentation window (10 s) so its kernels compile
# during warmup rather than on the first real request.
WARMUP_SECONDS = 12.0


class TTDiarizationRunner(BaseMetalDeviceRunner):
    """Speaker diarization (pyannote community-1) with both nets on device.

    The device itself comes from ``BaseMetalDeviceRunner``: the worker is handed
    a ``device_id`` by the Scheduler, ``set_device()`` opens the mesh described
    by ``settings.device_mesh_shape``, and ``close_device()`` releases it. Only
    the l1_small reservation is model-specific, which is what
    ``get_pipeline_device_params`` is for.
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.backend = None

    def get_pipeline_device_params(self):
        return {"l1_small_size": DIARIZATION_L1_SMALL_SIZE}

    def _model_path(self) -> str:
        """Repo id to load community-1 from.

        ``HF_MODEL`` is how the other runners let an operator redirect the repo
        (llama_runner, embedding_runner); it matters here because the canonical
        repo is gated, so an ungated mirror is the only way to start without a
        token.
        """
        return (
            os.environ.get("HF_MODEL")
            or self.settings.model_weights_path
            or SupportedModels.PYANNOTE_SPEAKER_DIARIZATION_COMMUNITY_1.value
        )

    @log_execution_time(
        "Diarization model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        """Build the pipeline on the device and compile its kernels.

        Every failure propagates. The model is served because the two nets run
        on the accelerator, so a runner that cannot reach it has nothing to
        offer; letting it report success would leave the Scheduler advertising a
        worker that silently is not the thing being asked for.
        """
        device = self.set_device()
        model_path = self._model_path()
        self.logger.info(
            f"Device {self.device_id}: Loading diarization pipeline from {model_path}"
        )
        self.backend = DiarizationBackend(
            model_path=model_path,
            device="cpu",
            nn_accelerator=make_tt_accelerator(device),
        )

        self.logger.info(f"Device {self.device_id}: Compiling kernels...")
        self.backend.diarize(
            _silence(self.settings.default_sample_rate), exclusive=True
        )
        self.logger.info(f"Device {self.device_id}: Diarization pipeline ready")
        return True

    @log_execution_time(
        "Run diarization inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[DiarizationRequest]):
        if self.backend is None:
            raise RuntimeError("Diarization pipeline not loaded. Call warmup() first.")
        if self.ttnn_device is None:
            raise RuntimeError("TTNN device not initialized")

        # pyannote's pipeline mutates instance state during __call__, so it is
        # not batchable; max_batch_size is 1 in the catalog and the Scheduler
        # hands over one request at a time.
        return [self._diarize(request) for request in requests]

    def _diarize(self, request: DiarizationRequest):
        import torch

        waveform = torch.from_numpy(request._audio_array).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return self.backend.diarize(
            {
                "waveform": waveform,
                "sample_rate": self.settings.default_sample_rate,
            },
            num_speakers=request.num_speakers,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
            exclusive=request.exclusive,
        )

    def is_request_batchable(self, request, batch=None):
        return False


def _silence(sample_rate: int) -> dict:
    """A short silent waveform, in the in-memory shape pyannote accepts.

    Handed as a tensor rather than a path so the pipeline never decodes a file:
    torchcodec cannot load against the torch pin this image ships.
    """
    import torch

    samples = int(WARMUP_SECONDS * sample_rate)
    return {
        "waveform": torch.zeros(1, samples, dtype=torch.float32),
        "sample_rate": int(sample_rate),
    }
