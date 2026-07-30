# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Speaker diarization service (CPU, pyannote.audio 4.x / community-1).

This is an audio-category service that is SEPARATE from AudioService (which does
ASR/whisper preprocessing). It runs pyannote diarization on CPU via
DiarizationBackend and returns pyannoteAI-shaped speaker turns only (no ASR).

It does not use the device Scheduler / model runner path: pyannote diarization
is a CPU pipeline, so __init__ is overridden to skip Scheduler + HF auto-download
and just construct the CPU backend. Concurrency safety (pyannote pipeline is not
thread-safe) is handled inside DiarizationBackend via a lock.
"""

import os
import tempfile

from config.constants import SupportedModels
from config.settings import settings
from domain.diarization_request import DiarizationRequest
from domain.diarization_response import DiarizationResponse, DiarizationSegment
from utils.decorators import log_execution_time
from utils.diarization_backend import DiarizationBackend
from utils.ffmpeg_utils import decode_to_wav
from utils.logger import TTLogger


class DiarizationService:
    """CPU speaker-diarization service (not a device/runner-backed BaseService)."""

    def __init__(self):
        self.logger = TTLogger()
        model_path = (
            settings.model_weights_path
            or settings.preprocessing_model_weights_path
            or SupportedModels.PYANNOTE_SPEAKER_DIARIZATION_COMMUNITY_1.value
        )
        self.logger.info(f"DiarizationService using model: {model_path}")
        self._backend = DiarizationBackend(model_path=model_path, device="cpu")

    @log_execution_time("Diarization request")
    async def process_request(
        self, request: DiarizationRequest
    ) -> DiarizationResponse:
        audio_bytes = request.file
        if isinstance(audio_bytes, str):
            import base64

            audio_bytes = base64.b64decode(audio_bytes)

        # Normalize any input to 16 kHz mono WAV (pyannote 4.x expects exact
        # sample-count crops; compressed inputs otherwise raise length errors).
        wav_bytes = decode_to_wav(audio_bytes, sample_rate=settings.default_sample_rate)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name
            result = self._backend.diarize(
                tmp_path,
                num_speakers=request.num_speakers,
                min_speakers=request.min_speakers,
                max_speakers=request.max_speakers,
                exclusive=request.exclusive,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        segments = [DiarizationSegment(**s) for s in result["segments"]]
        exclusive = None
        if result.get("exclusiveDiarization") is not None:
            exclusive = [
                DiarizationSegment(**s) for s in result["exclusiveDiarization"]
            ]
        return DiarizationResponse(segments=segments, exclusiveDiarization=exclusive)

    def start_workers(self):
        """No device workers to start (CPU, in-process backend).

        Part of the service lifecycle contract invoked by the app lifespan. The
        pyannote pipeline is lazy-loaded on first request (and warmed by the
        readiness check below), so there is nothing to spin up here.
        """
        return None

    def check_is_model_ready(self) -> dict:
        """Readiness for /health and /tt-liveness.

        The CPU backend has no device to probe; it is ready as soon as the
        service is constructed. Weights load lazily on first diarize call.
        """
        return {"model_ready": True, "runner_in_use": "diarization-cpu"}

    def stop_workers(self):
        """No background workers to stop (CPU backend is in-process)."""
        return None
