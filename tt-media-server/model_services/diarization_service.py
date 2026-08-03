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
from utils.diarization_warnings import (
    build_speaker_count_warning,
    count_distinct_speakers,
)
from utils.diarized_asr_coordinator import DiarizedAsrCoordinator
from utils.asr_http_client import encode_wav_pcm16, transcribe_wav_bytes
from utils.composite_model_id import parse_model_id
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
        nn_accelerator = self._maybe_build_tt_accelerator()
        self._backend = DiarizationBackend(
            model_path=model_path, device="cpu", nn_accelerator=nn_accelerator
        )

    def _maybe_build_tt_accelerator(self):
        """Build a Tenstorrent NN accelerator hook when DIARIZATION_TT_DEVICE_ID is set.

        Offloads community-1's segmentation (PyanNet) and embedding (WeSpeaker)
        NNs onto a p150 via ttnn (see tt_port/tt_nn_accelerator). When the env var
        is unset, returns None and the service runs pure-CPU.
        """
        import os

        dev_id = os.getenv("DIARIZATION_TT_DEVICE_ID")
        if dev_id is None or dev_id == "":
            return None
        try:
            import ttnn
            import sys

            sys.path.insert(
                0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tt_port")
            )
            from tt_nn_accelerator import make_tt_accelerator

            l1 = int(os.getenv("DIARIZATION_TT_L1_SMALL", "32768"))
            device = ttnn.open_device(device_id=int(dev_id), l1_small_size=l1)
            self._tt_device = device
            self.logger.info(f"DiarizationService: TT NN acceleration on device {dev_id}")
            return make_tt_accelerator(device)
        except Exception as e:  # noqa: BLE001 - fall back to CPU on any TT error
            self.logger.warning(
                f"TT NN acceleration requested but unavailable ({e}); using CPU"
            )
            return None

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
        warning = build_speaker_count_warning(
            count_distinct_speakers(result["segments"]),
            num_speakers=request.num_speakers,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )
        return DiarizationResponse(
            segments=segments, exclusiveDiarization=exclusive, warning=warning
        )

    def start_workers(self):
        """Warm up the pipeline (and compile ttnn kernels) at service start.

        Part of the service lifecycle contract invoked by the app lifespan.
        The pyannote pipeline weights are lazy-loaded and, when TT acceleration
        is enabled, every ttnn kernel is JIT/auto-shard compiled on first use.
        Doing that on the first real request would make it slow and emit
        one-off device log noise. So we run a single short dummy diarization
        here to pay the load + compile cost up front; failures are non-fatal
        (the first real request will simply compile lazily as before).
        """
        try:
            self.warmup()
        except Exception as e:  # noqa: BLE001 - warmup is best-effort
            self.logger.warning(f"Diarization warmup skipped ({e})")
        return None

    def warmup(self, seconds: float = 12.0) -> None:
        """Run one dummy diarization to load weights and compile ttnn kernels.

        Uses a short synthetic 16 kHz mono WAV. seconds is chosen long
        to exercise the real segmentation window (10 s) so its kernels compile
        during warmup rather than on the first user request.
        """
        import io
        import wave
        import numpy as np

        sr = int(settings.default_sample_rate)
        n = int(seconds * sr)
        # low-amplitude noise so VAD/segmentation produce a non-trivial graph
        rng = np.random.RandomState(0)
        pcm = (rng.randn(n) * 0.02 * 32768.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fobj:
                fobj.write(buf.getvalue())
                tmp_path = fobj.name
            self.logger.info("DiarizationService: warming up pipeline...")
            self._backend.diarize(tmp_path, exclusive=True)
            self.logger.info("DiarizationService: warmup complete")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def check_is_model_ready(self) -> dict:
        """Readiness for /health and /tt-liveness.

        The CPU backend has no device to probe; it is ready as soon as the
        service is constructed. Weights load lazily on first diarize call.
        """
        return {"model_ready": True, "runner_in_use": "diarization-cpu"}

    def _wav_bytes_to_waveform(self, wav_bytes):
        """Decode 16-bit PCM mono WAV bytes to a float32 numpy waveform."""
        import wave

        import numpy as np

        with wave.open(__import__("io").BytesIO(wav_bytes), "rb") as w:
            n = w.getnframes()
            raw = w.readframes(n)
        arr = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
        return arr

    @log_execution_time("Diarized transcription request")
    async def diarized_transcription(
        self, request: DiarizationRequest, model: str, language=None, prompt=None
    ) -> dict:
        """Diarize + per-turn ASR -> OpenAI diarized_json.

        ``model`` is the composite id "<asr>+<diarization>"; the ASR part is sent
        to settings.asr_url per turn. Requires settings.asr_url to be configured.
        """
        parsed = parse_model_id(model)
        if not parsed.wants_diarization:
            raise ValueError(
                "diarized_json requires a composite model id '<asr>+<diarization>'"
            )
        if not settings.asr_url:
            raise ValueError(
                "diarized transcription requires ASR_URL (settings.asr_url) to be set"
            )

        audio_bytes = request.file
        if isinstance(audio_bytes, str):
            import base64

            audio_bytes = base64.b64decode(audio_bytes)
        wav_bytes = decode_to_wav(
            audio_bytes, sample_rate=settings.default_sample_rate
        )
        waveform = self._wav_bytes_to_waveform(wav_bytes)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name

            def transcribe_slice(chunk, sr):
                wb = encode_wav_pcm16(chunk, sr)
                return transcribe_wav_bytes(
                    settings.asr_url,
                    parsed.asr_model,
                    wb,
                    language=language,
                    prompt=prompt,
                    timeout=settings.asr_timeout_s,
                )

            coordinator = DiarizedAsrCoordinator(
                diarize_fn=self._backend.diarize,
                transcribe_slice=transcribe_slice,
                sample_rate=settings.default_sample_rate,
            )
            return coordinator.run(
                tmp_path,
                waveform,
                num_speakers=request.num_speakers,
                min_speakers=request.min_speakers,
                max_speakers=request.max_speakers,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def stop_workers(self):
        """No background workers to stop (CPU backend is in-process)."""
        return None
