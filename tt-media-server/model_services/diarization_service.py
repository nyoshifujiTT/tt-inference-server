# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Speaker diarization service (pyannote.audio 4.x / community-1).

An audio-category service, separate from AudioService (which does ASR/whisper
preprocessing): diarization returns pyannoteAI-shaped speaker turns only, with
no transcript.

Like every other service here it is a thin BaseService: request decoding lives
in ``pre_process``, response shaping in ``post_process``, and everything about
the device -- opening it, warming it, reporting its health, restarting a dead
worker -- belongs to the Scheduler and to TTDiarizationRunner. The pyannote
pipeline is not thread-safe, which the catalog expresses as max_batch_size 1
and the runner as ``is_request_batchable() -> False``, rather than as a lock
around a pipeline this process owns.
"""

from config.settings import settings
from domain.diarization_request import DiarizationRequest
from domain.diarization_response import DiarizationResponse, DiarizationSegment
from model_services.base_service import BaseService
from telemetry.telemetry_client import TelemetryEvent
from utils.decorators import log_execution_time
from utils.diarization_warnings import (
    build_speaker_count_warning,
    count_distinct_speakers,
)
from utils.diarized_asr_coordinator import DiarizedAsrCoordinator
from utils.asr_http_client import encode_wav_pcm16, transcribe_wav_bytes
from utils.composite_model_id import parse_model_id
from utils.ffmpeg_utils import decode_to_wav


def _wav_bytes_to_waveform(wav_bytes: bytes) -> dict:
    """Decode in-memory WAV into pyannote's ``{"waveform", "sample_rate"}`` input.

    Handing pyannote a path makes it decode the file itself through torchcodec,
    whose wheels are built per torch release; the image pins torch to the version
    tt-vllm-plugin requires, so the installed torchcodec fails to load its
    extension and every request errors with "torchcodec is not available".
    Decoding here with the standard library sidesteps that dependency entirely --
    the audio is already normalized 16 kHz mono PCM at this point.
    """
    import io
    import wave

    import numpy as np
    import torch

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        channels = w.getnchannels()
        sample_rate = w.getframerate()
        frames = w.readframes(w.getnframes())

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    samples = samples.reshape(-1, channels).T  # (channel, time)
    return {"waveform": torch.from_numpy(samples.copy()), "sample_rate": sample_rate}


class DiarizationService(BaseService):
    """Speaker diarization over the standard Scheduler + device-runner path."""

    def __init__(self):
        super().__init__()

    @log_execution_time(
        "Diarization preprocessing", TelemetryEvent.PRE_PROCESSING, None
    )
    async def pre_process(self, request: DiarizationRequest) -> DiarizationRequest:
        """Decode the submitted audio to the waveform the runner will diarize.

        Normalizing to 16 kHz mono here rather than in the runner keeps ffmpeg
        off the device worker, and pyannote 4.x wants exact sample-count crops:
        a compressed input handed straight through raises length errors.
        """
        audio_bytes = request.file
        if isinstance(audio_bytes, str):
            import base64

            audio_bytes = base64.b64decode(audio_bytes)

        wav_bytes = decode_to_wav(audio_bytes, sample_rate=settings.default_sample_rate)
        request._audio_array = self._wav_bytes_to_samples(wav_bytes)
        return request

    async def post_process(self, result, input_request=None) -> DiarizationResponse:
        """Shape the runner's turns into the pyannoteAI response."""
        segments = [DiarizationSegment(**s) for s in result["segments"]]
        exclusive = None
        if result.get("exclusiveDiarization") is not None:
            exclusive = [
                DiarizationSegment(**s) for s in result["exclusiveDiarization"]
            ]
        warning = None
        if input_request is not None:
            warning = build_speaker_count_warning(
                count_distinct_speakers(result["segments"]),
                num_speakers=input_request.num_speakers,
                min_speakers=input_request.min_speakers,
                max_speakers=input_request.max_speakers,
            )
        return DiarizationResponse(
            segments=segments, exclusiveDiarization=exclusive, warning=warning
        )

    def _wav_bytes_to_samples(self, wav_bytes):
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

        request = await self.pre_process(request)
        samples = request._audio_array

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

        # Diarize through the Scheduler, so the turns come from the same device
        # worker the plain endpoint uses rather than from a second pipeline
        # living in this process. The await has to happen here because the
        # coordinator's slicing loop is synchronous.
        request.exclusive = True
        turns = await self.process(request)

        coordinator = DiarizedAsrCoordinator(
            diarize_fn=lambda *_args, **_kwargs: turns,
            transcribe_slice=transcribe_slice,
            sample_rate=settings.default_sample_rate,
        )
        return coordinator.run(
            None,
            samples,
            num_speakers=request.num_speakers,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )
