# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Coordinator for diarized transcription: diarization -> slice -> per-turn ASR.

This is the reusable glue between:
  - the diarization backend (utils.diarization_backend, homework 1),
  - the composite model-id parser (utils.composite_model_id, homework 6),
  - the diarized_json assembler (utils.diarized_transcription, homework 4).

The ASR step is injected as ``transcribe_slice(waveform_slice, sample_rate) -> str``
so the coordinator stays device-independent and unit-testable; the audio route
supplies a real closure over the ASR model runner.

Flow:
  1. diarize the whole waveform -> exclusive (one-speaker-at-a-time) turns
  2. for each turn, slice the waveform [start,end) and transcribe it
  3. assemble OpenAI diarized_json (segments[].speaker + text)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from utils.diarized_transcription import build_diarized_json


def slice_waveform(waveform, sample_rate: int, start_s: float, end_s: float):
    """Return waveform[start:end] by sample index (bounds-clamped)."""
    n = len(waveform)
    i0 = max(0, int(round(start_s * sample_rate)))
    i1 = min(n, int(round(end_s * sample_rate)))
    if i1 <= i0:
        return waveform[0:0]
    return waveform[i0:i1]


class DiarizedAsrCoordinator:
    """Combine a diarization backend with a per-slice ASR callable."""

    def __init__(
        self,
        diarize_fn: Callable[..., Dict[str, Optional[List[Dict]]]],
        transcribe_slice: Callable[[object, int], str],
        sample_rate: int = 16000,
    ):
        # diarize_fn(audio_path_or_obj, num_speakers=..., exclusive=True) -> {segments, exclusiveDiarization}
        self._diarize_fn = diarize_fn
        self._transcribe_slice = transcribe_slice
        self._sample_rate = sample_rate

    def run(
        self,
        diarize_input,
        waveform,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> Dict:
        diar = self._diarize_fn(
            diarize_input,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            exclusive=True,
        )
        # Prefer non-overlapping turns so each ASR slice has a single speaker.
        turns = diar.get("exclusiveDiarization") or diar.get("segments") or []

        if duration is None and waveform is not None:
            duration = len(waveform) / float(self._sample_rate)

        def _asr(start_s: float, end_s: float) -> str:
            chunk = slice_waveform(waveform, self._sample_rate, start_s, end_s)
            if len(chunk) == 0:
                return ""
            return self._transcribe_slice(chunk, self._sample_rate)

        return build_diarized_json(turns, _asr, duration=duration)
