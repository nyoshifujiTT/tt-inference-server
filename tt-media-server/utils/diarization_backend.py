# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""CPU backend for pyannote.audio 4.x speaker diarization (community-1).

Design notes
------------
- Uses ``pyannote.audio`` 4.x directly (``Pipeline.from_pretrained`` +
  ``pipeline(...)`` returning a ``DiarizeOutput`` with
  ``speaker_diarization`` / ``exclusive_speaker_diarization``). It does NOT go
  through whisperx, whose pinned pyannote 3.x cannot load community-1.
- The pyannote ``Pipeline`` is not thread-safe (it mutates instance state during
  ``__call__``), so a single shared pipeline is guarded by a lock and calls are
  serialized. This mirrors the reference gbase-asr server (``_pipeline`` singleton
  + ``DIAR_CONCURRENCY=1``).
- ``lightning>=2.6`` flips ``torch.load(weights_only=True)``; pyannote checkpoints
  contain non-tensor objects, so we restore the legacy default on load.
- ``torch`` / ``pyannote.audio`` are imported lazily so this module can be
  imported (and unit-tested) in environments without them installed.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional


def _install_torch_load_shim() -> None:
    """Restore pre-2.6 ``torch.load(weights_only=False)`` default for pyannote."""
    import torch

    if getattr(torch.load, "_diar_shim", False):
        return
    original = torch.load

    def patched(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original(*args, **kwargs)

    patched._diar_shim = True
    torch.load = patched


def annotation_to_segments(annotation) -> List[Dict]:
    """Convert a pyannote.core Annotation to a list of {start,end,speaker}."""
    segments: List[Dict] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "speaker": str(speaker),
            }
        )
    return segments


def build_pipeline_kwargs(
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> Dict[str, int]:
    """Assemble pyannote pipeline call kwargs from speaker-count hints.

    ``num_speakers`` takes precedence; otherwise min/max are passed when set.
    """
    kwargs: Dict[str, int] = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    else:
        if min_speakers:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers:
            kwargs["max_speakers"] = int(max_speakers)
    return kwargs


class DiarizationBackend:
    """Lazy, thread-safe wrapper around a pyannote.audio 4.x pipeline (CPU)."""

    def __init__(self, model_path: str, device: str = "cpu", nn_accelerator=None):
        self._model_path = model_path
        self._device = device
        self._pipeline = None
        self._lock = threading.Lock()
        # Optional hook (pipeline) -> None to offload the community-1 NNs
        # (segmentation PyanNet / embedding WeSpeaker) onto an accelerator such
        # as a Tenstorrent p150 (see tt_model_runners/diarization_nn_accelerator).
        # When None, everything runs on
        # the configured torch device (CPU).
        self._nn_accelerator = nn_accelerator

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is None:
                import torch
                from pyannote.audio import Pipeline

                _install_torch_load_shim()
                pipe = Pipeline.from_pretrained(self._model_path)
                pipe.to(torch.device(self._device))
                if self._nn_accelerator is not None:
                    self._nn_accelerator(pipe)
                self._pipeline = pipe
        return self._pipeline

    def diarize(
        self,
        audio,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        exclusive: bool = True,
    ) -> Dict[str, Optional[List[Dict]]]:
        """Run diarization and return {'segments', 'exclusiveDiarization'}.

        ``audio`` may be a filepath or an in-memory ``{"waveform","sample_rate"}``
        mapping accepted by pyannote. Calls are serialized (pipeline not
        thread-safe).
        """
        pipe = self._ensure_pipeline()
        kwargs = build_pipeline_kwargs(num_speakers, min_speakers, max_speakers)
        with self._lock:
            output = pipe(audio, **kwargs)
        segments = annotation_to_segments(output.speaker_diarization)
        result: Dict[str, Optional[List[Dict]]] = {"segments": segments}
        if exclusive:
            result["exclusiveDiarization"] = annotation_to_segments(
                output.exclusive_speaker_diarization
            )
        return result
