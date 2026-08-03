# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Diarized-transcription orchestration (diarization + per-turn ASR).

Given speaker turns (from the diarization backend) and a callable that
transcribes a single audio slice, assemble the OpenAI-compatible
``diarized_json`` response:

    {
      "task": "transcribe",
      "text": "<full text>",
      "duration": <seconds>,
      "segments": [
        {"id": 0, "speaker": "SPEAKER_00", "start": 0.2, "end": 1.6, "text": "..."},
        ...
      ]
    }

The ASR step is injected as ``transcribe_slice(start_s, end_s) -> str`` so this
module is pure/testable and independent of any device/model runner. Ordering
follows the OpenAI ``diarized_json`` schema (segment-level, one speaker per
segment), which matches community-1's exclusive diarization.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional


def build_diarized_json(
    turns: List[Dict],
    transcribe_slice: Callable[[float, float], str],
    duration: Optional[float] = None,
    drop_empty: bool = True,
) -> Dict:
    """Assemble a diarized_json result from diarization turns + an ASR callable.

    Args:
      turns: list of {"start","end","speaker"} in seconds (already
        post-processed / non-overlapping). Order is preserved by start time.
      transcribe_slice: called per turn with (start_s, end_s), returns text.
      duration: total audio duration in seconds (for the response); if None,
        derived from the max turn end.
      drop_empty: skip segments whose transcription is blank.
    """
    ordered = sorted(turns, key=lambda t: (float(t["start"]), float(t["end"])))
    segments: List[Dict] = []
    for turn in ordered:
        start = float(turn["start"])
        end = float(turn["end"])
        speaker = turn["speaker"]
        text = (transcribe_slice(start, end) or "").strip()
        if drop_empty and not text:
            continue
        segments.append(
            {
                "id": len(segments),
                "speaker": speaker,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            }
        )

    if duration is None:
        duration = max((float(t["end"]) for t in ordered), default=0.0)

    full_text = " ".join(s["text"] for s in segments).strip()
    return {
        "task": "transcribe",
        "text": full_text,
        "duration": round(float(duration), 3),
        "segments": segments,
    }
