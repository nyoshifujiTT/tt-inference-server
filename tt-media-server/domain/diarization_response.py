# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import List, Optional

from pydantic import BaseModel


class DiarizationSegment(BaseModel):
    """A single speaker turn (start/end in seconds, speaker label)."""

    start: float
    end: float
    speaker: str


class DiarizationResponse(BaseModel):
    """Speaker diarization result.

    Field layout mirrors the pyannoteAI cloud diarization output so clients can
    switch base URLs without changing parsing code:
      - ``segments``: speaker turns (may include overlapped speech).
      - ``exclusiveDiarization``: non-overlapping turns (one speaker at a time),
        present when the request set ``exclusive=true``. community-1 provides
        this via ``output.exclusive_speaker_diarization``.

    Note: ``confidence`` / ``transcription`` fields of the pyannoteAI API are
    precision-2 (paid) only and are intentionally not emitted here.
    """

    segments: List[DiarizationSegment]
    exclusiveDiarization: Optional[List[DiarizationSegment]] = None

    def to_dict(self) -> dict:
        out = {
            "segments": [
                {"start": s.start, "end": s.end, "speaker": s.speaker}
                for s in self.segments
            ]
        }
        if self.exclusiveDiarization is not None:
            out["exclusiveDiarization"] = [
                {"start": s.start, "end": s.end, "speaker": s.speaker}
                for s in self.exclusiveDiarization
            ]
        return out
