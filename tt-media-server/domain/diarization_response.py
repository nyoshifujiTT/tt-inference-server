# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import List, Optional

from pydantic import BaseModel


class DiarizationSegment(BaseModel):
    """A single speaker turn.

    Mirrors the pyannoteAI ``DiarizationSegment`` schema
    (https://docs.pyannote.ai/openapi.json): required fields ``speaker``,
    ``start``, ``end`` (seconds).
    """

    speaker: str
    start: float
    end: float


class DiarizationResponse(BaseModel):
    """Speaker diarization result.

    ``to_dict()`` emits exactly the pyannoteAI ``DiarizationJobOutput`` shape
    (https://docs.pyannote.ai/openapi.json) so a client can switch base URL
    only:
      - ``diarization``: speaker turns (required; may include overlapped speech).
      - ``exclusiveDiarization``: non-overlapping turns (one speaker at a time),
        present when the request set ``exclusive=true``. community-1 provides
        this via ``output.exclusive_speaker_diarization``.

    Note: ``confidence`` / ``wordLevelTranscription`` / ``turnLevelTranscription``
    fields of the pyannoteAI API are precision-2 (paid) only and are
    intentionally not emitted here. The ``diarization`` list is kept internally
    as ``segments`` for readability.
    """

    segments: List[DiarizationSegment]
    exclusiveDiarization: Optional[List[DiarizationSegment]] = None

    def to_dict(self) -> dict:
        out = {
            "diarization": [
                {"speaker": s.speaker, "start": s.start, "end": s.end}
                for s in self.segments
            ]
        }
        if self.exclusiveDiarization is not None:
            out["exclusiveDiarization"] = [
                {"speaker": s.speaker, "start": s.start, "end": s.end}
                for s in self.exclusiveDiarization
            ]
        return out
