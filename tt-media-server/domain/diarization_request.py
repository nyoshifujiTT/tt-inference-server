# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import Optional, Union

import numpy as np
from domain.base_request import BaseRequest
from pydantic import PrivateAttr, field_validator


class DiarizationRequest(BaseRequest):
    """Speaker diarization request.

    Schema is aligned with the pyannoteAI cloud diarization API
    (POST https://api.pyannote.ai/v1/diarize) so that a client written against
    pyannoteAI can target this self-hosted endpoint by only changing the base
    URL. Unlike the media-server audio transcription request, diarization does
    NOT produce a transcript: the response is speaker turns only.

    Input is provided as an uploaded audio file (multipart) decoded by the
    router into raw bytes, or a base64-encoded audio string.
    """

    # Required: raw audio bytes (from multipart upload) or base64-encoded string.
    file: Union[str, bytes]

    # Optional speaker-count hints (same semantics as pyannote pipeline kwargs).
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    # When true, also include a non-overlapping (one speaker at a time) view.
    # community-1 exposes this via output.exclusive_speaker_diarization.
    exclusive: bool = True

    # Internal: decoded 16 kHz mono waveform, filled by pre_process.
    _audio_array: Optional[np.ndarray] = PrivateAttr(default=None)
    _duration: float = PrivateAttr(default=0.0)

    @field_validator("num_speakers", "min_speakers", "max_speakers", mode="before")
    @classmethod
    def _validate_speaker_counts(cls, v):
        if v is None or v == "":
            return None
        iv = int(v)
        if iv < 1:
            raise ValueError("speaker count must be >= 1")
        return iv

    @field_validator("exclusive", mode="before")
    @classmethod
    def _validate_exclusive(cls, v):
        if isinstance(v, bool):
            return v
        if v is None or v == "":
            return True
        return str(v).strip().lower() in ("1", "true", "yes", "on")
