# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Build the pyannoteAI-style ``warning`` string for a diarization result.

The pyannoteAI cloud API adds a ``warning`` to the output when it cannot honor
the requested speaker-count constraints (see the ``numSpeakers`` description at
https://docs.pyannote.ai/openapi.json). community-1's pipeline output has no
warning field, so we compute the equivalent here from the requested constraints
vs. the number of distinct speakers actually detected.
"""

from typing import Iterable, Optional


def count_distinct_speakers(segments: Iterable[dict]) -> int:
    """Number of distinct speaker labels across diarization segments."""
    return len({s["speaker"] for s in segments})


def build_speaker_count_warning(
    detected: int,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Optional[str]:
    """Return a warning string if the detected speaker count violates the request.

    Mirrors pyannoteAI: an exact ``numSpeakers`` request that is not met, or a
    detected count outside a requested ``minSpeakers``/``maxSpeakers`` range,
    yields a warning. Returns None when the request was honored (or unset).
    """
    if num_speakers is not None and detected != num_speakers:
        return (
            f"requested numSpeakers={num_speakers} could not be honored; "
            f"detected {detected} speakers"
        )
    if min_speakers is not None and detected < min_speakers:
        return (
            f"detected {detected} speakers, fewer than requested "
            f"minSpeakers={min_speakers}"
        )
    if max_speakers is not None and detected > max_speakers:
        return (
            f"detected {detected} speakers, more than requested "
            f"maxSpeakers={max_speakers}"
        )
    return None
