# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Composite ``model`` id parsing for speaker-diarized transcription.

The OpenAI audio API only carries a single ``model`` field, but a diarized
transcription needs both an ASR model and a diarization model. We encode both in
the one ``model`` string using a ``+`` separator, as an explicit *extension* of
the OpenAI ``model`` field (not part of the OpenAI spec):

    model="<asr_model>"                       -> ASR only
    model="<asr_model>+<diarization_model>"   -> diarized transcription

Rules (see worklog "model合成ID構文"):
- Separator is ``+``. Whitespace around parts is trimmed.
- Ordering is fixed: first part = ASR (generation) model, second = diarization.
- At most two parts; three or more is an error.
- Empty parts are an error.
- This is a *selection* over already-loaded/allowed combinations, NOT a request
  to dynamically load arbitrary models (matches the one-server-one-config model).

The parser is pure/stateless. Enforcement that the requested pair is actually
served is the caller's responsibility (e.g. validate against configured names).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

COMPOSITE_SEPARATOR = "+"


class CompositeModelIdError(ValueError):
    """Raised when a composite model id is malformed."""


@dataclass(frozen=True)
class ParsedModelId:
    asr_model: str
    diarization_model: Optional[str] = None

    @property
    def wants_diarization(self) -> bool:
        return self.diarization_model is not None

    def canonical(self) -> str:
        """Re-serialize to the canonical composite string."""
        if self.diarization_model is None:
            return self.asr_model
        return f"{self.asr_model}{COMPOSITE_SEPARATOR}{self.diarization_model}"


def parse_model_id(model: str) -> ParsedModelId:
    """Parse a (possibly composite) ``model`` id into ASR + optional diarization.

    Raises CompositeModelIdError on malformed input.
    """
    if model is None or not isinstance(model, str) or not model.strip():
        raise CompositeModelIdError("model must be a non-empty string")

    parts = [p.strip() for p in model.split(COMPOSITE_SEPARATOR)]
    if any(p == "" for p in parts):
        raise CompositeModelIdError(
            f"empty component in composite model id: {model!r}"
        )
    if len(parts) == 1:
        return ParsedModelId(asr_model=parts[0])
    if len(parts) == 2:
        return ParsedModelId(asr_model=parts[0], diarization_model=parts[1])
    raise CompositeModelIdError(
        f"composite model id supports at most 2 components (asr+diarization), "
        f"got {len(parts)}: {model!r}"
    )
