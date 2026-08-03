# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Conformance: our diarization API matches the live pyannoteAI OpenAPI spec.

This test fetches the *official* pyannoteAI OpenAPI document fresh on every run
and parses it raw (no vendored copy, no hand-transcribed expectations), then
asserts that our implementation matches it exactly:

  - request: our diarize form fields cover the pyannoteAI ``DiarizeRequest``
    speaker-count controls with the exact camelCase names, and ``exclusive``;
  - response: ``DiarizationResponse.to_dict()`` emits exactly the pyannoteAI
    ``DiarizationJobOutput`` keys we support (``diarization`` +
    ``exclusiveDiarization``) with the required ``DiarizationSegment`` fields
    (``speaker``/``start``/``end``) in schema field order.

The official spec is a hard dependency of this test: if it cannot be fetched or
parsed, the test FAILS (it does not skip). A green run therefore always means we
were actually checked against the live pyannoteAI schema, never against a stale
or absent copy. Point ``PYANNOTEAI_OPENAPI_URL`` elsewhere only to pin a
known-good mirror in a deliberately offline environment.
"""

import inspect
import json
import os
import urllib.request

import pytest

from domain.diarization_response import DiarizationResponse, DiarizationSegment

PYANNOTEAI_OPENAPI_URL = os.environ.get(
    "PYANNOTEAI_OPENAPI_URL", "https://docs.pyannote.ai/openapi.json"
)


def _fetch_official_spec() -> dict:
    """Fetch + raw-parse the live pyannoteAI OpenAPI document.

    Any failure (unreachable host, non-200 status, non-JSON body) is a hard
    error: we must never certify conformance without actually reading the
    official spec.
    """
    with urllib.request.urlopen(PYANNOTEAI_OPENAPI_URL, timeout=20) as resp:
        status = getattr(resp, "status", 200)
        assert status == 200, (
            f"pyannoteAI OpenAPI {PYANNOTEAI_OPENAPI_URL} returned HTTP {status}"
        )
        raw = resp.read()
    return json.loads(raw)


@pytest.fixture(scope="module")
def official_schemas() -> dict:
    spec = _fetch_official_spec()
    schemas = spec.get("components", {}).get("schemas", {})
    for name in ("DiarizeRequest", "DiarizationJobOutput", "DiarizationSegment"):
        assert name in schemas, f"pyannoteAI spec is missing schema {name!r}"
    return schemas


def test_segment_fields_match_official(official_schemas):
    seg = official_schemas["DiarizationSegment"]
    official_required = set(seg.get("required", []))
    assert official_required == {"speaker", "start", "end"}

    ours = list(DiarizationSegment.model_fields.keys())
    assert set(ours) == official_required
    official_order = [f for f in seg["properties"].keys() if f in official_required]
    assert ours == official_order


def test_response_output_keys_match_official(official_schemas):
    out_schema = official_schemas["DiarizationJobOutput"]
    official_props = out_schema["properties"]
    assert "diarization" in out_schema.get("required", [])

    resp = DiarizationResponse(
        segments=[DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0)],
        exclusiveDiarization=[
            DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0)
        ],
    )
    body = resp.to_dict()

    for key in body:
        assert key in official_props, f"{key!r} is not a pyannoteAI output field"
    assert "diarization" in body
    assert "exclusiveDiarization" in body

    official_seg_required = set(
        official_schemas["DiarizationSegment"].get("required", [])
    )
    for key in ("diarization", "exclusiveDiarization"):
        for seg in body[key]:
            assert set(seg.keys()) == official_seg_required


def test_request_speaker_controls_match_official(official_schemas):
    from open_ai_api import diarization

    req_props = set(official_schemas["DiarizeRequest"]["properties"].keys())
    expected = {"numSpeakers", "minSpeakers", "maxSpeakers", "exclusive"}
    assert expected <= req_props, f"official DiarizeRequest dropped {expected - req_props}"

    params = inspect.signature(diarization.parse_diarization_request).parameters
    aliases = set()
    for p in params.values():
        alias = getattr(p.default, "alias", None)
        aliases.add(alias if alias else p.name)
    for name in expected:
        assert name in aliases, f"diarize form does not accept official field {name!r}"
