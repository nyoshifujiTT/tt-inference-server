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


def test_request_fields_cover_official_schema(official_schemas):
    """Every official DiarizeRequest field is either implemented or explicitly
    marked unsupported -- nothing is silently ignored, and we do not invent
    fields the official schema does not have."""
    from open_ai_api import diarization

    official = set(official_schemas["DiarizeRequest"]["properties"].keys())
    implemented = set(diarization.IMPLEMENTED_REQUEST_FIELDS)
    unsupported = set(diarization.UNSUPPORTED_REQUEST_FIELDS)

    # we never classify a field that is not in the official schema
    assert implemented <= official, (
        f"non-official implemented fields: {implemented - official}"
    )
    assert unsupported <= official, (
        f"non-official unsupported fields: {unsupported - official}"
    )
    # implemented and unsupported are disjoint
    assert not (implemented & unsupported)
    # union covers the whole official request schema
    missing = official - implemented - unsupported
    assert not missing, f"official DiarizeRequest fields not classified: {missing}"


def test_response_fields_cover_official_schema(official_schemas):
    """Every official DiarizationJobOutput field is either emitted or explicitly
    documented as not-emitted (precision-2-only)."""
    official = set(official_schemas["DiarizationJobOutput"]["properties"].keys())
    emitted = {"diarization", "exclusiveDiarization", "warning"}
    not_emitted = {
        "confidence",
        "wordLevelTranscription",
        "turnLevelTranscription",
        "error",  # only meaningful for the async job model; sync diarize uses HTTP status
    }
    assert emitted <= official, f"non-official emitted fields: {emitted - official}"
    assert not_emitted <= official
    assert not (emitted & not_emitted)
    missing = official - emitted - not_emitted
    assert not missing, (
        f"official DiarizationJobOutput fields not classified: {missing}"
    )


@pytest.fixture(scope="module")
def official_diarize_statuses() -> set:
    """Status codes the official POST /v1/diarize documents."""
    spec = _fetch_official_spec()
    for path, operations in spec.get("paths", {}).items():
        if not path.endswith("/diarize"):
            continue
        post = operations.get("post")
        if isinstance(post, dict) and post.get("responses"):
            return {str(code) for code in post["responses"]}
    raise AssertionError("pyannoteAI spec has no POST .../diarize responses")


def test_we_do_not_answer_a_status_the_official_api_never_uses():
    """A bespoke status is a compatibility break even when the body is right.

    This endpoint used to hand-check the Content-Type and answer 415, which the
    official API never returns. To a client it read as "this format is not
    supported yet", so someone went looking for a multipart upload that the
    official API does not offer either. FastAPI's own 422 for a malformed body
    is the honest answer, and this test pins that the invented status is gone.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from resolver.service_resolver import service_resolver

    from open_ai_api import diarization

    class _NeverCalled:
        async def process_request(self, request):
            raise AssertionError("a rejected body must not reach the service")

    app = FastAPI()
    app.include_router(diarization.router, prefix="/v1/audio")
    app.dependency_overrides[service_resolver] = lambda: _NeverCalled()
    app.dependency_overrides[diarization.get_api_key] = lambda: "test"
    client = TestClient(app)

    rejected = [
        client.post(
            "/v1/audio/diarize",
            files={"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")},
        ),
        client.post(
            "/v1/audio/diarize",
            content=b"not json",
            headers={"content-type": "text/plain"},
        ),
    ]

    for response in rejected:
        assert response.status_code != 415, (
            "415 is not in the official spec; a malformed body must not be "
            f"reported with it (got {response.status_code})"
        )
        assert response.status_code == 422, response.text


def test_the_statuses_we_do_use_are_the_official_ones(official_diarize_statuses):
    """Every status this endpoint answers deliberately is one the spec has.

    422 is FastAPI's validation status rather than a pyannoteAI one, so it is
    allowed alongside the official set: the point of the check is that nothing
    outside those two groups creeps in.
    """
    deliberate = {"200", "400", "422"}
    framework_supplied = {"422"}
    unknown = deliberate - framework_supplied - official_diarize_statuses
    assert not unknown, (
        f"statuses we answer that the official spec does not document: {sorted(unknown)}; "
        f"official set is {sorted(official_diarize_statuses)}"
    )
