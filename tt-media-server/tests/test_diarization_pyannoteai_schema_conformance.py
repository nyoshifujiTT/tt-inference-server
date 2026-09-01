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


def test_inline_base64_is_a_documented_extension_not_official_behaviour(
    official_schemas,
):
    """Accepting base64 in ``url`` is ours, not pyannoteAI's.

    The official field is a location -- "URL of the audio file to be processed"
    -- and nothing in the spec offers an inline form. This test reads that
    description off the live spec rather than trusting a comment, so if
    pyannoteAI ever does adopt an inline form the assertion fails and the
    "non-standard" wording gets revisited instead of quietly going stale.

    The extension is deliberate (see open_ai_api/diarization.py) and this test
    is what keeps it labelled: a reader must not conclude from the rest of this
    file that every input this server accepts is portable to the cloud service.
    """
    url_schema = official_schemas["DiarizeRequest"]["properties"]["url"]

    # A bare string with no pattern is what makes the extension expressible
    # without emitting a request the spec would reject. It is not permission.
    assert url_schema["type"] == "string"
    assert "pattern" not in url_schema

    description = url_schema.get("description", "").lower()
    assert "url" in description, (
        "the official url field no longer describes itself as a URL; recheck "
        f"whether inline audio is now supported upstream: {description!r}"
    )
    for inline_word in ("base64", "inline", "data:"):
        assert inline_word not in description, (
            f"the official spec now mentions {inline_word!r} in DiarizeRequest."
            "url; inline audio may no longer be a non-standard extension"
        )

    # The docstring has to keep saying so, since that is where a reader looks.
    from open_ai_api import diarization

    assert "NON-STANDARD EXTENSION" in diarization.__doc__
    assert "NON-STANDARD EXTENSION" in diarization._fetch_audio.__doc__


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
    app.include_router(diarization.async_router, prefix="/v1")
    app.dependency_overrides[service_resolver] = lambda: _NeverCalled()
    app.dependency_overrides[diarization.get_api_key] = lambda: "test"
    client = TestClient(app)

    rejected = [
        client.post(
            "/v1/diarize",
            files={"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")},
        ),
        client.post(
            "/v1/diarize",
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


@pytest.fixture(scope="module")
def official_paths() -> set:
    """Every path the official spec documents."""
    spec = _fetch_official_spec()
    paths = set(spec.get("paths") or {})
    assert paths, "pyannoteAI spec has no paths"
    return paths


def _our_served_paths() -> set:
    """Paths this service registers, as the OpenAPI document reports them.

    Built the way ``open_ai_api`` mounts them for MODEL_SERVICE=diarization, so
    the prefixes are the served ones rather than the router-local ones.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from resolver.service_resolver import service_resolver

    from open_ai_api import diarization, media

    app = FastAPI()
    app.include_router(diarization.async_router, prefix="/v1")
    app.include_router(media.router, prefix="/v1/media")
    app.dependency_overrides[service_resolver] = lambda: object()
    app.dependency_overrides[diarization.get_api_key] = lambda: "test"
    app.dependency_overrides[media.get_api_key] = lambda: "test"

    served = TestClient(app).get("/openapi.json").json()["paths"]
    return set(served)


def test_every_path_we_publish_is_one_the_official_spec_has(official_paths):
    """A path outside the official set breaks "switch base URL only".

    The rest of this file compares field names and status codes, so it never
    noticed that the service published /v1/audio/diarize, /audio/diarize and
    PUT /v1/media/input/{object_key} -- none of which exist upstream. Compare
    the paths too, and fail on anything invented.
    """
    ours = _our_served_paths()
    invented = ours - official_paths
    assert not invented, (
        f"paths we publish that the official spec does not document: "
        f"{sorted(invented)}; official set is {sorted(official_paths)}"
    )


def test_the_paths_we_publish_use_the_official_parameter_names(official_paths):
    """`{job_id}` and `{jobId}` are different paths to a spec differ."""
    ours = _our_served_paths()
    templated = {p for p in ours if "{" in p}
    for path in templated:
        assert path in official_paths, (
            f"{path} is templated but not spelled as the official spec spells it; "
            f"official templated paths are "
            f"{sorted(p for p in official_paths if '{' in p)}"
        )
