# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Speaker-diarization-only endpoint.

Schema aligned with the pyannoteAI cloud diarization API so a client can switch
base URL only. Unlike /v1/audio/transcriptions this returns speaker turns only
(no transcript). The /diarize input is a pyannoteAI-style JSON body with a
``url`` (http(s):// or media://); see https://docs.pyannote.ai/openapi.json
(DiarizeRequest).

NON-STANDARD EXTENSION: ``url`` additionally accepts inline base64 audio, which
the official API does NOT support -- it documents the field as "URL of the audio
file to be processed" and accepts only a fetchable location. Code written
against this extension will not work against pyannoteAI's cloud service. It
exists so a deployment with no object storage the server can reach still has a
way to send audio; see ``_fetch_audio``.
"""

import re

from domain.diarization_request import DiarizationRequest
from fastapi import APIRouter, Body, Depends, HTTPException, Security
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

# pyannoteAI DiarizeRequest fields this server implements (accepted in the JSON
# body) vs. deliberately-unsupported ones. Kept explicit so the schema
# conformance test can assert the union covers the whole official schema and
# nothing is silently ignored. See https://docs.pyannote.ai/openapi.json.
IMPLEMENTED_REQUEST_FIELDS = frozenset(
    {
        "url",
        "model",
        "numSpeakers",
        "minSpeakers",
        "maxSpeakers",
        "exclusive",
        # async job model (POST /v1/diarize + GET /v1/jobs/{id})
        "webhook",
        "webhookStatusOnly",
    }
)
# precision-2-only options: accepted by the parser only to reject them with 400.
UNSUPPORTED_REQUEST_FIELDS = frozenset(
    {
        "confidence",
        "turnLevelConfidence",
        "transcription",
        "transcriptionConfig",
    }
)


# A leading "<scheme>://" is how a caller signals "go fetch this", so a value
# shaped that way is never read as base64 -- ':' and '/' are outside the base64
# alphabet, so no real payload can collide with the test.
_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def _looks_like_a_url(value: str) -> bool:
    return bool(_URL_SCHEME_RE.match(value))


def _base64_len_for(max_bytes: int) -> int:
    """Longest base64 text that can decode to ``max_bytes`` bytes.

    Checked before decoding so an oversize payload is refused without first
    materialising its decoded form.
    """
    return ((max_bytes + 2) // 3) * 4 + 4


def _validate_served_model(model):
    """Reject diarization models this server does not serve.

    The pyannoteAI ``DiarizeRequest.model`` enum is ``community-1`` /
    ``precision-2`` (https://docs.pyannote.ai/openapi.json). This self-hosted
    server serves ``community-1`` only; anything else (notably the paid
    ``precision-2``) is rejected with HTTP 400.
    """
    from config.constants import PyannoteAiDiarizationModel, SERVED_DIARIZATION_MODEL

    if model is None or model == "":
        return
    served = SERVED_DIARIZATION_MODEL.value
    valid = {m.value for m in PyannoteAiDiarizationModel}
    if model not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"unknown diarization model {model!r}; valid values are {sorted(valid)}",
        )
    if model != served:
        raise HTTPException(
            status_code=400,
            detail=(
                f"diarization model {model!r} is not served by this server; "
                f"only {served!r} is available"
            ),
        )


def _reject_precision2_only_options(**options) -> None:
    """Reject pyannoteAI options that only the paid precision-2 model supports.

    ``confidence`` / ``turnLevelConfidence`` / ``transcription`` /
    ``transcriptionConfig`` are precision-2-only in the pyannoteAI cloud API
    (https://docs.pyannote.ai/openapi.json) and cannot be produced by
    community-1. If a client requests any of them, fail with HTTP 400 rather
    than silently ignoring the flag.
    """
    requested = [
        name for name, value in options.items() if value not in (None, False, "")
    ]
    if requested:
        raise HTTPException(
            status_code=400,
            detail=(
                "the following pyannoteAI options require the paid precision-2 "
                f"model and are not supported by this community-1 server: "
                f"{sorted(requested)}"
            ),
        )


async def _read_json_body(body: dict = Body(...)) -> dict:
    """The pyannoteAI DiarizeRequest body.

    Declared as a body parameter rather than sniffed off the raw request: a
    request that is not a JSON object then gets FastAPI's own 422, which is
    what the official API documents (400/401/429 -- never 415). Hand-checking
    the Content-Type here used to answer 415 instead, an invented status that
    read like "this format is not supported yet" and sent people looking for a
    multipart upload the official API does not have either.
    """
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="request body must be a JSON object"
        )
    return body


async def _fetch_audio(url: str) -> bytes:
    """Fetch the bytes a pyannoteAI ``DiarizeRequest.url`` points at.

    ``media://`` keys name an object in the configured storage service: they
    are signed into a GET url and then fetched down the same path as any other
    url, so there is one place that reads from the network. Everything
    else is either an http(s) URL or inline base64. http(s) goes through the
    server's hardened
    ``media_downloader`` -- the same path ``open_ai_api/video.py`` uses for
    presigned image URLs. That downloader is where the SSRF guard lives: a
    required hostname allowlist re-checked on every redirect hop, one deadline
    covering all hops, a streamed body checked against ``media_url_max_bytes``,
    and query strings kept out of logs because presigned URLs carry credentials
    there. A second fetch helper next to it would be a second hole to keep
    closed, which is exactly how the reference servers reintroduced SSRF.

    Inline base64 is a NON-STANDARD EXTENSION -- the official API does not
    accept it. The official schema describes ``url`` as "URL of the audio file
    to be processed" and types it as a bare ``{"type": "string"}``; the absence
    of a pattern is what makes the extension *possible* without emitting a
    request the spec would reject, but it does not make base64 *supported*
    upstream. A client relying on it is no longer "switch base URL only": point
    it at pyannoteAI's cloud service and the request fails.

    It exists because the official request carries audio in exactly one field
    and that field is a location, so a deployment with no object storage the
    server can reach otherwise has no way to send audio at all. It follows the
    same "URL or base64 in one field" shape
    ``domain/video_i2v_generate_request.py`` already uses for images, and costs
    a third in wire size, so it is the fallback and not the recommendation.

    Statuses follow video.py so one taxonomy covers every URL-valued field:
    policy violation 400, over the cap 413, origin/network/deadline 422.
    """
    import base64
    import binascii

    from utils import media_downloader
    from utils.media_downloader import (
        MediaDownloadFetchError,
        MediaDownloadPolicyError,
        MediaDownloadTooLargeError,
        download_media_url,
        is_media_url,
    )
    from utils.media_object_storage import (
        MEDIA_SCHEME,
        MediaStorageError,
        MediaStorageNotConfigured,
        presigned_get_url,
    )

    if url.startswith(MEDIA_SCHEME):
        try:
            url = presigned_get_url(url)
        except MediaStorageError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except MediaStorageNotConfigured as e:
            raise HTTPException(status_code=501, detail=str(e))

    if is_media_url(url):
        try:
            return await download_media_url(url)
        except MediaDownloadPolicyError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except MediaDownloadTooLargeError as e:
            raise HTTPException(status_code=413, detail=str(e))
        except MediaDownloadFetchError as e:
            raise HTTPException(status_code=422, detail=str(e))

    if _looks_like_a_url(url):
        # A scheme we do not serve. Saying so beats letting it fall through to
        # the base64 decoder and reporting it as malformed base64.
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported audio url scheme: {url!r}; use http(s):// or "
                "media://, or this server's non-standard extension of inline "
                "base64 audio in the same field"
            ),
        )

    # The same cap the downloader enforces, read off the downloader's own
    # settings object rather than a fresh import: how the bytes arrived should
    # not change how many of them this server is willing to hold, and two
    # imports could not drift apart.
    max_bytes = media_downloader.settings.media_url_max_bytes
    if len(url) > _base64_len_for(max_bytes):
        raise HTTPException(
            status_code=413,
            detail=(
                f"inline base64 audio (a non-standard extension) is over the "
                f"{max_bytes}-byte cap for this server; upload it and pass a "
                "url instead"
            ),
        )
    try:
        audio_bytes = base64.b64decode(url, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=400,
            detail=(
                "'url' is neither an http(s):// url, a media:// key, nor valid "
                "base64 audio (inline base64 is a non-standard extension of "
                "this server; the official API takes a url only)"
            ),
        )
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="'url' decoded to no audio")
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"inline base64 audio decodes to {len(audio_bytes)} bytes, over "
                f"the {max_bytes}-byte cap for this server"
            ),
        )
    return audio_bytes


async def _build_request_from_body(body: dict) -> DiarizationRequest:
    """Validate a pyannoteAI DiarizeRequest body and resolve it to a request.

    Validates ``model`` and rejects precision-2-only options, requires ``url``
    (http(s):// or media://, or inline base64 as a non-standard extension of
    this server -- the official API takes a url only), fetches the audio, and
    builds the internal
    DiarizationRequest. See https://docs.pyannote.ai/openapi.json.
    """
    _validate_served_model(body.get("model"))
    _reject_precision2_only_options(
        confidence=body.get("confidence"),
        turnLevelConfidence=body.get("turnLevelConfidence"),
        transcription=body.get("transcription"),
        transcriptionConfig=body.get("transcriptionConfig"),
    )

    url = body.get("url")
    if not url:
        raise HTTPException(
            status_code=400,
            detail=(
                "'url' is required: an http(s):// url or a media:// key, or "
                "inline base64 audio (a non-standard extension of this server)"
            ),
        )
    audio_bytes = await _fetch_audio(url)

    exclusive = body.get("exclusive")
    return DiarizationRequest(
        file=audio_bytes,
        num_speakers=body.get("numSpeakers"),
        min_speakers=body.get("minSpeakers"),
        max_speakers=body.get("maxSpeakers"),
        exclusive=True if exclusive is None else exclusive,
    )


async def parse_diarization_request(
    body: dict = Depends(_read_json_body),
) -> DiarizationRequest:
    """Parse a pyannoteAI-style diarization JSON body into a DiarizationRequest."""
    return await _build_request_from_body(body)


async def _run_diarization_job(job_id, request, service, webhook, webhook_status_only):
    """Background worker: run diarization, store the job output, fire webhook."""
    import asyncio

    from utils.diarization_jobs import get_job_store, post_webhook

    store = get_job_store()
    store.set_running(job_id)
    try:
        result = await service.process_request(request)
        output = result.to_dict()
        warning = output.get("warning")
        store.set_succeeded(job_id, output, warning=warning)
    except Exception as e:  # noqa: BLE001 - record failure on the job
        store.set_failed(job_id, str(e))

    if webhook:
        job = store.get(job_id)
        if job is not None:
            payload = job.created_dict() if webhook_status_only else job.job_dict()
            await asyncio.to_thread(post_webhook, webhook, payload)


# ---------------------------------------------------------------------------
# pyannoteAI job API
#
# These sit at the official paths (POST /v1/diarize, GET /v1/jobs/{jobId}) so a
# pyannoteAI client can switch base URL only. There is deliberately no
# synchronous variant: the official API has none, and one published under
# /v1/audio/diarize was a path no pyannoteAI client would ever call.
# ---------------------------------------------------------------------------

async_router = APIRouter()


@async_router.post("/diarize", status_code=201)
async def create_diarization_job(
    body: dict = Depends(_read_json_body),
    service=Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Create an async diarization job; returns pyannoteAI JobCreated (201)."""
    import asyncio

    from utils.diarization_jobs import get_job_store

    diar_request = await _build_request_from_body(body)

    webhook = body.get("webhook")
    webhook_status_only = bool(body.get("webhookStatusOnly", False))

    job = get_job_store().create()
    asyncio.create_task(
        _run_diarization_job(
            job.job_id, diar_request, service, webhook, webhook_status_only
        )
    )
    return job.created_dict()


@async_router.get("/jobs/{jobId}")
async def get_diarization_job(
    jobId: str,  # noqa: N803 - the official spec spells the path parameter this way
    api_key: str = Security(get_api_key),
):
    """Return the pyannoteAI DiarizationJob for a job id."""
    from utils.diarization_jobs import get_job_store

    job = get_job_store().get(jobId)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {jobId!r} not found")
    return job.job_dict()
