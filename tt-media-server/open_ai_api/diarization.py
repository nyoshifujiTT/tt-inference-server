# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Speaker-diarization-only endpoint.

Schema aligned with the pyannoteAI cloud diarization API so a client can switch
base URL only. Unlike /v1/audio/transcriptions this returns speaker turns only
(no transcript). The /diarize input is a pyannoteAI-style JSON body with a
``url`` (http(s):// or media://); see https://docs.pyannote.ai/openapi.json
(DiarizeRequest).
"""

from domain.diarization_request import DiarizationRequest
from fastapi import APIRouter, Body, Depends, HTTPException, Security
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

router = APIRouter()

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


async def _build_request_from_body(body: dict) -> DiarizationRequest:
    """Validate a pyannoteAI DiarizeRequest body and resolve it to a request.

    Validates ``model`` and rejects precision-2-only options, requires ``url``
    (http(s):// or media://), fetches the audio bytes, and builds the internal
    DiarizationRequest. See https://docs.pyannote.ai/openapi.json.
    """
    from utils.audio_url_resolver import AudioUrlError, resolve_audio_url

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
            status_code=400, detail="'url' is required (http(s):// or media://)"
        )
    try:
        audio_bytes = resolve_audio_url(url)
    except AudioUrlError as e:
        raise HTTPException(status_code=400, detail=str(e))

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


@router.post("/diarize")
async def diarize(
    request: DiarizationRequest = Depends(parse_diarization_request),
    service=Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Run speaker diarization and return pyannoteAI-shaped segments."""
    try:
        result = await service.process_request(request)
    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 like audio route
        raise HTTPException(status_code=500, detail=str(e))
    return result.to_dict()


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
# pyannoteAI-native asynchronous job API
#
# These live at the pyannoteAI paths (POST /v1/diarize, GET /v1/jobs/{jobId})
# rather than under /v1/audio, so a pyannoteAI client can switch base URL only.
# The synchronous /v1/audio/diarize above is kept as a convenience that returns
# the DiarizationJobOutput directly.
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


@async_router.get("/jobs/{job_id}")
async def get_diarization_job(
    job_id: str,
    api_key: str = Security(get_api_key),
):
    """Return the pyannoteAI DiarizationJob for a job id."""
    from utils.diarization_jobs import get_job_store

    job = get_job_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    return job.job_dict()
