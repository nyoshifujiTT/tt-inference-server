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

from typing import Optional

from domain.diarization_request import DiarizationRequest
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Security, UploadFile
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

router = APIRouter()

# pyannoteAI DiarizeRequest fields this server implements (accepted in the JSON
# body) vs. deliberately-unsupported ones. Kept explicit so the schema
# conformance test can assert the union covers the whole official schema and
# nothing is silently ignored. See https://docs.pyannote.ai/openapi.json.
IMPLEMENTED_REQUEST_FIELDS = frozenset(
    {"url", "model", "numSpeakers", "minSpeakers", "maxSpeakers", "exclusive"}
)
# precision-2-only options: accepted by the parser only to reject them with 400.
UNSUPPORTED_REQUEST_FIELDS = frozenset(
    {
        "confidence",
        "turnLevelConfidence",
        "transcription",
        "transcriptionConfig",
        "webhook",
        "webhookStatusOnly",
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
    requested = [name for name, value in options.items() if value not in (None, False, "")]
    if requested:
        raise HTTPException(
            status_code=400,
            detail=(
                "the following pyannoteAI options require the paid precision-2 "
                f"model and are not supported by this community-1 server: "
                f"{sorted(requested)}"
            ),
        )


async def parse_diarization_request(request: Request) -> DiarizationRequest:
    """Parse a pyannoteAI-style diarization JSON body into a DiarizationRequest.

    Body matches the pyannoteAI ``DiarizeRequest`` schema
    (https://docs.pyannote.ai/openapi.json): required ``url`` (http(s):// or
    media://), optional ``model`` (validated against the served model),
    ``numSpeakers`` / ``minSpeakers`` / ``maxSpeakers``, and ``exclusive``. The
    precision-2-only options are rejected. The referenced audio is fetched here
    and passed to the service as bytes.
    """
    from utils.audio_url_resolver import AudioUrlError, resolve_audio_url

    if "application/json" not in request.headers.get("content-type", "").lower():
        raise HTTPException(
            status_code=415,
            detail="diarize requires application/json with a 'url' field",
        )
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

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


async def parse_diarized_transcription_request(
    file: UploadFile = File(...),
    model: str = Form(...),
    num_speakers: Optional[int] = Form(None, alias="numSpeakers"),
    min_speakers: Optional[int] = Form(None, alias="minSpeakers"),
    max_speakers: Optional[int] = Form(None, alias="maxSpeakers"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
) -> dict:
    file_content = await file.read()
    return {
        "request": DiarizationRequest(
            file=file_content,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            exclusive=True,
        ),
        "model": model,
        "language": language,
        "prompt": prompt,
    }


@router.post("/diarized-transcriptions")
async def diarized_transcriptions(
    parsed: dict = Depends(parse_diarized_transcription_request),
    service=Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Speaker-diarized transcription (OpenAI diarized_json).

    ``model`` is a composite id "<asr_model>+<diarization_model>". Diarization is
    run by this service (community-1); each speaker turn is transcribed by the
    ASR model via the configured ASR endpoint (settings.asr_url).
    """
    try:
        result = await service.diarized_transcription(
            parsed["request"],
            model=parsed["model"],
            language=parsed["language"],
            prompt=parsed["prompt"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    return result
