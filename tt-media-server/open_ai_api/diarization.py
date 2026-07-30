# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Speaker-diarization-only endpoint.

Schema aligned with the pyannoteAI cloud diarization API so a client can switch
base URL only. Unlike /v1/audio/transcriptions this returns speaker turns only
(no transcript). Accepts multipart file upload (OpenAI-audio style) so existing
audio clients can post the same way.
"""

from typing import Optional

from domain.diarization_request import DiarizationRequest
from fastapi import APIRouter, Depends, File, Form, HTTPException, Security, UploadFile
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

router = APIRouter()


async def parse_diarization_request(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    exclusive: Optional[bool] = Form(True),
) -> DiarizationRequest:
    file_content = await file.read()
    return DiarizationRequest(
        file=file_content,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        exclusive=exclusive if exclusive is not None else True,
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
