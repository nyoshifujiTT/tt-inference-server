# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""pyannoteAI-compatible temporary media input endpoint.

``POST /v1/media/input`` with ``{"url": "media://<object-key>"}`` declares an
object key and answers ``{"url": "<put-url>"}`` (MediaResponse), exactly as
https://docs.pyannote.ai/api-reference/upload-media describes. The put-url is a
pre-signed URL **on the object storage service**, which is what the official
API returns and what its clients expect: the upload goes straight to storage
and never through the process that schedules device work.

There is deliberately no ``PUT`` route here. This server used to serve the
upload itself, which put a receive path inside the official ``/v1/media``
namespace that no pyannoteAI client would ever call, and made every staged
recording travel through the inference API twice.
"""

from fastapi import APIRouter, HTTPException, Security
from security.api_key_checker import get_api_key
from utils.media_object_storage import (
    MediaStorageError,
    MediaStorageNotConfigured,
    presigned_put_url,
)

router = APIRouter()


@router.post("/input", status_code=201)
async def create_media_input(
    body: dict,
    api_key: str = Security(get_api_key),
):
    """Declare a media:// object key and return a pre-signed PUT url."""
    url = body.get("url") if isinstance(body, dict) else None
    if not url:
        raise HTTPException(status_code=400, detail="'url' (media://<key>) is required")
    try:
        return {"url": presigned_put_url(url)}
    except MediaStorageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except MediaStorageNotConfigured as e:
        # 501, not 500: the server is working as configured, this optional
        # capability was simply not turned on, and the message names the two
        # inputs that work without it.
        raise HTTPException(status_code=501, detail=str(e))
