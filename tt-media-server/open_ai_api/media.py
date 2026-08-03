# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""pyannoteAI-compatible temporary media input endpoints.

Implements the two-step upload flow from
https://docs.pyannote.ai/api-reference/upload-media:

  - ``POST /v1/media/input`` with ``{"url": "media://<object-key>"}`` declares an
    object key and returns ``{"url": "<put-url>"}`` (MediaResponse). The put-url
    is served by this same server.
  - ``PUT /v1/media/input/{object_key}`` receives the raw file bytes.

The staged ``media://<object-key>`` can then be passed as
``DiarizeRequest.url``.
"""

from fastapi import APIRouter, HTTPException, Request, Security
from security.api_key_checker import get_api_key
from utils.media_storage import MediaStorageError, get_media_storage, parse_media_key

router = APIRouter()


@router.post("/input", status_code=201)
async def create_media_input(
    body: dict,
    request: Request,
    api_key: str = Security(get_api_key),
):
    """Declare a media:// object key and return a PUT url (MediaResponse)."""
    url = body.get("url") if isinstance(body, dict) else None
    if not url:
        raise HTTPException(status_code=400, detail="'url' (media://<key>) is required")
    try:
        key = get_media_storage().declare(url)
    except MediaStorageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    base = str(request.base_url).rstrip("/")
    return {"url": f"{base}/v1/media/input/{key}"}


@router.put("/input/{object_key:path}")
async def put_media_input(
    object_key: str,
    request: Request,
    api_key: str = Security(get_api_key),
):
    """Store the raw bytes for a previously-declared object key."""
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty request body")
    try:
        get_media_storage().put(object_key, data)
    except MediaStorageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"url": f"media://{object_key}"}
