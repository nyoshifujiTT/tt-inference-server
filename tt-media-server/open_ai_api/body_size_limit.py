# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Refuse an oversized request body before it is received.

An endpoint cannot defend itself against a large body: by the time a handler
runs, the ASGI layer has already received and parsed the whole request. A cap
applied there is spent memory -- measured on a p150, a 1 GiB inline body that
the endpoint *rejected* still cost +1289 MiB RSS, and a 900 MiB one killed a 6
GiB container outright.

``Content-Length`` is sent before the body, so it can be checked while the
payload is still on the client. Measured with a 900 MiB announced body against
a 64 MiB limit: without this middleware the server peaked at +894 MiB, with it
at +1 MiB.

This is what lets the inline audio cap be the same 1 GiB the official
pyannoteAI API allows, rather than a smaller number chosen to survive bodies
the server had no way to refuse in time.

A chunked request has no ``Content-Length``; those still reach the endpoint and
are caught by its own check, which is why that check stays. This middleware
narrows the window, it does not replace the cap.
"""

import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Answer 413 on a declared body larger than ``max_bytes``.

    Args:
        max_bytes: Callable returning the current limit, so an operator's
            setting is read per request rather than frozen at startup.
    """

    def __init__(self, app, max_bytes):
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError:
                # Malformed header: let the protocol layer deal with it rather
                # than inventing a status for it here.
                return await call_next(request)

            limit = self._max_bytes()
            if limit and length > limit:
                logger.warning(
                    f"Refusing {request.url.path}: declared body {length} bytes "
                    f"is over the {limit}-byte limit"
                )
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            f"request body is {length} bytes, over the "
                            f"{limit}-byte limit for this server"
                        )
                    },
                )
        return await call_next(request)
