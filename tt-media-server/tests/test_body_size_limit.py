# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""An oversized body is refused before it is received.

Applying a size cap in the endpoint is too late: the ASGI layer has already
buffered the request. Measured on a p150, a 1 GiB inline body the endpoint
*rejected* still cost +1289 MiB RSS, and a 900 MiB one OOM-killed a 6 GiB
container. Checking the declared Content-Length instead refuses while the
payload is still on the client -- +1 MiB rather than +894 MiB for a 900 MiB
announced body.

That difference is what lets the inline audio cap be the official 1 GiB rather
than a smaller number picked to survive what could not be refused in time.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from open_ai_api.body_size_limit import BodySizeLimitMiddleware


def _client(limit):
    app = FastAPI()

    @app.post("/v1/diarize")
    async def diarize(body: dict):
        return {"received": len(body.get("url", ""))}

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=lambda: limit)
    return TestClient(app)


def test_a_body_over_the_limit_is_refused():
    resp = _client(1024).post("/v1/diarize", json={"url": "A" * 4096})
    assert resp.status_code == 413, resp.text
    assert "over the" in resp.json()["detail"]


def test_the_handler_never_runs_for_an_oversized_body():
    """The point is that the payload is not processed, not merely that the
    status is 413 -- a 413 produced after buffering saves nothing."""
    resp = _client(1024).post("/v1/diarize", json={"url": "A" * 4096})
    assert "received" not in resp.json()


def test_a_body_inside_the_limit_passes_through():
    resp = _client(1024 * 1024).post("/v1/diarize", json={"url": "A" * 128})
    assert resp.status_code == 200, resp.text
    assert resp.json()["received"] == 128


def test_a_zero_limit_disables_the_check():
    """0 means "no declared-length limit"; the endpoint's own cap still runs."""
    resp = _client(0).post("/v1/diarize", json={"url": "A" * 4096})
    assert resp.status_code == 200, resp.text


def test_a_malformed_content_length_is_left_to_the_protocol_layer():
    """Inventing a status for a malformed header would mask a protocol error
    as a size error."""
    client = _client(1024)
    resp = client.post(
        "/v1/diarize",
        content=b'{"url":"A"}',
        headers={"content-type": "application/json", "content-length": "not-a-number"},
    )
    assert resp.status_code != 413
