# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import http.server
import threading

import pytest

from utils.audio_url_resolver import AudioUrlError, resolve_audio_url
from utils.media_storage import MediaStorage


class _Handler(http.server.BaseHTTPRequestHandler):
    payload = b"RIFFwavdata"

    def do_GET(self):  # noqa: N802
        if self.path == "/ok.wav":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(self.payload)
        elif self.path == "/big.wav":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"x" * 100)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):  # silence
        pass


@pytest.fixture(scope="module")
def http_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_resolve_http_ok(http_server):
    assert resolve_audio_url(http_server + "/ok.wav") == b"RIFFwavdata"


def test_resolve_http_404(http_server):
    with pytest.raises(AudioUrlError):
        resolve_audio_url(http_server + "/missing.wav")


def test_resolve_http_oversize(http_server):
    with pytest.raises(AudioUrlError):
        resolve_audio_url(http_server + "/big.wav", max_bytes=10)


def test_resolve_media(tmp_path):
    st = MediaStorage(root=str(tmp_path))
    st.put("s/a.wav", b"MEDIABYTES")
    assert resolve_audio_url("media://s/a.wav", storage=st) == b"MEDIABYTES"


def test_resolve_media_missing(tmp_path):
    st = MediaStorage(root=str(tmp_path))
    with pytest.raises(AudioUrlError):
        resolve_audio_url("media://nope.wav", storage=st)


def test_unsupported_scheme():
    with pytest.raises(AudioUrlError):
        resolve_audio_url("ftp://host/a.wav")


def test_empty_url():
    with pytest.raises(AudioUrlError):
        resolve_audio_url("")
