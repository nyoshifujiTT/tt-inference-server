# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os
import time

import pytest

from utils.media_storage import (
    MediaStorage,
    MediaStorageError,
    parse_media_key,
)


def test_parse_media_key_ok():
    assert parse_media_key("media://foo/bar-1_2.wav") == "foo/bar-1_2.wav"


@pytest.mark.parametrize(
    "bad",
    [
        "https://x/y.wav",  # wrong scheme
        "media://",  # empty key
        "media://../etc/passwd",  # traversal
        "media://a b",  # space not allowed
    ],
)
def test_parse_media_key_rejects(bad):
    with pytest.raises(MediaStorageError):
        parse_media_key(bad)


def test_declare_put_get_roundtrip(tmp_path):
    st = MediaStorage(root=str(tmp_path))
    key = st.declare("media://sess/audio.wav")
    assert key == "sess/audio.wav"
    st.put(key, b"RIFFDATA")
    assert st.get("media://sess/audio.wav") == b"RIFFDATA"
    assert st.get(key) == b"RIFFDATA"


def test_get_missing_raises(tmp_path):
    st = MediaStorage(root=str(tmp_path))
    with pytest.raises(MediaStorageError):
        st.get("media://nope.wav")


def test_expiry(tmp_path):
    st = MediaStorage(root=str(tmp_path), retention_seconds=1)
    st.put("k.wav", b"x")
    # backdate mtime beyond retention
    p = os.path.join(str(tmp_path), "k.wav")
    old = time.time() - 10
    os.utime(p, (old, old))
    with pytest.raises(MediaStorageError):
        st.get("k.wav")
    assert not os.path.exists(p)  # expired object removed on access


def test_put_rejects_traversal_key(tmp_path):
    st = MediaStorage(root=str(tmp_path))
    with pytest.raises(MediaStorageError):
        st.put("../evil", b"x")


def test_sweep_removes_expired(tmp_path):
    st = MediaStorage(root=str(tmp_path), retention_seconds=1)
    st.put("a.wav", b"x")
    st.put("b.wav", b"y")
    for name in ("a.wav", "b.wav"):
        p = os.path.join(str(tmp_path), name)
        old = time.time() - 10
        os.utime(p, (old, old))
    assert st.sweep() == 2
