# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The configured object-storage host is always downloadable.

``media_url_allowed_domains`` is the SSRF guard for URLs a *client* chose. A
``media://`` key is not one of those: it resolves to a url this server signed
against its own storage endpoint, so the allowlist has nothing to guard there.
Leaving the host out made staging fail closed on every deployment, and the
obvious fix for that -- an operator widening the allowlist by hand -- is the
bigger hole.
"""

import pytest
import utils.media_object_storage as mos
from utils.media_downloader import (
    MediaDownloadError,
    MediaDownloadPolicyError,
    check_media_url_policy,
)
from utils.media_downloader import settings as downloader_settings

_ENDPOINT = "https://storage.internal:9000"


@pytest.fixture()
def storage_configured(monkeypatch):
    monkeypatch.setattr(downloader_settings, "media_storage_endpoint", _ENDPOINT, False)
    monkeypatch.setattr(downloader_settings, "media_storage_bucket", "media", False)
    monkeypatch.setattr(downloader_settings, "media_url_allowed_domains", "", False)
    mos.reset_client()
    yield
    mos.reset_client()


def test_the_storage_host_is_allowed_without_being_listed(storage_configured):
    parsed = check_media_url_policy(f"{_ENDPOINT}/media/sess/a.wav?X-Amz-Signature=x")
    assert parsed.host == "storage.internal"


def test_it_does_not_open_the_allowlist_to_anything_else(storage_configured):
    """Adding the storage host must not amount to allowing every host."""
    with pytest.raises(MediaDownloadPolicyError):
        check_media_url_policy("https://evil.example.com/a.wav")


def test_a_listed_domain_still_works_alongside_it(monkeypatch, storage_configured):
    monkeypatch.setattr(
        downloader_settings, "media_url_allowed_domains", "cdn.example.com", False
    )
    assert check_media_url_policy("https://cdn.example.com/a.wav").host
    assert check_media_url_policy(f"{_ENDPOINT}/media/a.wav").host


def test_without_storage_the_allowlist_is_still_required(monkeypatch):
    monkeypatch.setattr(downloader_settings, "media_storage_endpoint", "", False)
    monkeypatch.setattr(downloader_settings, "media_storage_bucket", "", False)
    monkeypatch.setattr(downloader_settings, "media_url_allowed_domains", "", False)
    mos.reset_client()
    with pytest.raises(MediaDownloadPolicyError):
        check_media_url_policy("https://cdn.example.com/a.wav")


def test_a_bad_allowlist_entry_is_still_operator_error(monkeypatch, storage_configured):
    """The storage host is added inside the same guarded block, so it must not
    have changed how a malformed entry is reported."""
    monkeypatch.setattr(downloader_settings, "media_url_allowed_domains", "*", False)
    with pytest.raises(MediaDownloadError) as excinfo:
        check_media_url_policy(f"{_ENDPOINT}/media/a.wav")
    assert not isinstance(excinfo.value, MediaDownloadPolicyError)
