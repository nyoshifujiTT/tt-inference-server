# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""DiarizationService TT-acceleration env toggle (device-independent)."""
import os
import sys
import types

import pytest


def _load_service(monkeypatch, captured):
    # Stub DiarizationBackend to capture nn_accelerator without needing pyannote.
    import model_services.diarization_service as svc

    class _FakeBackend:
        def __init__(self, model_path, device="cpu", nn_accelerator=None):
            captured["nn_accelerator"] = nn_accelerator

    monkeypatch.setattr(svc, "DiarizationBackend", _FakeBackend)
    return svc


def test_no_tt_env_means_cpu_no_accelerator(monkeypatch):
    monkeypatch.delenv("DIARIZATION_TT_DEVICE_ID", raising=False)
    captured = {}
    svc = _load_service(monkeypatch, captured)
    svc.DiarizationService()
    assert captured["nn_accelerator"] is None


def test_tt_env_but_ttnn_unavailable_falls_back_to_cpu(monkeypatch):
    monkeypatch.setenv("DIARIZATION_TT_DEVICE_ID", "0")
    # Ensure importing ttnn fails -> graceful fallback (nn_accelerator None)
    monkeypatch.setitem(sys.modules, "ttnn", None)  # import ttnn -> ImportError
    captured = {}
    svc = _load_service(monkeypatch, captured)
    svc.DiarizationService()
    assert captured["nn_accelerator"] is None
