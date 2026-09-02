# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parsing of TT_METAL_MODEL_DEV_MOUNTS into docker bind-mount arguments."""

import pytest

from workflows.run_docker_server import _tt_metal_model_dev_mounts

USER_HOME = "/home/container_app_user"


def test_unset_adds_no_mounts(monkeypatch):
    monkeypatch.delenv("TT_METAL_MODEL_DEV_MOUNTS", raising=False)
    assert _tt_metal_model_dev_mounts(USER_HOME) == []


def test_relative_dst_resolves_under_tt_metal(monkeypatch, tmp_path):
    src = tmp_path / "qwen36"
    src.mkdir()
    monkeypatch.setenv(
        "TT_METAL_MODEL_DEV_MOUNTS", f"{src}:models/demos/blackhole/qwen36"
    )
    assert _tt_metal_model_dev_mounts(USER_HOME) == [
        "--mount",
        f"type=bind,src={src.resolve()},dst={USER_HOME}/tt-metal/models/demos/blackhole/qwen36",
    ]


def test_absolute_dst_is_used_as_given(monkeypatch, tmp_path):
    src = tmp_path / "platform.py"
    src.write_text("")
    dst = "/opt/venv/lib/python3.10/site-packages/vllm_tt_plugin/platform.py"
    monkeypatch.setenv("TT_METAL_MODEL_DEV_MOUNTS", f"{src}:{dst}")
    assert _tt_metal_model_dev_mounts(USER_HOME) == [
        "--mount",
        f"type=bind,src={src.resolve()},dst={dst}",
    ]


def test_multiple_entries_keep_their_order(monkeypatch, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv(
        "TT_METAL_MODEL_DEV_MOUNTS", f"{first}:models/a, {second}:models/b"
    )
    mounts = _tt_metal_model_dev_mounts(USER_HOME)
    assert mounts[1].endswith("dst=/home/container_app_user/tt-metal/models/a")
    assert mounts[3].endswith("dst=/home/container_app_user/tt-metal/models/b")


def test_entry_without_dst_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_METAL_MODEL_DEV_MOUNTS", str(tmp_path))
    with pytest.raises(ValueError, match="is not 'src:dst'"):
        _tt_metal_model_dev_mounts(USER_HOME)


def test_missing_source_is_rejected(monkeypatch, tmp_path):
    # A typo would otherwise let docker create an empty directory and silently serve image code.
    monkeypatch.setenv("TT_METAL_MODEL_DEV_MOUNTS", f"{tmp_path / 'absent'}:models/x")
    with pytest.raises(FileNotFoundError):
        _tt_metal_model_dev_mounts(USER_HOME)
