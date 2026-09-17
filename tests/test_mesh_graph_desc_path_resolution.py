# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""TT_MESH_GRAPH_DESC_PATH must mean the same thing in and out of a container.

The catalog spells this path relative ("../../tt-metal/...") and
``set_runtime_env_vars`` exports catalog values verbatim, so the string used to
be resolved against whatever cwd the server ran in:

* Docker runs from ``$APP_DIR/src`` with tt-metal as a sibling two levels up, so
  the relative path happened to land on ``$TT_METAL_HOME``.
* ``--local-server`` runs from the repo's ``vllm-tt-metal/src`` with tt-metal
  wherever the caller built it, so the same string pointed at a non-existent
  path and the device failed to open with a TT_FATAL on the descriptor.

Anchoring to ``TT_METAL_HOME`` fixes the local path while leaving the Docker
result byte-identical. These tests lock in both halves of that claim, plus the
cases that must *not* be rewritten.
"""
import builtins
import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

from workflows.utils import get_repo_root_path

MODULE_PATH = get_repo_root_path() / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"

DEV_EMBEDDING_SPECS = (
    get_repo_root_path() / "workflows" / "model_specs" / "dev" / "embedding.yaml"
)

SPEC_RELATIVE_VALUE = (
    "../../tt-metal/tt_metal/fabric/mesh_graph_descriptors/"
    "p150_mesh_graph_descriptor.textproto"
)
DESCRIPTOR_TAIL = (
    "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
)


@pytest.fixture
def run_vllm_api_server_module(monkeypatch):
    """Load the launcher without importing vLLM, mirroring test_run_vllm_api_server."""
    module_name = "test_mesh_graph_desc_path_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)

    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = MagicMock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    vllm = types.ModuleType("vllm")
    vllm.ModelRegistry = MagicMock()
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    utils = types.ModuleType("utils")
    utils.__path__ = [str(get_repo_root_path() / "utils")]
    monkeypatch.setitem(sys.modules, "utils", utils)

    logging_utils = types.ModuleType("utils.logging_utils")
    logging_utils.set_vllm_logging_config = MagicMock()
    monkeypatch.setitem(sys.modules, "utils.logging_utils", logging_utils)

    prompt_client = types.ModuleType("utils.prompt_client")
    prompt_client.run_background_trace_capture = MagicMock()
    monkeypatch.setitem(sys.modules, "utils.prompt_client", prompt_client)

    vllm_run_utils = types.ModuleType("utils.vllm_run_utils")
    vllm_run_utils.create_model_symlink = MagicMock()
    vllm_run_utils.get_encoded_api_key = MagicMock(return_value="encoded-api-key")
    monkeypatch.setitem(sys.modules, "utils.vllm_run_utils", vllm_run_utils)

    assert spec is not None and spec.loader is not None
    real_import = builtins.__import__

    def no_vllm_import(name, *args, **kwargs):
        assert not (name == "vllm" or name.startswith("vllm.")), (
            f"the launcher imported {name} at module scope"
        )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_vllm_import)
    try:
        spec.loader.exec_module(module)
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
    return module


def test_docker_layout_resolves_to_the_same_absolute_path_as_before(
    monkeypatch, run_vllm_api_server_module
):
    """The container is the behaviour we must not change."""
    monkeypatch.setenv("TT_METAL_HOME", "/home/container_app_user/tt-metal")
    model_spec = {
        "device_model_spec": {
            "env_vars": {"TT_MESH_GRAPH_DESC_PATH": SPEC_RELATIVE_VALUE}
        }
    }

    run_vllm_api_server_module.set_runtime_env_vars(model_spec)

    # Exactly what "../../tt-metal/..." used to resolve to from $APP_DIR/src.
    assert os.environ["TT_MESH_GRAPH_DESC_PATH"] == (
        f"/home/container_app_user/tt-metal/{DESCRIPTOR_TAIL}"
    )


def test_local_server_layout_resolves_into_the_built_checkout(
    monkeypatch, run_vllm_api_server_module
):
    """The bug: an arbitrary --tt-metal-home must still be found."""
    monkeypatch.setenv("TT_METAL_HOME", "/home/ubuntu/ttm-local")
    model_spec = {
        "device_model_spec": {
            "env_vars": {"TT_MESH_GRAPH_DESC_PATH": SPEC_RELATIVE_VALUE}
        }
    }

    run_vllm_api_server_module.set_runtime_env_vars(model_spec)

    resolved = os.environ["TT_MESH_GRAPH_DESC_PATH"]
    assert resolved == f"/home/ubuntu/ttm-local/{DESCRIPTOR_TAIL}"
    assert os.path.isabs(resolved), "a cwd-relative value would break device open"
    assert "tt-metal/tt-metal" not in resolved, "the checkout name was duplicated"


def test_absolute_values_are_left_alone(monkeypatch, run_vllm_api_server_module):
    """FORGE/Quetzal specs point into a wheel, not into the tt-metal checkout."""
    wheel_path = (
        "/home/container_app_user/app/server/venv-worker/lib/python3.12/"
        "site-packages/pjrt_plugin_tt/tt-metal/tt_metal/fabric/"
        "mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
    )
    monkeypatch.setenv("TT_METAL_HOME", "/home/container_app_user/tt-metal")
    model_spec = {
        "device_model_spec": {"env_vars": {"TT_MESH_GRAPH_DESC_PATH": wheel_path}}
    }

    run_vllm_api_server_module.set_runtime_env_vars(model_spec)

    assert os.environ["TT_MESH_GRAPH_DESC_PATH"] == wheel_path


def test_without_tt_metal_home_the_value_is_passed_through(
    monkeypatch, run_vllm_api_server_module
):
    """No anchor to use: keep the old behaviour rather than inventing a path."""
    monkeypatch.delenv("TT_METAL_HOME", raising=False)
    model_spec = {
        "device_model_spec": {
            "env_vars": {"TT_MESH_GRAPH_DESC_PATH": SPEC_RELATIVE_VALUE}
        }
    }

    run_vllm_api_server_module.set_runtime_env_vars(model_spec)

    assert os.environ["TT_MESH_GRAPH_DESC_PATH"] == SPEC_RELATIVE_VALUE


def test_other_env_vars_are_not_path_rewritten(
    monkeypatch, run_vllm_api_server_module
):
    """Only the descriptor path is anchored; everything else stays verbatim."""
    monkeypatch.setenv("TT_METAL_HOME", "/home/ubuntu/ttm-local")
    model_spec = {
        "device_model_spec": {
            "env_vars": {
                "VLLM__MAX_MODEL_LENGTH": "8192",
                "MAX_BATCH_SIZE": "32",
            }
        }
    }

    run_vllm_api_server_module.set_runtime_env_vars(model_spec)

    assert os.environ["VLLM__MAX_MODEL_LENGTH"] == "8192"
    assert os.environ["MAX_BATCH_SIZE"] == "32"


def test_the_qwen3_embedding_p150_specs_still_use_the_relative_form():
    """If the catalog switches to absolute paths, this resolution is dead code."""
    body = DEV_EMBEDDING_SPECS.read_text(encoding="utf-8")
    assert SPEC_RELATIVE_VALUE in body, (
        "the Qwen3-Embedding P150 specs no longer carry the relative descriptor "
        "path this resolution exists for; re-check whether it is still needed"
    )
