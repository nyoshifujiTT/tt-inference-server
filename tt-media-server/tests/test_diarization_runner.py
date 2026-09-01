# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Diarization reaches the device the same way every other model does.

The point of these is that nothing here is diarization-specific: the device is
opened by ``BaseMetalDeviceRunner`` from the mesh the catalog resolved, the
runner is reached through the fabric, and a device that cannot be opened is an
error. The service used to do all of this itself, with its own device-id
parsing and its own ``ttnn.open_device`` call, which is how a mesh descriptor
that did not match the host turned into a server quietly running on CPU.
"""

import types
from unittest.mock import MagicMock, patch

import pytest


def _stub_ttnn(monkeypatch, open_mesh_device):
    """Point the module's already-imported ttnn at a stand-in.

    base_metal_device_runner does ``import ttnn`` at module scope, so replacing
    sys.modules after the fact would be ignored; patch the bound name instead.
    """
    import tt_model_runners.base_metal_device_runner as runner_module

    fake = types.SimpleNamespace(
        open_mesh_device=open_mesh_device,
        close_mesh_device=lambda device: None,
        get_device_ids=lambda: [0],
        MeshShape=lambda shape: shape,
        DispatchCoreConfig=lambda *a, **k: object(),
        DispatchCoreAxis=types.SimpleNamespace(ROW=object(), COL=object()),
        device=types.SimpleNamespace(is_blackhole=lambda: False),
        set_fabric_config=lambda config: None,
        FabricConfig=types.SimpleNamespace(DISABLED=object()),
    )
    monkeypatch.setattr(runner_module, "ttnn", fake)
    return fake


@patch("tt_model_runners.runner_fabric.settings")
@patch("tt_model_runners.base_device_runner.get_settings")
@patch("tt_model_runners.base_device_runner.setup_runner_environment")
def test_the_fabric_builds_the_diarization_runner(
    _mock_setup_env, mock_get_settings, mock_fabric_settings
):
    """The Scheduler asks the fabric for a runner by name; it must be there."""
    from tt_model_runners.runner_fabric import get_device_runner

    mock_fabric_settings.model_runner = "tt-pyannote-diarization"
    settings = MagicMock()
    settings.device_mesh_shape = (1, 1)
    mock_get_settings.return_value = settings

    runner = get_device_runner("0")

    assert type(runner).__name__ == "TTDiarizationRunner"


@patch("tt_model_runners.base_device_runner.get_settings")
@patch("tt_model_runners.base_device_runner.setup_runner_environment")
def test_the_l1_reservation_is_passed_through_the_standard_device_params(
    _mock_setup_env, mock_get_settings
):
    """The only device knob this model owns travels the normal route."""
    from tt_model_runners.diarization_runner import (
        DIARIZATION_L1_SMALL_SIZE,
        TTDiarizationRunner,
    )

    settings = MagicMock()
    settings.device_mesh_shape = (1, 1)
    mock_get_settings.return_value = settings

    params = TTDiarizationRunner("0").get_pipeline_device_params()

    assert params == {"l1_small_size": DIARIZATION_L1_SMALL_SIZE}


@patch("tt_model_runners.base_device_runner.get_settings")
@patch("tt_model_runners.base_device_runner.setup_runner_environment")
def test_a_device_that_cannot_be_opened_fails_the_runner(
    _mock_setup_env, mock_get_settings, monkeypatch
):
    """No CPU fallback: the nets on the device are the reason to serve this."""
    from tt_model_runners.diarization_runner import TTDiarizationRunner

    settings = MagicMock()
    settings.device_mesh_shape = (1, 1)
    mock_get_settings.return_value = settings

    def boom(**kwargs):
        raise RuntimeError("Physical chip id 0 not found in control plane chip mapping")

    _stub_ttnn(monkeypatch, boom)

    with pytest.raises(RuntimeError):
        TTDiarizationRunner("0").set_device()


@patch("tt_model_runners.base_device_runner.get_settings")
@patch("tt_model_runners.base_device_runner.setup_runner_environment")
def test_running_before_warmup_is_an_error(_mock_setup_env, mock_get_settings):
    """A worker that never loaded must not answer as though it had."""
    from tt_model_runners.diarization_runner import TTDiarizationRunner

    settings = MagicMock()
    settings.device_mesh_shape = (1, 1)
    mock_get_settings.return_value = settings

    with pytest.raises(RuntimeError, match="not loaded"):
        TTDiarizationRunner("0").run([])


@patch("tt_model_runners.base_device_runner.get_settings")
@patch("tt_model_runners.base_device_runner.setup_runner_environment")
def test_requests_are_not_batched(_mock_setup_env, mock_get_settings):
    """pyannote mutates pipeline state per call, so batching would corrupt it."""
    from tt_model_runners.diarization_runner import TTDiarizationRunner

    settings = MagicMock()
    settings.device_mesh_shape = (1, 1)
    mock_get_settings.return_value = settings

    assert TTDiarizationRunner("0").is_request_batchable(object()) is False
