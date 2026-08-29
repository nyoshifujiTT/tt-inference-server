# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The knobs the diarization service reads have to reach the container.

Only two things cross the container boundary on the standard path: the secrets
in ``.env`` and whatever ``run.py`` puts on the docker command. A setting that
is merely exported in the operator's shell is silently dropped, and the service
then runs its default without complaining -- which is exactly how a documented
``export DIARIZATION_TT_SEGMENTATION=1`` turned into a no-op. Pin the setting to
the catalog entry, which ``run_docker_server`` propagates.
"""

from workflows.model_spec import load_templates_from_yaml
from workflows.utils import get_repo_root_path
from workflows.workflow_types import DeviceTypes


SEGMENTATION_ENV_VAR = "DIARIZATION_TT_SEGMENTATION"


def _diarization_device_spec():
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "audio_tts.yaml"
    )
    for template in templates:
        if "pyannote/speaker-diarization-community-1" in template.weights:
            for device_spec in template.device_model_specs:
                if device_spec.device == DeviceTypes.P150:
                    return device_spec
    raise AssertionError("the diarization template lost its P150 device spec")


def test_the_segmentation_toggle_is_declared_in_the_catalog():
    """Declared here, so it lands on the docker command instead of being dropped."""
    device_spec = _diarization_device_spec()
    assert SEGMENTATION_ENV_VAR in device_spec.env_vars


def test_the_segmentation_toggle_defaults_to_host_side():
    """The default keeps the short-lived segmentation net off the device."""
    device_spec = _diarization_device_spec()
    assert str(device_spec.env_vars[SEGMENTATION_ENV_VAR]) == "0"
