# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""``--dev-mode`` has to run the checkout's tt-media-server, not the image's.

Dev mode bind-mounts the repo's tt-media-server over the copy baked into the
image, but only for the model types listed in run_docker_server. Diarization
was not listed, so ``--dev-mode`` silently served whatever the image was built
from and a fix under test never reached the container.
"""

from pathlib import Path
from unittest.mock import patch

from workflows.model_spec import get_model_spec_map, load_templates_from_yaml
from workflows.run_docker_server import generate_docker_run_command
from workflows.runtime_config import RuntimeConfig
from workflows.utils import get_repo_root_path

_MEDIA_SERVER_DST = "/home/container_app_user/tt-metal/server"


def _diarization_spec():
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "audio_tts.yaml"
    )
    specs = get_model_spec_map(
        [t for t in templates if "pyannote/speaker-diarization-community-1" in t.weights]
    )
    return next(s for s in specs.values() if s.device_type.name == "P150")


def _docker_command(dev_mode):
    spec = _diarization_spec()
    runtime_config = RuntimeConfig(
        model=spec.model_name,
        device="p150",
        workflow="server",
        service_port="8000",
        dev_mode=dev_mode,
        docker_server=True,
        no_auth=True,
    )
    with patch(
        "workflows.run_docker_server.get_repo_root_path", return_value=Path("/repo")
    ), patch("workflows.run_docker_server.DeviceTypes"):
        command, _ = generate_docker_run_command(spec, runtime_config)
    return " ".join(str(part) for part in command)


def test_dev_mode_mounts_the_checkout_media_server():
    assert (
        f"type=bind,src=/repo/tt-media-server,dst={_MEDIA_SERVER_DST}"
        in _docker_command(dev_mode=True)
    )


def test_without_dev_mode_the_image_copy_is_used():
    assert "src=/repo/tt-media-server" not in _docker_command(dev_mode=False)
