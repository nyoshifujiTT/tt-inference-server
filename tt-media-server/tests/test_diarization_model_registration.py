# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""The diarization model must resolve through the standard startup path.

`run.py --docker-server` passes the model name in MODEL, and Settings turns that
into a runner and a device config via ModelNames -> ModelRunners ->
ModelConfigs. Registering the model in SupportedModels alone is not enough: the
container starts, hits `ModelNames(model_to_run)` and dies with "is not a valid
ModelNames" before serving anything. That is exactly what happened, and these
tests pin each link of the chain so it cannot happen again.
"""

import pytest

from config.constants import (
    INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP,
    MODEL_SERVICE_RUNNER_MAP,
    DeviceTypes,
    ModelConfigs,
    ModelNames,
    ModelRunners,
    ModelServices,
    SupportedModels,
)

MODEL_NAME = "speaker-diarization-community-1"


def test_model_name_is_registered():
    """`ModelNames(MODEL)` is the first thing Settings does with MODEL."""
    assert ModelNames(MODEL_NAME) is ModelNames.PYANNOTE_SPEAKER_DIARIZATION_COMMUNITY_1


def test_model_maps_to_a_runner():
    """Without this mapping Settings raises 'No model runner found for model'."""
    runners = [
        runner
        for runner, names in INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP.items()
        if ModelNames.PYANNOTE_SPEAKER_DIARIZATION_COMMUNITY_1 in names
    ]
    assert runners == [ModelRunners.TT_PYANNOTE_DIARIZATION]


def test_runner_has_a_device_config():
    """Settings looks up (runner, device); a miss leaves the config unapplied."""
    config = ModelConfigs[(ModelRunners.TT_PYANNOTE_DIARIZATION, DeviceTypes.P150)]
    # One device, no mesh: pyannote runs on host and only the two nets offload.
    assert config["device_mesh_shape"] == (1, 1)
    assert config["is_galaxy"] is False


def test_runner_maps_to_the_diarization_service():
    """run.py passes only MODEL and DEVICE -- never MODEL_SERVICE.

    Settings deduces the service from the runner through this map, so without
    an entry the container comes up with no service selected and serves
    nothing, even once the name resolves.
    """
    services = [
        service
        for service, runners in MODEL_SERVICE_RUNNER_MAP.items()
        if ModelRunners.TT_PYANNOTE_DIARIZATION in runners
    ]
    assert services == [ModelServices.DIARIZATION]


def test_weights_path_resolves_from_the_model_name():
    """Settings reads SupportedModels by the ModelNames *member name*.

    The two enums have to agree on that name or the weights path silently stays
    at its default.
    """
    member = ModelNames.PYANNOTE_SPEAKER_DIARIZATION_COMMUNITY_1.name
    assert getattr(SupportedModels, member).value == (
        "pyannote/speaker-diarization-community-1"
    )


def test_settings_resolves_the_whole_chain():
    """End to end, the way the container does it.

    Run in a subprocess: conftest replaces config.settings with a stub for the
    rest of the suite, so importing it in-process would exercise the stub
    rather than the resolution being tested here.
    """
    import json
    import os
    import subprocess
    import sys

    script = (
        "from config.settings import Settings;"
        "s = Settings();"
        "import json;"
        "print(json.dumps({'runner': s.model_runner, 'weights': s.model_weights_path,"
        " 'service': s.model_service}))"
    )
    # Exactly what run.py --docker-server sets: no MODEL_SERVICE, no runner,
    # no weights path. Everything else has to be deduced.
    env = dict(os.environ, MODEL=MODEL_NAME, DEVICE="p150", PYTHONPATH=".")
    for leftover in ("MODEL_SERVICE", "MODEL_RUNNER", "MODEL_WEIGHTS_PATH"):
        env.pop(leftover, None)

    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    resolved = json.loads(out.stdout.strip().splitlines()[-1])

    assert resolved["runner"] == ModelRunners.TT_PYANNOTE_DIARIZATION.value
    assert resolved["weights"] == "pyannote/speaker-diarization-community-1"
    assert resolved["service"] == ModelServices.DIARIZATION.value


def test_an_unregistered_model_still_fails_loudly():
    """Guard the guard: the lookup must reject unknown names, not accept them."""
    with pytest.raises(ValueError):
        ModelNames("speaker-diarization-not-a-real-model")


def test_the_diarization_api_is_served_for_this_model():
    """The point of all the registration: MODEL alone must serve the API.

    Routes are chosen from settings.model_service at import time, so this is
    the end of the chain -- if any earlier link is missing the app comes up
    with no diarization endpoints even though the container stays alive.
    """
    import json
    import os
    import subprocess
    import sys

    script = (
        "from fastapi import FastAPI;"
        "from fastapi.testclient import TestClient;"
        "from open_ai_api import api_router;"
        "app = FastAPI();"
        "app.include_router(api_router);"
        "import json;"
        "print(json.dumps(sorted(TestClient(app).get('/openapi.json').json()['paths'])))"
    )
    env = dict(os.environ, MODEL=MODEL_NAME, DEVICE="p150", NO_AUTH="1", PYTHONPATH=".")
    for leftover in ("MODEL_SERVICE", "MODEL_RUNNER", "MODEL_WEIGHTS_PATH"):
        env.pop(leftover, None)

    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    served = set(json.loads(out.stdout.strip().splitlines()[-1]))

    # The pyannoteAI-shaped surface, plus the media staging the diarize body
    # references by media:// url.
    assert "/v1/diarize" in served
    assert "/v1/jobs/{jobId}" in served
    assert "/v1/media/input" in served
