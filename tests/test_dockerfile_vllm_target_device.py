# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Guards the VLLM_TARGET_DEVICE value used to build vLLM in the dev image.

Since tenstorrent/vllm ae0f073 ("Fully separate TT code to vllm_tt_plugin") the
fork's setup.py only recognises empty/cuda/hip/tpu/cpu/xpu. Building with the
image-wide VLLM_TARGET_DEVICE="tt" makes get_vllm_version() fall through to
`raise RuntimeError("Unknown runtime environment")`, which breaks the whole dev
image build. The build step must therefore override it to "empty", while the
runtime ENV stays "tt" (the TT platform is provided by the plugin at runtime).
"""

import re

from workflows.utils import get_repo_root_path

DOCKERFILE = (
    get_repo_root_path() / "vllm-tt-metal" / "vllm.tt-metal.src.dev.Dockerfile"
)


def _vllm_install_step() -> str:
    text = DOCKERFILE.read_text()
    match = re.search(
        r"RUN /bin/bash -c \"git clone \$\{TT_VLLM_REPO_URL\}.*?\"\n",
        text,
        re.DOTALL,
    )
    assert match, "vLLM install RUN step not found in dev Dockerfile"
    return match.group(0)


def _tt_metal_build_step() -> str:
    text = DOCKERFILE.read_text()
    match = re.search(
        r"RUN /bin/bash -c \"git clone \$\{TT_METAL_REPO_URL\}.*?\"\n",
        text,
        re.DOTALL,
    )
    assert match, "tt-metal build RUN step not found in dev Dockerfile"
    return match.group(0)


def test_vllm_editable_install_forces_empty_target_device():
    step = _vllm_install_step()
    assert "VLLM_TARGET_DEVICE=empty uv pip install" in step, (
        "vLLM must be built with VLLM_TARGET_DEVICE=empty; 'tt' is not a valid "
        "build target in tenstorrent/vllm setup.py"
    )


def test_runtime_target_device_stays_tt():
    text = DOCKERFILE.read_text()
    assert "VLLM_TARGET_DEVICE=tt" in text, (
        "the runtime ENV must keep VLLM_TARGET_DEVICE=tt so the TT platform is "
        "selected when the server runs"
    )


def test_tt_plugin_is_installed_after_vllm():
    """The tt platform comes from the fork-bundled plugin, not from vLLM core.

    Without this install the container starts and then dies with
    "RuntimeError: Failed to infer device type", because no
    "vllm.platform_plugins" entry point advertises the tt platform.
    """
    step = _vllm_install_step()
    assert "-e plugins/vllm-tt-plugin" in step, (
        "the fork-bundled vllm-tt-plugin must be installed in the dev image"
    )
    assert step.index("VLLM_TARGET_DEVICE=empty uv pip install") < step.index(
        "-e plugins/vllm-tt-plugin"
    ), "the plugin must be installed after vLLM so resolution cannot replace it"


def test_vllm_repo_url_is_overridable_and_defaults_to_tenstorrent():
    """A bring-up must be able to build from a fork branch.

    The commits of an in-flight bring-up are not on tenstorrent/vllm yet, so the
    clone URL has to be a build arg; the default must stay tenstorrent/vllm so
    ordinary builds are unchanged.
    """
    text = DOCKERFILE.read_text()
    assert (
        "ARG TT_VLLM_REPO_URL=https://github.com/tenstorrent/vllm.git" in text
    ), "TT_VLLM_REPO_URL must exist and default to tenstorrent/vllm"
    assert "git clone ${TT_VLLM_REPO_URL}" in _vllm_install_step(), (
        "the clone must honour TT_VLLM_REPO_URL instead of hardcoding the URL"
    )


def test_torchaudio_is_installed_and_pinned_to_the_installed_torch():
    """The empty target does not pull torchaudio, but vLLM imports it anyway.

    vllm/transformers_utils/processors/__init__.py imports funasr_processor
    unconditionally and that module imports torchaudio at module scope, so
    without it *every* architecture inspection fails with
    "ModuleNotFoundError: No module named 'torchaudio'".
    """
    step = _vllm_install_step()
    assert "torchaudio==" in step, "torchaudio must be installed explicitly"
    assert "TORCH_VERSION" in step, (
        "torchaudio must be pinned to the torch tt-metal already installed, so "
        "resolution cannot swap in a different torch build"
    )


def test_tt_metal_repo_url_is_overridable_and_defaults_upstream():
    """Model code for a bring-up can live only on a tt-metal fork.

    Qwen3-ASR's vLLM adapter (models/demos/audio/qwen3_asr/tt/generator_vllm.py)
    is not on the pinned upstream commit, so a hardcoded clone URL makes the
    server fail with
    "ModuleNotFoundError: No module named
    'models.demos.audio.qwen3_asr.tt.generator_vllm'".
    """
    text = DOCKERFILE.read_text()
    assert (
        "ARG TT_METAL_REPO_URL=https://github.com/tenstorrent-metal/tt-metal.git"
        in text
    ), "TT_METAL_REPO_URL must exist and keep the upstream default"
    assert "git clone ${TT_METAL_REPO_URL}" in _tt_metal_build_step()


def test_tt_metal_repo_url_arg_is_declared_late_to_preserve_cache():
    """Declaring the ARG up top would bust the cache of every preceding layer.

    tt-metal's C++ build takes hours, so the ARG must sit immediately before the
    step that uses it.
    """
    text = DOCKERFILE.read_text()
    arg_pos = text.index("ARG TT_METAL_REPO_URL=")
    metal_clone_pos = text.index("git clone ${TT_METAL_REPO_URL}")
    apt_pos = text.index("# Install only essential build dependencies")
    assert apt_pos < arg_pos < metal_clone_pos, (
        "ARG TT_METAL_REPO_URL must be declared just before the tt-metal clone, "
        "after the long cacheable layers"
    )
