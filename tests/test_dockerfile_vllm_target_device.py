# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Guards how the dev image obtains vLLM and the TT platform.

The image installs the standalone ``tenstorrent/vllm-tt-plugin``, which owns the
vLLM version pin and its dependency overrides in ``docs/install-vllm-tt.sh``.
``TT_VLLM_COMMIT_SHA_OR_TAG`` therefore names a *plugin* commit.

An earlier bring-up shape cloned the ``tenstorrent/vllm`` fork instead and
installed its bundled ``plugins/vllm-tt-plugin``. These tests keep that shape
from creeping back: the fork carried no Qwen3-ASR change that the standalone
plugin lacks, and restating the install here would duplicate (and drift from)
the pin the plugin owns.
"""

import re

from workflows.utils import get_repo_root_path

DOCKERFILE = (
    get_repo_root_path() / "vllm-tt-metal" / "vllm.tt-metal.src.dev.Dockerfile"
)


def _vllm_install_step() -> str:
    text = DOCKERFILE.read_text()
    match = re.search(
        r"RUN /bin/bash -c \"git clone "
        r"https://github\.com/tenstorrent/vllm-tt-plugin\.git.*?\"\n",
        text,
        re.DOTALL,
    )
    assert match, "vllm-tt-plugin install RUN step not found in dev Dockerfile"
    return match.group(0)


def _tt_metal_build_step() -> str:
    text = DOCKERFILE.read_text()
    match = re.search(
        r"RUN /bin/bash -c \"git clone (?:--depth 1 )?"
        r"https://github\.com/tenstorrent-metal/tt-metal\.git.*?\"\n",
        text,
        re.DOTALL,
    )
    assert match, "tt-metal build RUN step not found in dev Dockerfile"
    return match.group(0)


def test_vllm_install_is_delegated_to_the_plugin_script():
    """The plugin owns the vLLM pin; the Dockerfile must not restate it.

    ``docs/install-vllm-tt.sh`` installs ``vllm==0.24.0`` with
    ``VLLM_TARGET_DEVICE=empty`` and removes the unusable torchaudio wheel.
    Spelling any of that out here would let the two drift apart.
    """
    step = _vllm_install_step()
    assert "source docs/install-vllm-tt.sh" in step, (
        "the vLLM install must be delegated to the plugin's install script"
    )
    tail = step.split("install-vllm-tt.sh", 1)[1]
    assert "uv pip install" not in tail, (
        "nothing may be installed after the plugin script; it owns the "
        "resolved environment"
    )


def test_runtime_target_device_stays_tt():
    text = DOCKERFILE.read_text()
    assert "VLLM_TARGET_DEVICE=tt" in text, (
        "the runtime ENV must keep VLLM_TARGET_DEVICE=tt so the TT platform is "
        "selected when the server runs"
    )


def test_the_tt_platform_comes_from_the_standalone_plugin():
    """The tt platform is advertised by the plugin's entry point.

    Without the plugin installed the container starts and then dies with
    "RuntimeError: Failed to infer device type".
    """
    step = _vllm_install_step()
    assert "vllm-tt-plugin.git ${vllm_tt_plugin_dir}" in step, (
        "the standalone tenstorrent/vllm-tt-plugin must be cloned"
    )
    assert "git checkout ${TT_VLLM_COMMIT_SHA_OR_TAG}" in step, (
        "TT_VLLM_COMMIT_SHA_OR_TAG must pin the plugin commit"
    )


def test_the_vllm_fork_is_not_cloned():
    """The fork shape must not come back.

    Every Qwen3-ASR change that lived on the fork branch has an equivalent on
    the standalone plugin, and the fork's HF-config fix ships in the
    vllm==0.24.0 release the plugin pins.
    """
    text = DOCKERFILE.read_text()
    assert "TT_VLLM_REPO_URL" not in text, (
        "no build arg may redirect a vLLM clone; the plugin owns the vLLM pin"
    )
    assert "tenstorrent/vllm.git" not in text, "the vLLM fork must not be cloned"
    assert "-e plugins/vllm-tt-plugin" not in text, (
        "the fork-bundled plugin copy must not be installed"
    )


def test_torchaudio_is_not_reinstalled_behind_the_plugin_script():
    """The plugin script uninstalls torchaudio on purpose.

    Its CUDA wheel cannot load next to the CPU torch tt-metal installs, and
    transformers>=5.12 imports it if it is merely present. An earlier shape
    installed it explicitly because vLLM used to import funasr_processor (and
    thus torchaudio) unconditionally; vLLM 0.24.0 imports processors lazily via
    ``__getattr__``, so that reason is gone.
    """
    text = DOCKERFILE.read_text()
    assert "torchaudio" not in _vllm_install_step(), (
        "torchaudio must not be reinstalled after install-vllm-tt.sh removed it"
    )
    assert "TORCH_VERSION" not in text


def test_the_plugin_source_tree_is_copied_into_the_runtime_stage():
    """The plugin is an editable install, so its tree must survive the copy.

    It has to land at the same absolute path as in the builder or the .pth link
    dangles and ``import vllm_tt_plugin`` fails at runtime.
    """
    text = DOCKERFILE.read_text()
    assert "${vllm_tt_plugin_dir} ${vllm_tt_plugin_dir}" in text, (
        "the vllm-tt-plugin tree must be COPYed to the same path"
    )
    assert "${vllm_dir} ${vllm_dir}" not in text, (
        "the fork's vLLM tree must no longer be copied"
    )


def test_no_clone_url_is_a_build_arg():
    """Redirecting a clone belongs in a temporary patch, not a build arg.

    A bring-up whose commits are not upstream yet follows the recipe PR#4837
    established: ``git apply`` the clone-URL line, build, then ``git checkout``
    to restore it. Build args for this were added during bring-up and
    withdrawn -- they let an image that came from a fork claim to have come
    from the committed Dockerfile.
    """
    text = DOCKERFILE.read_text()
    for arg in ("TT_METAL_REPO_URL", "TT_VLLM_REPO_URL"):
        assert arg not in text, f"{arg} must not exist; patch the clone instead"


def test_the_clone_urls_are_the_upstream_ones():
    """The committed Dockerfile must always describe an upstream build."""
    step = _tt_metal_build_step()
    assert "https://github.com/tenstorrent-metal/tt-metal.git" in step, (
        "tt-metal must be cloned from upstream in the committed file"
    )
    # upstream clones shallow and then fetches just the pinned commit: a
    # full-history clone has taken over an hour on CI and dropped with
    # "fatal: early EOF"
    assert "--depth 1" in step, "the shallow clone must not be undone"
    assert (
        "git clone https://github.com/tenstorrent/vllm-tt-plugin.git"
        in _vllm_install_step()
    ), "the plugin must be cloned from upstream in the committed file"
