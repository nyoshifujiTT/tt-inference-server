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
        r"RUN /bin/bash -c \"git clone https://github\.com/tenstorrent/vllm\.git.*?\"\n",
        text,
        re.DOTALL,
    )
    assert match, "vLLM install RUN step not found in dev Dockerfile"
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
