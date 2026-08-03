# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Contract tests for the src-dev image repo-URL build args.

The src-dev Dockerfile clones tt-metal and vllm at build time. To build a
bring-up branch that is not yet merged upstream (e.g. a fork), the clone source
must be overridable via a build arg, while defaulting to the canonical upstream
repos so stock builds are unchanged. These static checks lock that contract in
so a future edit cannot re-hardcode the URLs.
"""
from pathlib import Path

from workflows.utils import get_repo_root_path

DOCKERFILE = (
    get_repo_root_path()
    / "vllm-tt-metal"
    / "vllm.tt-metal.src.dev.Dockerfile"
)
BUILD_SCRIPT = get_repo_root_path() / "scripts" / "build_single_docker.sh"

UPSTREAM_TT_METAL = "https://github.com/tenstorrent-metal/tt-metal.git"
UPSTREAM_TT_VLLM = "https://github.com/tenstorrent/vllm.git"


def _dockerfile_text():
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_declares_repo_url_args_defaulting_to_upstream():
    text = _dockerfile_text()
    assert f"ARG TT_METAL_REPO_URL={UPSTREAM_TT_METAL}" in text, (
        "TT_METAL_REPO_URL must be an ARG defaulting to the upstream tt-metal repo"
    )
    assert f"ARG TT_VLLM_REPO_URL={UPSTREAM_TT_VLLM}" in text, (
        "TT_VLLM_REPO_URL must be an ARG defaulting to the upstream vllm repo"
    )


def test_dockerfile_clones_via_the_repo_url_args():
    text = _dockerfile_text()
    assert "git clone ${TT_METAL_REPO_URL} ${TT_METAL_HOME}" in text
    assert "git clone ${TT_VLLM_REPO_URL} ${vllm_dir}" in text


def test_dockerfile_does_not_hardcode_the_clone_urls_anymore():
    text = _dockerfile_text()
    # The URLs may still appear as ARG defaults, but must not be used directly in
    # a `git clone <url>` command.
    assert f"git clone {UPSTREAM_TT_METAL}" not in text
    assert f"git clone {UPSTREAM_TT_VLLM}" not in text


def test_build_script_forwards_repo_url_build_args():
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "--tt-metal-repo-url" in text
    assert "--tt-vllm-repo-url" in text
    assert '--build-arg TT_METAL_REPO_URL="${TT_METAL_REPO_URL}"' in text
    assert '--build-arg TT_VLLM_REPO_URL="${TT_VLLM_REPO_URL}"' in text
