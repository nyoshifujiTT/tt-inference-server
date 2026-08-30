# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Spec invariants for the vLLM-served Qwen3-ASR bring-up."""

import os

import pytest

from workflows.model_spec import MODEL_SPECS
from workflows.utils import get_repo_root_path
from workflows.workflow_types import InferenceEngine, ModelType

ASR_SPEC_IDS = [
    "id_tt-vllm-plugin_Qwen3-ASR-1.7B_p150",
    "id_tt-vllm-plugin_Qwen3-ASR-1.7B-JA_p150",
]


@pytest.mark.parametrize("spec_id", ASR_SPEC_IDS)
def test_asr_spec_is_vllm_served_audio(spec_id):
    spec = MODEL_SPECS[spec_id]
    assert spec.model_type == ModelType.AUDIO
    assert spec.inference_engine == InferenceEngine.VLLM.value


@pytest.mark.parametrize("spec_id", ASR_SPEC_IDS)
def test_asr_spec_declares_builtin_warmup(spec_id):
    """Generic background trace capture must not run against this model.

    run_vllm_api_server skips the background trace capture when a spec declares
    has_builtin_warmup. Without it the capture drives /v1/completions with
    synthetic text prompts against a transcription-only model while the
    adapter's own decode trace is already active, and the first real
    transcription then never completes.
    """
    assert MODEL_SPECS[spec_id].has_builtin_warmup is True


def test_readme_docker_image_tag_matches_the_spec():
    """The runbook's --override-docker-image must be the tag the spec resolves to.

    The tag encodes the pinned tt-metal and vLLM commits, so a spec bump that
    leaves the README behind sends people to an image that cannot serve the
    model.
    """
    readme = (
        get_repo_root_path() / "scripts" / "qwen3_asr" / "README.md"
    ).read_text()
    spec = MODEL_SPECS["id_tt-vllm-plugin_Qwen3-ASR-1.7B-JA_p150"]
    # the runbook builds the dev image; the spec names the release one
    dev_image = spec.docker_image.replace("-release-", "-dev-")
    _, _, version_tag = dev_image.partition(":")
    _, metal_commit, vllm_commit = version_tag.split("-")

    assert f"{metal_commit}-{vllm_commit}" in readme, (
        f"README must reference the image tag for the pinned commits "
        f"({metal_commit}-{vllm_commit})"
    )
    assert f"--build-metal-commit {metal_commit}" in readme
    assert f"ubuntu-22.04-amd64:{metal_commit}" in readme, (
        "the base-image bake command must tag the pinned tt-metal commit"
    )


def test_tt_metal_commit_matches_the_rebased_bring_up_head():
    """The pinned tt-metal commit must be the CURRENT bring-up head.

    The bring-up branch was rebased onto upstream/yito/qwen3_asr_pr. A spec left
    pinned to a pre-rebase commit would build a docker image from a tree that no
    longer exists on the branch, so the served model would silently differ from
    what the repo tests.
    """
    import re

    spec_src = open(os.path.join(os.path.dirname(__file__), "..", "workflows", "model_spec.py")).read()
    # model_spec.py pins a commit for every model; scope the search to the
    # Qwen3-ASR entry so another model's pin cannot satisfy this test.
    anchor = spec_src.index("Qwen3-ASR-1.7B-JA")
    window = spec_src[anchor : anchor + 4000]
    match = re.search(r'tt_metal_commit="([0-9a-f]{7,})"', window)
    assert match, "the Qwen3-ASR spec must pin a tt_metal_commit"
    pinned = match.group(1)

    readme = open(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "qwen3_asr", "README.md")
    ).read()
    assert pinned in readme, (
        f"the documented build steps must use the pinned commit ({pinned}); "
        "a stale README sends the reader to build a different tree"
    )


def test_no_pre_rebase_commit_is_referenced_anywhere():
    # ddb7ace was the head before the rebase onto upstream/yito/qwen3_asr_pr.
    root = os.path.join(os.path.dirname(__file__), "..")
    for rel in ("workflows/model_spec.py", "scripts/qwen3_asr/README.md"):
        text = open(os.path.join(root, rel)).read()
        assert "ddb7ace" not in text, f"{rel} still references the pre-rebase commit"
