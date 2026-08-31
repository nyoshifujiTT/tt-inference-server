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


# Every tt-metal commit this repo has pinned and then moved past. A superseded
# pin builds a docker image from a tree the repo no longer tests, so the served
# model silently differs from what CI checked. Append (never remove) an entry
# when bumping the pin.
SUPERSEDED_TT_METAL_COMMITS = (
    "97b36e1",  # pre-bring-up base
    "d53d8d7",  # decode trace default ON
    "ddb7ace",  # head before the rebase onto upstream/yito/qwen3_asr_pr
    "986aad1",  # pre-rebase branch head
)


def test_no_superseded_commit_is_referenced_anywhere():
    root = os.path.join(os.path.dirname(__file__), "..")
    for rel in ("workflows/model_spec.py", "scripts/qwen3_asr/README.md"):
        text = open(os.path.join(root, rel)).read()
        for stale in SUPERSEDED_TT_METAL_COMMITS:
            assert stale not in text, (
                f"{rel} still references the superseded tt-metal commit {stale}"
            )


def test_the_readme_states_when_the_pin_may_lag_the_head():
    """The pin names the BUILT tree, so test-only commits need no bump.

    Without this written down the next reader either rebuilds for ~7 h on a
    test-only commit, or bumps the pin without rebuilding and ships an image
    that does not correspond to the pinned tree.
    """
    readme = open(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "qwen3_asr", "README.md")
    ).read()
    assert "Why the pin may lag the branch head" in readme
    # the documented check must be the one that proves there is no runtime diff
    assert "git diff --name-only" in readme
    assert "grep -v '/tests/'" in readme


BRING_UP_BRANCH = "nyoshifujiTT/qwen3-asr-17b_p150x1"


def _readme():
    return open(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "qwen3_asr", "README.md")
    ).read()


def test_the_readme_names_the_branch_the_forks_must_carry():
    """The clone URLs are useless without knowing which branch holds the pins."""
    readme = _readme()
    assert BRING_UP_BRANCH in readme, "the bring-up branch name must be documented"
    for repo in ("nyoshifujiTT/tt-metal", "nyoshifujiTT/vllm"):
        assert repo in readme, f"{repo} must be listed with its branch"


def test_the_readme_explains_the_vllm_fork_clone():
    """This base still clones vLLM itself; upstream clones only the plugin.

    Without the note, the next reader compares against upstream, sees no vLLM
    clone there, and assumes vllm_commit names a vllm-tt-plugin commit.
    """
    readme = _readme()
    assert "Why this clones a vLLM fork at all" in readme
    assert "vllm-tt-plugin" in readme, "the upstream layout must be contrasted"


def test_the_readme_documents_the_unpushed_branch_path():
    """The recipe must not imply the forks already carry the pinned commits.

    While the bring-up branch is local-only the build cannot clone from GitHub,
    so the actual images were built against a git daemon on the docker host. A
    README that only shows the fork URLs hides that, and the next reader gets a
    clone failure with no idea why.
    """
    readme = open(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "qwen3_asr", "README.md")
    ).read()
    assert "If the branch is not pushed yet" in readme
    assert "git daemon" in readme, "the loopback-serving workaround must be spelled out"
    assert "git://172.17.0.1:9418" in readme, "the URL that was actually used must be shown"
    assert "Push the branch and drop this step" in readme, (
        "the workaround must be marked as temporary, not as the delivered recipe"
    )


def test_the_current_pin_is_not_itself_listed_as_superseded():
    """Guard the bookkeeping: bumping the pin must also retire the old entry.

    Without this, someone could add the NEW commit to the list above and the
    test would then demand the spec not reference the very commit it pins.
    """
    import re

    spec_src = open(os.path.join(os.path.dirname(__file__), "..", "workflows", "model_spec.py")).read()
    anchor = spec_src.index("Qwen3-ASR-1.7B-JA")
    window = spec_src[anchor : anchor + 4000]
    pinned = re.search(r'tt_metal_commit="([0-9a-f]{7,})"', window).group(1)
    assert pinned not in SUPERSEDED_TT_METAL_COMMITS, (
        f"the active pin {pinned} is listed as superseded"
    )
