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


def test_vllm_commit_pins_a_plugin_commit_not_a_fork_commit():
    """vllm_commit names a vllm-tt-plugin commit, as it does upstream.

    The dev image clones tenstorrent/vllm-tt-plugin and lets its
    docs/install-vllm-tt.sh pull the vLLM release it pins, so a value left over
    from the days of cloning the tenstorrent/vllm fork would check out a SHA
    that does not exist in the plugin repo and fail the build.
    """
    import re

    spec_src = open(
        os.path.join(os.path.dirname(__file__), "..", "workflows", "model_spec.py")
    ).read()
    anchor = spec_src.index("Qwen3-ASR-1.7B-JA")
    window = spec_src[anchor : anchor + 4000]
    match = re.search(r'vllm_commit="([0-9a-f]{7,})"', window)
    assert match, "the Qwen3-ASR spec must pin a vllm_commit"
    assert match.group(1) not in SUPERSEDED_VLLM_FORK_COMMITS, (
        "vllm_commit still points at a tenstorrent/vllm fork commit; it must "
        "name a vllm-tt-plugin commit now that the image clones the plugin"
    )
    # the comment must say what the field means, or the next reader repeats the
    # mistake the rename does not prevent
    assert "vllm-tt-plugin* commit" in window or "vllm-tt-plugin commit" in window


# vLLM *fork* commits this spec pinned back when the dev image cloned
# tenstorrent/vllm. None of them exist in tenstorrent/vllm-tt-plugin.
SUPERSEDED_VLLM_FORK_COMMITS = (
    "e1a3825",  # fork upstream base
    "5e69638",  # fork bring-up branch head
)


def test_no_superseded_vllm_fork_commit_is_referenced_anywhere():
    root = os.path.join(os.path.dirname(__file__), "..")
    for rel in ("workflows/model_spec.py", "scripts/qwen3_asr/README.md"):
        text = open(os.path.join(root, rel)).read()
        for stale in SUPERSEDED_VLLM_FORK_COMMITS:
            assert stale not in text, (
                f"{rel} still references the vLLM fork commit {stale}; the "
                "image no longer clones that repo"
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
    for repo in ("nyoshifujiTT/tt-metal", "nyoshifujiTT/vllm-tt-plugin"):
        assert repo in readme, f"{repo} must be listed with its branch"


def test_the_readme_says_what_vllm_commit_names():
    """The field name says vLLM but the value is a plugin commit.

    Without the note the next reader pins a tenstorrent/vllm SHA, which does
    not exist in the plugin repo, and the build fails at `git checkout`.
    """
    readme = _readme()
    assert "What `vllm_commit` names" in readme
    assert "vllm-tt-plugin" in readme, "the repo actually cloned must be named"
    assert "vllm==0.24.0" in readme, (
        "record that the fork's HF-config fix already ships in the pinned vLLM"
    )

def test_the_readme_says_which_clip_to_sanity_check_with():
    """A bare "clip.wav" leaves the reader to grab any file they can find.

    During bring-up that meant synthetic fixtures with no reference transcript
    got picked up by others as if they were sanity clips.
    """
    readme = _readme()
    assert "The clip to check with" in readme
    assert "google/fleurs" in readme, "the clip must be fetchable, not copied around"
    assert "ja_jp" in readme and "test" in readme, "the exact split must be pinned"
    # the reference transcript must be present, or the output cannot be judged
    assert "インターネットで" in readme, "the reference transcript must be quoted"


def test_the_readme_does_not_overstate_the_16khz_requirement():
    """16 kHz is where the mel front-end is calibrated, not an API constraint.

    Stating it as an input requirement makes callers build conversion they do
    not need, and hides that the served path already resamples and downmixes
    (measured: 44.1 kHz mono/stereo and 16 kHz stereo all return the golden
    transcript).
    """
    readme = _readme()
    assert "preprocessor_config.json" in readme, "cite where 16 kHz comes from"
    assert "not a requirement on" in readme.replace("*", ""), (
        "the README must say the client is not required to convert"
    )
    assert "44.1 kHz" in readme, "the measurement that proves it must be recorded"


def test_the_readme_rejects_the_synthetic_fixtures_for_accuracy():
    """ja_words.wav / test15s.wav have no reference transcript.

    They were made during bring-up (a word concatenation for TT-vs-CPU decoder
    comparison, and the looped-English warmup waveform). Judging accuracy with
    either is meaningless, so the README has to say so explicitly.
    """
    readme = _readme()
    for fixture in ("ja_words.wav", "test15s.wav"):
        assert fixture in readme, f"{fixture} must be called out"
    assert "Do not use" in readme
    assert "QWEN3ASR_WARMUP_WAV" in readme, "say what test15s.wav actually is"


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
    assert "If a branch is not pushed yet" in readme
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


def test_the_readme_patch_applies_to_the_committed_dockerfile():
    """A runbook patch that does not apply is worse than no runbook.

    An earlier bring-up shipped a patch whose line numbers had drifted after an
    upstream merge, so the documented build failed with "patch does not apply".
    Extract the block the README tells the reader to pipe into `git apply` and
    check it against the tree.
    """
    import re
    import subprocess

    readme = _readme()
    match = re.search(r"git apply <<'PATCH'\n(.*?)\nPATCH\n", readme, re.DOTALL)
    assert match, "the README must carry the clone-URL patch as a git apply block"
    patch = match.group(1) + "\n"

    root = os.path.join(os.path.dirname(__file__), "..")
    result = subprocess.run(
        ["git", "apply", "--check", "-"],
        input=patch,
        text=True,
        cwd=root,
        capture_output=True,
    )
    assert result.returncode == 0, (
        f"the README patch does not apply to the committed tree: {result.stderr}"
    )


def test_the_readme_patch_redirects_both_clones_to_the_forks():
    """Either half missing produces an image that cannot serve the model."""
    readme = _readme()
    assert "nyoshifujiTT/tt-metal.git" in readme, (
        "without the tt-metal half the image lacks the vLLM adapter"
    )
    assert "nyoshifujiTT/vllm-tt-plugin.git" in readme, (
        "without the plugin half the TT adapter is never registered"
    )


def test_the_readme_says_the_patch_is_the_only_one():
    """The pins are committed source; nothing under workflows/ is touched."""
    readme = _readme()
    assert "The one manual patch" in readme
    assert "git checkout vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile" in readme, (
        "the patch must be restored after the build, as PR#4837 does"
    )
    assert "no build arg" in readme, (
        "record that clone-URL build args were withdrawn, so they are not "
        "reintroduced as a shortcut"
    )


def test_the_readme_records_why_the_tree_is_not_on_upstream_main():
    """The Dockerfile matches main; the rest of the tree does not.

    Without this written down, the next reader either assumes the whole tree is
    current, or re-opens "should we rebase onto main" without the numbers. It
    also has to say why the pins sit in model_spec.py, which would be wrong on
    main where prod specs are yaml and must not be edited.
    """
    readme = _readme()
    assert "What this tree does *not* follow upstream on" in readme
    # the reorganisation is the actual cost, so name what moved
    assert "llm_module" in readme and "report_module" in readme
    assert "workflows/model_specs" in readme, (
        "say that main splits the specs into yaml, which is why the inline pin "
        "here is not a licence to edit prod specs there"
    )
