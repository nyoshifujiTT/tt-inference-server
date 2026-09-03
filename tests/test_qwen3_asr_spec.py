# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Spec invariants for the vLLM-served Qwen3-ASR bring-up."""

import os
import re

import pytest

from workflows.model_spec import load_templates_from_yaml, get_model_spec_map
from workflows.utils import get_repo_root_path
from workflows.workflow_types import InferenceEngine, ModelType


def _dev_specs():
    """Resolve the dev catalog regardless of MODEL_SPECS_ENV.

    Qwen3-ASR is a bring-up and lives only in workflows/model_specs/dev. The
    module-level MODEL_SPECS honours MODEL_SPECS_ENV, which defaults to prod, so
    importing it would make these tests depend on how the runner is invoked.
    Promotion to prod is the release process's job, not this bring-up's.
    """
    path = get_repo_root_path() / "workflows" / "model_specs" / "dev" / "audio_tts.yaml"
    return get_model_spec_map(load_templates_from_yaml(path, env="dev"))


MODEL_SPECS = _dev_specs()

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


def test_readme_pins_agree_with_the_runbook_tag():
    """The patched pins and the image tag the runbook runs must be the same.

    Dev specs carry no pins (and so no docker_image); the build gets them from
    the patch in the runbook. If that patch and the --override-docker-image tag
    drift apart, the reader builds one image and then starts another.
    """
    readme = _readme()
    metal, vllm = _patched_pins(readme)

    assert f"--build-metal-commit {metal}" in readme, (
        "the build command must use the tt_metal_commit the patch sets"
    )
    assert f"ubuntu-22.04-amd64:{metal}" in readme, (
        "the base-image bake command must tag the pinned tt-metal commit"
    )
    assert f"{metal}-{vllm}" in readme, (
        f"the runbook must start the image built from the pinned commits "
        f"({metal}-{vllm})"
    )


def _patched_pins(readme):
    """Return (tt_metal_commit, vllm_commit) as set by the runbook's patch."""
    metal = re.search(r'^\+  tt_metal_commit: "([0-9a-f]{7,})"', readme, re.M)
    vllm = re.search(r'^\+  vllm_commit: "([0-9a-f]{7,})"', readme, re.M)
    assert metal and vllm, "the patch must add both release pins"
    return metal.group(1), vllm.group(1)


def test_the_patched_pin_is_the_one_the_docs_build_from():
    """The pin names the tree the image is built from.

    It used to live in model_spec.py; upstream moved catalogs to YAML and the
    dev contract rejects pins, so it now reaches the build through the runbook
    patch. Wherever it lives, a stale value builds a tree the repo no longer
    tests.
    """
    readme = _readme()
    metal, _ = _patched_pins(readme)
    assert readme.count(metal) >= 3, (
        f"the pinned commit {metal} must appear in the patch, the bake tag and "
        "the build command; a partial update builds a different tree"
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
    "3b1b9ad",  # before the eval-side 16 kHz resample fix
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

    The Dockerfile resolves TT_VLLM_COMMIT_SHA_OR_TAG against
    tenstorrent/vllm-tt-plugin, so a value left over from the days of cloning
    the tenstorrent/vllm fork would fail at git checkout in the builder --
    exactly what scripts/release/README.md warns about.
    """
    _, vllm = _patched_pins(_readme())
    assert vllm not in SUPERSEDED_VLLM_FORK_COMMITS, (
        "vllm_commit still points at a tenstorrent/vllm fork commit; it must "
        "name a vllm-tt-plugin commit"
    )


# vLLM *fork* commits this spec pinned back when the dev image cloned
# tenstorrent/vllm. None of them exist in tenstorrent/vllm-tt-plugin.
SUPERSEDED_VLLM_FORK_COMMITS = (
    "e1a3825",  # fork upstream base
    "5e69638",  # fork bring-up branch head
    "2bcb717",  # plugin head before the upstream merge (vLLM 0.24.0)
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
    assert "the plugin pins" in readme, (
        "record that the fork's HF-config fix already ships in the vLLM the "
        "plugin pins; do not hardcode the version, which moves with upstream"
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
    # both clones point at forks now, so serving only tt-metal leaves the build
    # failing on the plugin clone hours in
    assert "/tmp/ttmetal-src.git" in readme
    assert "/tmp/vllmttplugin-src.git" in readme, (
        "the plugin fork needs the same loopback treatment as tt-metal"
    )
    assert "git ls-remote" in readme, (
        "reachability must be checkable before a multi-hour build"
    )
    assert "Push the branches and drop this" in readme, (
        "the workaround must be marked as temporary, not as the delivered recipe"
    )


def test_the_current_pin_is_not_itself_listed_as_superseded():
    """Guard the bookkeeping: bumping the pin must also retire the old entry."""
    metal, _ = _patched_pins(_readme())
    assert metal not in SUPERSEDED_TT_METAL_COMMITS, (
        f"the active pin {metal} is listed as superseded"
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


def test_the_readme_describes_the_merged_upstream_relationship():
    """Both server and plugin have upstream merged; saying otherwise misleads.

    Earlier revisions of this section described the branch as trailing main,
    then as deliberately holding the plugin back from vLLM 0.26.0. Both stopped
    being true when the merges landed, and a reader would either look for
    deleted directories or re-do an upgrade that is already done.

    What remains genuinely unfollowed is tt-metal, and the reason has to stay.
    """
    readme = _readme()
    assert "Relationship to upstream" in readme
    # stale claims must not come back
    assert "725 commits" not in readme
    assert "no `workflows/model_specs/` at all" not in readme
    assert "stays on its current base" not in readme, (
        "the plugin no longer holds back from upstream"
    )
    # the one real exception, with its reason
    assert "upstream/yito/qwen3_asr_pr" in readme
    # and the executor still has to be justified, since upstream ships none
    assert "TTUniProcExecutor" in readme and "0.26.0" in readme


def test_the_readme_backs_the_vllm_upgrade_with_measurements():
    """Moving two vLLM releases is only safe if it was measured.

    The plugin merge took the installed vLLM from 0.24.0 to 0.26.0 and dropped
    three of our commits. Without the numbers a reader cannot tell whether that
    preserved behaviour.
    """
    readme = _readme()
    section = readme[readme.index("Relationship to upstream") :][:2500]
    assert "0.1002" in section and "0.1668" in section, (
        "both corpus CERs must be quoted across the upgrade"
    )
    assert "12.61" in section, "the post-upgrade throughput must be recorded"


def test_the_readme_does_not_claim_the_old_layout():
    """Directories the merge removed must not be presented as current."""
    readme = _readme()
    for gone in ("`evals/`", "`benchmarking/`", "workflows/model_spec.py`"):
        assert gone not in readme, (
            f"{gone} no longer exists after the upstream merge"
        )


def test_the_readme_says_the_fork_is_deprecated():
    """The fork is not an alternative route; it is retired upstream.

    tenstorrent/vllm's README says "This repository is deprecated. Do not use
    it", TT issues are redirected to vllm-tt-plugin, and tt-inference-server
    switched off it in PR #4907 (merged into this branch). An earlier revision
    of this file presented fork and plugin as two supported routes, which would
    send a reader to a repository that is scheduled for archival.
    """
    readme = _readme()
    assert "deprecated" in readme, "the fork's status must be stated"
    assert "#4907" in readme, "cite the upstream switch this branch merged"
    assert "one supported route" in readme, (
        "the README must not present the fork as a live alternative"
    )
    assert "Both vLLM routes are still supported" not in readme


def test_the_readme_backs_the_switch_with_measurements():
    """"Nothing is lost" is a claim about accuracy, so it needs numbers.

    Without them the next reader cannot tell whether the fork was dropped after
    verification or on reasoning alone.
    """
    readme = _readme()
    assert "0.1002" in readme, "the TED CER measured after the switch must be quoted"
    assert "0.1668" in readme, "the MagicHub CER measured after the switch must be quoted"


def test_the_readme_keeps_the_migration_evidence():
    """"Nothing was lost" is a claim about accuracy, so it needs both columns.

    The fork run is kept as the record that the move preserved output. Quoting
    only the plugin numbers would leave the claim uncheckable.
    """
    readme = _readme()
    section = readme[readme.index("Evidence that the move changed no output") :][:2500]
    assert "0.1002" in section and "0.1668" in section, (
        "both corpora must be shown, since one alone would not show parity"
    )
    assert "12.73" in section and "11.97" in section, (
        "both routes' measured throughput must be shown"
    )
    assert "not an invitation to run the fork" in section, (
        "the table must not read as a supported configuration"
    )


def test_the_prod_catalog_has_no_committed_qwen3_asr_entry():
    """The prod entry exists only inside the runbook patch.

    Prod is written by promote_dev_spec_to_prod.py for leaves listed in the CI
    config; a bring-up is not one, so committing an entry there would forge a
    release artifact. The patch adds it for the build and reverts it.
    """
    prod = (
        get_repo_root_path() / "workflows" / "model_specs" / "prod" / "audio_tts.yaml"
    ).read_text()
    assert "Qwen3-ASR" not in prod, (
        "the prod catalog must stay free of the bring-up entry; it belongs in "
        "the temporary build patch only"
    )


def test_the_readme_explains_why_prod_is_patched():
    """Editing prod needs a stated reason, or it reads as a violation."""
    readme = _readme()
    assert "promote_dev_spec_to_prod.py" in readme, (
        "name the tool that normally owns prod entries"
    )
    assert "models-ci-config.json" in readme, (
        "say why promotion cannot produce this entry"
    )
    assert "never committed" in readme
    assert "git checkout" in readme and "prod/audio_tts.yaml" in readme, (
        "the revert step must cover the prod file too"
    )


def test_the_runbook_image_tag_carries_the_repo_version():
    """The tag the runbook starts must be the one the build produces.

    build_docker_images.py prefixes the image tag with the repo VERSION, so
    merging upstream (which bumped VERSION) silently invalidates a hardcoded
    tag: the reader builds 0.21.0-... and then tries to start 0.13.0-... .
    """
    version = (get_repo_root_path() / "VERSION").read_text().strip()
    readme = _readme()
    metal, vllm = _patched_pins(readme)
    expected = f"{version}-{metal}-{vllm}"
    assert expected in readme, (
        f"the runbook must reference the image tag the build produces "
        f"({expected}); a stale VERSION prefix points at an image that was "
        "never built"
    )


def test_the_spec_emits_additional_config_not_override_tt_config():
    """TT plugin config must reach vLLM as --additional-config.

    ``override_tt_config`` is not a vLLM CLI flag; emitting it into vllm_args
    makes the arg parser reject an unrecognized ``--override_tt_config``. The
    older spec format did emit it, and this bring-up carried a fold-in step in
    run_vllm_api_server.py to compensate. Upstream now serializes
    ``additional_config`` directly, so that step was deleted -- this test fails
    if the old shape returns and the deletion silently breaks serving.
    """
    spec_src = (
        get_repo_root_path() / "workflows" / "model_spec.py"
    ).read_text()
    assert '"additional_config": json.dumps({"tt": self.override_tt_config})' in spec_src
    assert '"override_tt_config": json.dumps(' not in spec_src, (
        "override_tt_config must not be serialized into vllm_args; it is not a "
        "vLLM CLI flag"
    )

    server_src = (
        get_repo_root_path() / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"
    ).read_text()
    assert "override_tt_config" not in server_src, (
        "the fold-in step is dead code once the spec emits additional_config"
    )
