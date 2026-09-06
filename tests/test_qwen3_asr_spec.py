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
    # Scope to the runbook's own patch block. The section above it quotes the
    # original PR #4837 recipe, which also contains a "+  vllm_commit:" line;
    # searching the whole file picked that up and reported the wrong pin.
    block = readme[readme.index("git apply <<'PATCH'") : readme.index("\nPATCH\n")]
    metal = re.search(r'^\+  tt_metal_commit: "([0-9a-f]{7,})"', block, re.M)
    vllm = re.search(r'^\+  vllm_commit: "([0-9a-f]{7,})"', block, re.M)
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
    # the tt-metal filter also drops the golden tooling and offline eval, which
    # ship in the image but are not reachable from tt/ -- see the pin-form tests
    assert "grep -vE '/tests/|" in readme


def test_the_readme_does_not_claim_tests_are_absent_from_the_image():
    """Both clones are whole trees, so tests/ IS inside the image.

    The section used to justify a lagging pin with "never copied into the
    image". Checked on a running container, that is false:
    tt-metal .../qwen3_asr/tests has 18 files and
    /home/container_app_user/vllm-tt-plugin/tests has 28 .py files. A reader who
    believed the old wording would conclude any file present in the image
    requires a pin bump, which is the wrong rule.
    """
    readme = _readme()
    assert "never copied into the image" not in readme
    # the real reason is the import graph, and it has to be stated
    assert "src/vllm_tt_plugin" in readme
    assert "import graph" in readme


def test_the_readme_gives_the_no_runtime_diff_check_for_both_pins():
    """vllm_commit can lag too, and its tests live at a different path.

    Only the tt-metal form was documented, so a plugin test-only commit had no
    stated way to be cleared; the tt-metal filter ('/tests/') does not match the
    plugin layout ('tests/...' at the repo root).
    """
    readme = _readme()
    assert "grep -vE '/tests/|" in readme, "tt-metal form"
    assert "grep -v '^tests/'" in readme, "vllm-tt-plugin form"


def _readme_row(readme, leading_cell):
    for line in readme.splitlines():
        if line.startswith(f"| {leading_cell}"):
            return line
    raise AssertionError(f"no table row for {leading_cell} in the README")


def test_the_readme_warns_that_the_snapshot_default_is_a_literal_sha():
    """SNAP's default hardcodes a revision, and the guard exits if it moves.

    The script has
      SNAP="${SNAP:-$HOME/.cache/.../snapshots/987bda16...}"
    with no glob, and the pre-flight loop requires every path to exist. So a
    re-download at a newer revision does not fall back to the new snapshot --
    it makes the supervisor exit with "missing path" before launching, which
    reads as a broken install rather than a stale default.
    """
    readme = _readme()
    supervisor = _supervisor()

    # the default really is a literal revision, or this warning is stale
    assert "snapshots/987bda16" in supervisor, (
        "SNAP's default no longer pins a revision; update the README note"
    )
    assert '"$SNAP"' in supervisor, "SNAP must be covered by the missing-path guard"

    row = _readme_row(readme, "`SNAP`")
    assert "987bda16" in row, "show that the default names one revision"
    assert "override `SNAP`" in row, (
        "say what to do when the revision moves, not just that it can"
    )


def test_the_readme_ties_the_canary_clip_to_the_documented_one():
    """"the clip from above" was not checked to be that clip.

    Measured: $HOME/real_ja.wav and the runbook's clip are the same bytes,
    md5 3d43ec3ac2562231ec7c8c9ce4087ba4. Worth recording because the canary
    passes on any 200 with a "text" field, so a swapped file would keep the
    supervisor green while removing the one place a wrong transcript is
    visible.
    """
    readme = _readme()
    row = _readme_row(readme, "`CANARY_WAV`")
    assert "3d43ec3ac2562231ec7c8c9ce4087ba4" in row, (
        "identify the clip by hash, not by reference to another section"
    )
    # and be honest that the check itself does not verify the transcript
    assert "does not depend on the transcript" in row


def test_the_readme_and_supervisor_agree_on_the_startup_time():
    """The README said "~12 minutes"; the script said 7-12 and sized for 20.

    Timed on the delivery p150, /health turned 200 at 460 s -- 7.7 min. A
    single "~12" figure is wrong in both directions: it overstates a normal
    start, and it reads as a ceiling when the supervisor deliberately allows 20
    minutes because the observed range has an upper end it must not trip over.

    Pin the range and the measured point in both places, so a future timing
    that lands at one end does not get written up as the new single truth.
    """
    readme = _readme()
    supervisor = _supervisor()

    flat = " ".join(readme.split())
    assert "7-12 min" in flat, (
        "quote the cold-start range; a single figure is wrong at both ends"
    )
    assert "460 s" in flat, "record the timed measurement behind the range"

    # The fast end is not a fluke and must not be presented as the norm
    # either: it depends on the kernel cache surviving in a docker volume.
    assert "140 s" in flat, "record the warm-start measurement too"
    assert "kernel cache" in flat, (
        "name what decides which end you get, or a 140 s start looks wrong"
    )
    assert "volume_id_tt_vllm_plugin" in flat, (
        "identify the volume, so the cold case can be reproduced on purpose"
    )

    # the script's own budget must stay above the range, and say why
    assert "7-12 minutes" in supervisor
    assert "20 * 60" in supervisor, (
        "the budget has to exceed the range's upper end, not match a lucky run"
    )
    assert "460 s" in supervisor, (
        "carry the same measurement, so the two cannot drift apart silently"
    )


def test_the_readme_does_not_treat_a_printed_filename_as_a_verdict():
    """"If either prints anything, bump that pin" contradicts the next check.

    Run at the current pins, the tt-metal command is *not* silent -- it prints
    tt/generator_vllm.py, because a commit rewrote the ND-hang issue references
    inside a comment block. Read literally, the old sentence orders a ~7 h
    rebuild that cannot change a single executed byte; the comment-only check
    immediately below then says the opposite.

    The two are a sequence, not alternatives, and the README has to say so --
    with the file it actually prints, so the reader recognises the situation
    instead of assuming they have hit a new problem.
    """
    readme = _readme()
    body = readme[readme.index("Before leaving a pin behind its branch head") :]
    body = body[: body.index("The extra tt-metal exclusions")]

    assert "not yet a verdict" in body, (
        "a printed filename only starts the check; say so"
    )
    assert "both" in body, "state that a file must survive both checks"
    # the worked example, named, so the current output is recognisable
    assert "tt/generator_vllm.py" in body, (
        "name the file the check prints today, or its output looks like a fault"
    )
    assert "the pin stays" in body
    # and that the plugin side really is silent, which is the contrast
    assert "plugin command does print nothing" in body
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


def test_the_readme_clones_from_the_forks_only():
    """The recipe itself must clone from the forks, with no loopback detour.

    While the branches were local-only the images were built against a git
    daemon on the docker host, and the README documented that detour as part of
    the recipe. An image built from a daemon on one machine is not reproducible
    by anyone else, so the plumbing had to leave the instructions: a reader
    must not be told to set it up.

    The results section may still *state* that the current image was built that
    way -- that is a fact about what was run, not an instruction -- so this
    check is scoped to the build recipe rather than the whole file.
    """
    readme = _readme()
    recipe = readme[: readme.index("Measured on the delivery p150")]
    for leftover in (
        "If a branch is not pushed yet",
        "git daemon",
        "git://172.17.0.1:9418",
        "/tmp/ttmetal-src.git",
        "/tmp/vllmttplugin-src.git",
    ):
        assert leftover not in recipe, (
            f"{leftover!r} is loopback plumbing from before the branches were "
            "pushed; the recipe must clone from the forks"
        )

    # what must remain: both clones aimed at the pushed forks
    assert "nyoshifujiTT/tt-metal.git" in recipe
    assert "nyoshifujiTT/vllm-tt-plugin.git" in recipe


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


def test_the_readme_documents_how_to_measure():
    """A serving runbook that cannot be verified is half a runbook.

    Accuracy and throughput were the acceptance criteria for this bring-up, so
    the commands that produce them have to be written down -- otherwise the
    numbers quoted elsewhere in this file cannot be reproduced by the reader.
    """
    readme = _readme()
    assert "### 4. Eval and benchmark" in readme
    assert "asr_ja_eval.py" in readme, "the corpus CER harness must be named"
    assert "asr_openai_benchmark.py" in readme, "the throughput harness must be named"
    # the measured results, so a rerun can be compared against something
    assert "0.1002" in readme and "0.1668" in readme


def test_the_readme_gives_a_runnable_check_for_the_import_graph_claim():
    """"nothing the server loads imports those modules" was only reasoning.

    The pin-lag argument rests entirely on tests/ being outside the import
    graph, and that was supported by describing where packages resolve. It is
    directly observable instead: CPython writes __pycache__ only for modules it
    imports, so on a container that has served the corpus runs, tt/ carries
    .pyc files and tests/__pycache__ does not exist at all.

    Without the check in the README, the next person deciding whether to spend
    a ~7 h rebuild has an argument to weigh rather than a command to run.
    """
    readme = _readme()
    section = readme[readme.index("What makes a test-only commit safe") :]
    section = section[: section.index("\n## ")]
    assert "__pycache__" in section, (
        "name the artifact that proves import, not just the resolution rules"
    )
    # the contrast is the evidence: one exists, the other does not
    assert "No such file or directory" in section, (
        "show that tests/__pycache__ is absent; a listing of tt/ alone proves nothing"
    )
    assert "never imported" in section


def test_the_readme_reports_what_the_perf_probes_measured():
    """The runbook told you to run them and never said what they returned.

    Both probes are documented with their positional arguments, and the
    section promises "TTFT, prefill, decode TPS and TPS/user" -- but the
    results table listed only CER and rtfx. With no recorded numbers there is
    nothing to compare a rerun against, which is the whole reason the probes
    are committed rather than described.

    Measured, 60 requests at concurrency 4 on the FLEURS clip:
      non-streaming  TTFT 1.243 s, decode TPS/user 23.91, aggregate 41.45
      streaming      TTFT 1.338 s, decode TPS/user 22.22, aggregate 39.52
    """
    readme = _readme()
    section = readme[readme.index("Serving-level timings") :]
    section = section[: section.index("**How the current image was built.**")]

    # both probes' headline numbers, not just one column
    for value in ("1.243", "1.338", "23.91", "22.22", "41.45", "39.52"):
        assert value in section, (
            f"{value} was measured; record it or a rerun has no baseline"
        )
    assert "60 / 60" in section, "say how many requests the numbers came from"


def test_the_readme_explains_the_gap_between_the_two_probes():
    """Two columns that differ with no stated reason read as a contradiction.

    The streaming column is consistently the slower one, and one token lighter
    per request. Both are expected -- SSE framing and scheduling land on the
    client's clock but not on the decode counters, and the terminal chunk
    carries no new token -- but unexplained they look like one probe being
    wrong.
    """
    readme = _readme()
    section = readme[readme.index("Serving-level timings") :]
    section = section[: section.index("**How the current image was built.**")]
    assert "SSE framing" in section, "say why the client-side number is higher"
    assert "final chunk carries no new token" in section, (
        "explain the off-by-one in tokens per request"
    )
    # and a threshold, so "agrees" is not left to taste
    assert "tens of percent" in section, (
        "state how large a gap stops being framing overhead"
    )


def test_the_readme_says_why_the_upstream_audio_harness_is_not_used():
    """Otherwise the next reader re-discovers the 400 the hard way.

    test_module's generic audio path POSTs a JSON body and drops the /v1
    prefix, because it targets tt-media-server. vLLM's OpenAI-compatible
    endpoint wants multipart, and rejects that shape.
    """
    readme = _readme()
    section = readme[readme.index("### 4. Eval and benchmark") :]
    assert "_is_whisper" in section, "name the branch that excludes this model"
    assert "multipart" in section, "state what the endpoint actually accepts"
    assert "400" in section, "record the observed failure, not just the theory"


def test_the_readme_records_both_upstream_audio_failures():
    """The 400 alone suggests the /v1 prefix is the whole problem.

    Measured against the running server, the upstream shape fails twice over:

      POST /audio/transcriptions      (as eval_command builds it) -> 404
      POST /v1/audio/transcriptions   (prefix fixed by hand)      -> 400

    Recording only the 400 invites the next reader to "just add the /v1" and
    find the route exists but still rejects the body, because vLLM parses
    `file` as an upload rather than a base64 string. Both numbers, and the
    order they appear in, are what says the fix is a harness feature and not a
    one-line URL change.
    """
    readme = _readme()
    section = readme[readme.index("### 4. Eval and benchmark") :]
    section = section[: section.index("Corpus accuracy")]
    assert "404" in section, (
        "the path the harness actually builds 404s; record it, or the 400 reads "
        "as the only obstacle"
    )
    assert "400" in section
    # tie each code to the path that produces it, not just list both codes
    prefixless = section[section.index("404") - 200 : section.index("404")]
    assert "/audio/transcriptions" in prefixless and "no `/v1`" in prefixless, (
        "say which request 404s"
    )
    assert "not enough" in section, (
        "state that correcting the prefix does not make the harness work"
    )


def test_the_runbook_sets_the_dev_catalog_when_serving():
    """run.py resolves specs through MODEL_SPECS_ENV, which defaults to prod.

    Qwen3-ASR lives only in the dev catalog, so without this the documented
    command exits saying the model is unknown.
    """
    readme = _readme()
    run_section = readme[readme.index("### 3. Run") : readme.index("### 4. Eval")]
    assert "MODEL_SPECS_ENV=dev python3 run.py" in run_section, (
        "the serving command must select the dev catalog"
    )


def test_trace_mode_travels_in_override_tt_config_not_a_duplicate_flag():
    """TT plugin settings belong in override_tt_config, not a raw vllm_arg.

    The base spec always emits additional_config from override_tt_config.
    Setting "additional-config" directly in vllm_args does not replace that --
    the two keys differ by a hyphen -- so the server was launched with both
    --additional_config '{"tt": {}}' and --additional-config '{"tt":
    {"trace_mode": "decode_only"}}'. vLLM happened to honour the later one, so
    the trace mode was right by luck rather than by construction.
    """
    specs = _dev_specs()
    for spec_id in ASR_SPEC_IDS:
        args = specs[spec_id].device_model_spec.vllm_args
        assert "additional-config" not in args, (
            "hyphenated additional-config duplicates the generated "
            "additional_config; put TT settings in override_tt_config"
        )
        assert '"trace_mode": "decode_only"' in args["additional_config"], (
            "the decode-only trace mode must reach vLLM through the generated "
            "additional_config"
        )


def test_the_runbook_patch_carries_the_same_trace_mode_form_as_the_dev_spec():
    """The prod entry the runbook adds is what a release build actually reads.

    The dev catalog was moved to override_tt_config, but the temporary prod
    entry inside the runbook's git-apply patch kept the hyphenated
    "additional-config" vllm_arg. Anyone building from the runbook would then
    get the duplicated flag the dev spec was fixed to avoid, with the correct
    value surviving only because vLLM happens to take the later one.
    """
    readme = _readme()
    patch = readme[readme.index("git apply <<'PATCH'") : readme.index("\nPATCH\n")]
    assert "additional-config" not in patch, (
        "the runbook's prod entry must not set the hyphenated vllm_arg; it "
        "duplicates the additional_config the base spec generates"
    )
    assert "+      override_tt_config:" in patch
    assert "+        trace_mode: decode_only" in patch

def _supervisor():
    return (
        get_repo_root_path() / "scripts" / "qwen3_asr" / "asr_supervisor.sh"
    ).read_text()


def test_the_supervisor_launches_the_model_the_way_run_py_still_accepts():
    """The supervisor is the production restart path; a stale flag breaks it.

    It was written before the upstream merge and still used --device (renamed
    --tt-device), passed --vllm-dir (now reported as deprecated and ignored),
    and did not select the dev catalog -- so every relaunch would have exited
    saying the model is unknown, exactly when the service needed to come back.
    """
    sh = _supervisor()
    assert "--tt-device p150" in sh, "--device was renamed --tt-device"
    # " --device" with the leading space, so --tt-device does not match itself
    assert " --device " not in sh
    # the flag must not be *passed*; the comment explaining why may mention it
    launch = sh[sh.index("launch_server()") : sh.index("wait_healthy()")]
    passed_args = [ln for ln in launch.splitlines() if not ln.strip().startswith("#")]
    assert not any("--vllm-dir" in ln for ln in passed_args), (
        "run.py reports --vllm-dir as deprecated and ignored since the plugin "
        "switch"
    )
    # it must be on the launch line, not only in the comment that explains it:
    # run.py defaults to prod, where this model does not exist
    launch = sh[sh.index("launch_server()") : sh.index("wait_healthy()")]
    launch_code = [
        ln for ln in launch.splitlines() if not ln.strip().startswith("#")
    ]
    assert any("MODEL_SPECS_ENV=dev" in ln for ln in launch_code), (
        "the spec lives only in the dev catalog; run.py defaults to prod"
    )


def test_the_supervisor_canary_does_not_use_a_synthetic_fixture():
    """ja_words.wav has no reference transcript and is banned elsewhere here.

    Liveness only needs "did a transcription come back", but pointing at a
    scratch file invites the same misuse this repo already documented once.
    """
    sh = _supervisor()
    assert "ja_words.wav" not in sh
    assert "CANARY_WAV" in sh and "README.md" in sh, (
        "say where the canary clip comes from"
    )


def test_the_supervisor_paths_are_overridable_and_checked():
    """Hardcoded /data paths made this a no-op on the delivery host.

    Every path pointed at the original bring-up board's /data tree, which does
    not exist elsewhere, so the supervisor would have launched run.py with a
    nonexistent TT_METAL_HOME and venv and failed in a way that looks like a
    model problem rather than a configuration one.
    """
    sh = _supervisor()
    assert "/data/" not in sh, "no path may be pinned to the original board"
    for var in ("TTIS", "TT_METAL_HOME", "VENV", "SNAP", "CANARY_WAV"):
        assert f'{var}="${{{var}:-' in sh, f"{var} must be overridable"
    assert "missing path:" in sh, (
        "a missing prerequisite must be reported up front, not as a launch "
        "failure later"
    )


def test_the_systemd_unit_points_at_the_checked_out_script():
    """The unit ran a copy under /data, which the supervisor fix just retired.

    Pointing at a copy also lets the deployed script drift from the repo, which
    is how the stale run.py flags survived unnoticed for so long.
    """
    unit = (
        get_repo_root_path()
        / "scripts"
        / "qwen3_asr"
        / "qwen3asr-supervisor.service"
    ).read_text()
    assert "/data/" not in unit
    assert "scripts/qwen3_asr/asr_supervisor.sh" in unit, (
        "run the script from the checkout so it cannot drift from the repo"
    )


def test_the_supervisor_kills_the_engine_process_too():
    """pkill on run.py leaves the engine holding the device.

    vLLM runs its engine as a separate "VLLM::EngineCore" process. Killing only
    the run.py parent orphans it, and it keeps /dev/tenstorrent/* open -- every
    later launch then hangs in "Starting devices in cluster", and tt-smi -r does
    not help because the orphan reacquires the device after the reset. This was
    observed for real: a 10h-old orphan blocked three consecutive restarts.
    """
    sh = _supervisor()
    assert "VLLM::EngineCore" in sh, (
        "the engine process must be killed, not just its run.py parent"
    )
    assert "device_holders" in sh, (
        "verify the device is actually free before relaunching"
    )
    assert "kill -9" in sh, "escalate for anything that still holds the device"


def test_the_supervisor_finds_holders_that_lsof_cannot_see():
    """`lsof -t /dev/tenstorrent/*` misses a containerised holder entirely.

    Measured on this host while a --docker-server engine was serving: the host
    node is dev=5 inode=666 while the engine's fd resolves to dev=67 inode=13
    (the container's own node for the same chip), so

        sudo lsof -t /dev/tenstorrent/*   -> prints nothing, exit 1
        /proc/<pid>/fd/17                 -> /dev/tenstorrent/0

    The path form therefore reports the device free for precisely the holder
    that makes the next launch hang in "Starting devices in cluster", and the
    in_container filter downstream never gets the pid to spare.
    """
    sh = _supervisor()
    assert "\ndevice_holders() {" in sh, (
        "holder discovery must be a defined function, not an inline lsof"
    )
    # /proc walking is what sees a container's fd; lsof by path does not.
    assert "/proc/[0-9]*/fd" in sh, "walk /proc to see holders inside containers"
    assert "readlink" in sh, "resolve each fd to its target"
    # Ban the path form in *code*. The comment above device_holders quotes it
    # to explain why it was dropped, so a plain substring check on the whole
    # file would fail on the explanation rather than on a regression.
    code = "\n".join(
        line for line in sh.splitlines() if not line.lstrip().startswith("#")
    )
    assert "lsof -t /dev/tenstorrent" not in code, (
        "the path form silently reports containerised holders as absent"
    )
    # the measurement, so the next reader does not "simplify" it back to lsof
    assert "dev=67" in sh and "dev=5" in sh, (
        "record the two device nodes, or this looks like a stylistic choice"
    )


def test_the_supervisor_spares_containerised_servers():
    """A --docker-server engine looks identical in the host process table.

    The supervisor manages a --local-server run. A bare
    `pkill -f "VLLM::EngineCore"` also matches the engine inside a running
    container -- verified on this host, where the containerised engine appears
    as a plain "VLLM::EngineCore" pid whose cgroup is
    /system.slice/docker-<id>.scope. Killing it would take down an unrelated
    deployment, which is the exact accident this cleanup exists to prevent.
    """
    sh = _supervisor()
    # Require the definition, not just a mention: renaming the function away
    # leaves the call sites referencing a name that no longer exists, and a
    # substring check on "in_container" would still pass while the filter is
    # silently gone (bash treats the failed call as false, so every engine
    # would be killed).
    assert "\nin_container() {" in sh, (
        "the cleanup must define in_container to distinguish our processes "
        "from containerised ones"
    )
    assert "/docker-" in sh, "identify container processes by cgroup"
    # and it must actually be consulted on both paths
    assert sh.count("in_container ") >= 2, (
        "in_container must gate both the engine kill and the device-holder "
        "escalation"
    )
    # the engine kill must go through the filter, not be a bare pkill
    assert 'pkill -f "VLLM::EngineCore"' not in sh, (
        "a bare pkill on the engine pattern also kills containerised servers"
    )


def test_the_supervisor_recovery_checks_can_actually_fire():
    """The escalation path was unreachable, so recovery stopped at tt-smi -r.

    Three defects, all confirmed on the delivery host:
      - TTSMI defaulted to ~/ttsmi-venv/bin/tt-smi, which does not exist here
        (the binary is ~/ttvenv/bin/tt-smi, and also on PATH)
      - the "needs a power cycle" test grepped for "should be reset", a string
        this tt-smi build never prints (0 occurrences), so the ipmitool branch
        could never be taken
      - the post-reboot wait required /dev/tenstorrent/2, which does not exist
        on a single-board host, so the loop could never succeed
    """
    sh = _supervisor()
    assert "command -v tt-smi" in sh, (
        "resolve tt-smi from PATH rather than a venv that may not exist"
    )
    # the dead grep must be gone from the code; the comment explaining it may stay
    code = "\n".join(
        ln for ln in sh.splitlines() if not ln.strip().startswith("#")
    )
    assert "should be reset" not in code, (
        "that string is never printed by this tt-smi; the check was dead"
    )
    # the definition, not a mention: a call to a missing bash function is falsy,
    # so renaming it away would silently make every board look wedged
    assert "\ndevice_ok() {" in sh, (
        "health must be decided by something observable"
    )
    assert sh.count("device_ok") >= 3, (
        "device_ok must be consulted after tt-smi -r and again after the power "
        "cycle, not merely defined"
    )
    assert "/dev/tenstorrent/2" not in sh, (
        "a single-board p150 host only has /dev/tenstorrent/0"
    )
    assert "/dev/tenstorrent/0" in sh


def test_the_supervisor_waits_long_enough_for_startup():
    """A 5-minute budget guaranteed a false wedge on every launch.

    The server takes 7-12 minutes to reach /health 200 (weight load plus decode
    trace capture; the runbook quotes ~12, and a launch measured here took
    7m40s). The old wait_healthy gave 60 x 5s = 5 minutes, so it always timed
    out and the supervisor went straight to recover_device -- power-cycling a
    board that was merely still warming up, then repeating forever.
    """
    sh = _supervisor()
    start = sh.index("wait_healthy()")
    body = sh[start : sh.index("\n}", start)]
    assert "20 * 60" in body, (
        "the startup budget must exceed the measured 7-12 minute startup"
    )
    assert "seq 1 60" not in body, "the 5-minute loop must be gone"


def test_the_readme_scopes_the_supervisor_verification_claim():
    """"Verified" must not cover code that was later found broken.

    The end-to-end recovery demo ran on the original board, before the upstream
    merge. Auditing the script afterwards found five defects that would each
    have broken it on the delivery host, so presenting that demo as blanket
    verification would tell a reader the recovery path is proven here when only
    its parts have been exercised.
    """
    readme = _readme()
    assert "What has and has not been verified" in readme
    assert "not** re-verified" in readme, (
        "say plainly that the full wedge->power-cycle loop was not re-run here"
    )
    # the parts that *were* exercised must be listed, or the section is just a
    # disclaimer with nothing behind it
    for probe in ("device_ok", "in_container", "canary_ok", "TTSMI"):
        assert probe in readme, f"{probe} was exercised; say so"


def test_the_readme_documents_the_plugin_server_facing_tests():
    """tests/tt is the only per-request sampling coverage on this deployment.

    It was never mentioned, so nobody ran it: the ASR model answers
    /v1/completions, which is what those tests drive. Running it found two
    presence-penalty failures that are a property of this model, not a defect,
    and that distinction has to be written down or the next reader files a bug.
    """
    readme = _readme()
    assert "tests/tt" in readme
    assert "--tt-server-url" in readme and "--tt-model-name" in readme


def test_the_readme_explains_the_presence_penalty_failures():
    """presence subtracts at most 2.0 once; this model's top-2 gap is larger.

    frequency scales with occurrence count and repetition divides, so both do
    reorder the top token -- presence cannot. Keep the measured gap on record.
    """
    readme = _readme()
    assert "presence_penalty" in readme
    # The evidence has to be the measured logprobs, not just a prose range: an
    # "or" over the two let the sampled numbers be edited without failing.
    assert "3.5-5.8" in readme, "state the gap the penalty must overcome"
    for observed in ("-0.08", "-5.58"):
        assert observed in readme, (
            f"keep the measured top-2 logprobs ({observed}) on record; the range "
            "alone cannot be checked against a rerun"
        )
    # and the escape hatch for a clean run
    assert "--deselect" in readme
    assert "TestPresencePenalty::test_different_presence_penalties" in readme


def test_the_readme_accounts_for_the_skipped_plugin_test():
    """An unexplained skip reads as coverage nobody checked.

    tests/tt leaves one skip: test_all_vocab_logprobs asks for top_logprobs=-1
    and the server answers "Requested sample logprobs of 151936, which is
    greater than max allowed: 20". That is vLLM's max_logprobs default, left
    alone on purpose -- whole-vocabulary logprobs cost per token in proportion
    to the vocabulary and nothing in the transcription path wants them.
    """
    readme = _readme()
    assert "71 passed, 1 skipped, 2 deselected" in readme, (
        "record the full result, not just the passes"
    )
    assert "max_logprobs" in readme
    assert "greater than max allowed: 20" in readme, (
        "keep the server's own message, so the skip can be told from a failure"
    )


def test_the_readme_documents_every_supervisor_override():
    """The supervisor's knobs were only discoverable by reading the script.

    Every path it uses is overridable and pre-checked, but none of the variable
    names appeared in the runbook -- including that the checkout is `TTIS`,
    while the runbook itself exports `TT_INFERENCE_SERVER` for the same tree.
    Someone setting the runbook's name and expecting the service to follow gets
    the default instead.

    The list is derived from the script so a new knob fails until documented.
    """
    import re

    supervisor = (
        get_repo_root_path() / "scripts" / "qwen3_asr" / "asr_supervisor.sh"
    ).read_text()
    # assignments of the form VAR="${VAR:-default}" are the overridable ones
    knobs = set(re.findall(r'^([A-Z_]+)="\$\{\1:-', supervisor, re.M))
    assert knobs, "the supervisor does use ${VAR:-default}; keep this meaningful"

    readme = _readme()
    for knob in sorted(knobs):
        assert knob in readme, (
            f"{knob} is overridable in asr_supervisor.sh but undocumented; a "
            "reader cannot know it exists"
        )


def test_the_readme_warns_that_the_checkout_variable_is_named_differently():
    """TTIS vs TT_INFERENCE_SERVER is a silent-default trap."""
    readme = _readme()
    body = readme[readme.index("## Install") :]
    assert "TTIS" in body, "the Install section is where a deployer looks"
    # Both names appear in the file for unrelated reasons, so an "or" over that
    # let the warning itself be deleted. Require the contrast to be stated
    # where the knob is described.
    assert "TT_INFERENCE_SERVER" in body, (
        "the runbook's own name for the same tree must be contrasted here, or "
        "a deployer sets it and silently gets the default"
    )
    assert "Not**" in body or "not**" in body, (
        "state it as a warning, not as a passing mention"
    )


def test_the_readme_does_not_overstate_what_the_dockerfile_clones():
    """"clones only vllm-tt-plugin" is false read literally.

    The dev Dockerfile clones three repositories: tt-metal, vllm-tt-plugin and
    tt-smi. The sentence means "no second vLLM source" -- true and worth
    saying, since the field is named vllm_commit and used to point at the
    tenstorrent/vllm fork -- but a reader checking the Dockerfile finds three
    clones and stops trusting the section.
    """
    import re

    dockerfile = (
        get_repo_root_path() / "vllm-tt-metal" / "vllm.tt-metal.src.dev.Dockerfile"
    ).read_text()
    cloned = set(
        re.findall(r"git clone[^\"]*?github\.com/[^/]+/([a-zA-Z0-9._-]+)\.git", dockerfile)
    )
    assert cloned == {"tt-metal", "vllm-tt-plugin", "tt-smi"}, (
        f"the Dockerfile's clone set changed: {sorted(cloned)}; update the README"
    )

    readme = _readme()
    # the vLLM-scoped claim must be qualified, not left as a bare "only"
    assert "clones only\n`vllm-tt-plugin`" not in readme
    assert "There is no second vLLM source" in readme or (
        "no\nsecond vLLM source" in readme or "no second vLLM source" in readme
    ), "say what 'only' is scoped to"
    assert "three repositories in" in readme, (
        "name the real clone count so the claim can be checked"
    )


def test_the_readme_gives_a_cheap_wedge_check():
    """The no-hang claim rested on soaks nobody can rerun cheaply.

    A reader deciding whether decode tracing is safe on their board had only
    "we ran 900 requests" to go on. vLLM already exposes the answer:
    request_success_total is monotonic per engine process, so a stall is it
    ceasing to advance while num_requests_running stays non-zero.
    """
    readme = _readme()
    body = readme[readme.index("Scope note (important)") :]
    assert "vllm:request_success_total" in body
    assert "num_requests_running" in body, (
        "one counter alone cannot distinguish a stall from an idle server"
    )
    # The observed figure has to appear as the command's output line, not only
    # in the prose: the number occurs twice, so an "in body" check let the
    # sample output be blurred to "many" while the prose kept the digits.
    assert 'vllm:request_success_total{...,finished_reason="stop",...} 6675.0' in body, (
        "quote the counter line as the server prints it, so a rerun can be "
        "compared line for line"
    )


def test_the_readme_does_not_present_the_counter_as_a_number_to_match():
    """"compared line for line" is wrong for a per-process counter.

    6675 was a high-water mark on one long-lived engine. The counter resets
    with the process, so a healthy fresh server reads far lower -- 3064 after
    one pass of each suite, measured. Presenting 6675 as the reference invites
    reading a correct server as a regression.

    What is actually reproducible is the shape: error/abort at 0.0, stop
    advancing by exactly the requests issued, and no hang line in the log.
    """
    readme = _readme()
    body = readme[readme.index("Scope note (important)") :]
    assert "not a value to" in body, (
        "say the figure is not a target, or a lower reading looks like a fault"
    )
    assert "resets" in body
    # a second, much lower, healthy reading -- so "far lower" is concrete
    assert "3064" in body, "give a measured low reading, not just the caveat"


def test_the_readme_gives_the_arithmetic_that_makes_the_delta_evidence():
    """A delta only proves nothing was dropped if it is predicted.

    TED 509 + MagicHub 600 + bench 128 - 15 empty-wav failures + 1 golden
    clip = 1223, which is what the counter advanced by. Without the sum, the
    delta is just another number and a silently dropped request would not
    show up.
    """
    readme = _readme()
    body = readme[readme.index("Scope note (important)") :]
    assert "1223" in body
    # Assert the summed expression, not the bare digits: every one of these
    # numbers also occurs elsewhere in the section, so a digit check still
    # passed with the sum reduced to "one pass of each corpus".
    collapsed = " ".join(body.split())
    assert (
        "TED 509 + MagicHub 600 + benchmark 128, minus the 15 empty-wav" in collapsed
    ), "spell out the terms being summed, or the delta cannot be recomputed"
    assert "error` and `abort` stay at `0.0`" in body, (
        "the zero counters are the other half of the check"
    )


def test_the_readme_counts_the_probe_warmup_requests():
    """The sum came out 6 short of the measured delta, and I guessed why.

    Both probes transcribe the clip 3 times before timing anything, and the
    server counts those. Measured 1844 -> 3193 = 1349 with the probes included,
    which is 1 + 494 + 600 + 128 + 63 + 63 -- not ... + 60 + 60.

    An earlier worklog entry attributed the same 6 to "golden 1 + 5 spare",
    which was a guess that happened to reach the right total. Reconciling to a
    number by inventing terms is how a genuinely dropped request would get
    explained away, so the real source is now in the README.
    """
    readme = _readme()
    body = readme[readme.index("Scope note (important)") :]
    collapsed = " ".join(body.split())

    assert "add **6**, not 120" in collapsed, (
        "say how many extra requests the probes contribute"
    )
    # tie it to the code, so the claim is checkable rather than asserted
    assert "for _ in range(3)" in collapsed, "point at the warm-up loop itself"
    assert "1 + 494 + 600 + 128 + 63 + 63 = 1349" in collapsed, (
        "give the full-pass sum with the warm-ups folded in"
    )


@pytest.mark.parametrize(
    "script", ["asr_perf_probe.py", "asr_perf_stream.py"]
)
def test_both_probes_really_warm_up_three_times(script):
    """If a probe's warm-up count changes, the README's +6 is wrong."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "reference_config", "benchmarking", script
    )
    src = open(path).read()
    assert "for _ in range(3)" in src, (
        f"{script} no longer warms up 3 times; the README's arithmetic needs updating"
    )


def test_the_wedge_check_names_metrics_the_server_actually_exports():
    """A metric renamed upstream would make the check silently useless.

    Both names are exported by the running server on this deployment; pin them
    so a vLLM upgrade that renames either fails here rather than in the field.
    """
    plugin_metrics = {
        # vLLM's own metric names, asserted against what the live server
        # exported at 0.26.0 (curl /metrics | grep '^vllm:').
        "vllm:request_success_total",
        "vllm:num_requests_running",
    }
    readme = _readme()
    for metric in sorted(plugin_metrics):
        assert metric in readme, f"{metric} must be the one the runbook quotes"


def test_the_readme_does_not_cite_the_wrong_hang_issues():
    """Two of the three cited issues were not this failure mode.

    Checked against the tracker rather than recalled:
      #40592 -- Mistral, intermittent hang in AllGatherAsync on T3K. A CCL
                hang, not SDPA/decode, and this deployment short-circuits CCL.
      #4752  -- tt-inference-server, Falcon3-7B eval accuracy vs an L4
                reference. Not tt-metal, and not a hang.
    Citing them as "the same SDPA/decode class" sends the next reader to two
    unrelated threads and inflates the apparent corroboration from one issue to
    three.
    """
    readme = _readme()
    body = readme[readme.index("non-deterministic device hang") :]
    body = body[: body.index("Scope note")]

    # they may be named as excluded, but not offered as supporting evidence
    for wrong in ("#40592", "#4752"):
        assert f"issues {wrong}" not in body and f", {wrong}," not in body, (
            f"{wrong} is a different failure mode; do not cite it as this class"
        )
    assert "do **not** belong" in body, (
        "say why they were dropped, or they get re-added from memory"
    )


def test_the_readme_cites_the_issue_that_matches_the_signature():
    """#37543 is the one that actually describes this: ND, SDPA decode, traced."""
    readme = _readme()
    body = readme[readme.index("non-deterministic device hang") :]
    body = body[: body.index("Scope note")]
    assert "37543" in body
    assert "SDPA decode" in body
    # The near-match must be marked as such. "deterministic" alone is satisfied
    # by the surrounding "non-deterministic device hang" prose, so name what
    # makes #45052 different from ours.
    assert "45052" in body
    # "deterministic" alone is satisfied by the surrounding "non-deterministic
    # device hang" prose, so require the phrasing that separates #45052 from
    # ours. Scoped to the report rather than the defect -- see
    # test_the_readme_does_not_overstate_what_45052_establishes.
    assert "100% deterministic and reported only on P300x2" in body, (
        "#45052 is deterministic and was only reported on one board; presenting "
        "it as the same failure overstates how well this hang is understood "
        "upstream"
    )


def test_the_readme_does_not_overstate_what_45052_establishes():
    """Two claims about #45052 went further than the tracker does.

    Checked against the issue:
      - it is 100% deterministic, reproduced over five runs -- fine;
      - it is *reported* on a Blackhole P300x2 (1,4) mesh, but the underlying
        sparse-matmul deadlock is called architecture-agnostic in #45943, so
        "P300x2-specific" describes the report, not the defect;
      - PR #44118's merge (7eff69a85a0, vs 747215b good) is the first bad
        tested version; triage says causality is unproven and points at
        #43682.

    Both overstatements make the upstream picture look better understood than
    it is, and the second sends a reader to the wrong PR. What actually rules
    the issue out here is the mechanism: its stuck op is GPT-OSS MoE
    SparseMatmulDeviceOperation on a 4-device mesh, and this is one p150 with
    no MoE.
    """
    readme = _readme()
    body = readme[readme.index("non-deterministic device hang") :]
    body = body[: body.index("Scope note")]
    flat = " ".join(body.split())

    assert "reported only on P300x2" in flat, (
        "distinguish the report's scope from the defect's"
    )
    assert "45943" in flat, "cite the issue that calls the deadlock arch-agnostic"
    assert "first bad tested version" in flat and "43682" in flat, (
        "#44118 is a boundary, not a culprit; name the real bisection target"
    )
    assert "not a proven root cause" in flat or "not a proven cause" in flat
    # the durable reason it does not apply to this deployment
    assert "MoE" in flat and "SparseMatmul" in flat


def test_the_readme_attributes_the_patch_recipe_correctly():
    """"the recipe PR#4837 established" pointed at the wrong artifact.

    PR #4837 (e8da7006d) adds Qwen3.5-27B and Qwen3.6-27B specs -- three files,
    model_spec.py plus two catalogs -- and contains no `git apply` at all. The
    recipe came from a review comment on it. A reader who opened the merged
    diff looking for the procedure would find specs and conclude the runbook
    was wrong about something more important.
    """
    readme = _readme()
    assert "review comment on PR #4837" in readme
    assert "issuecomment-" in readme, "point at the comment, not just the PR"
    assert "not from that PR's merged diff" in readme


def test_the_readme_shows_how_the_original_recipe_differed():
    """It rewrote existing pins; a bring-up has none, so it adds an entry.

    Without that contrast the reader cannot tell whether deviating from the
    quoted original is a mistake or the point.
    """
    readme = _readme()
    assert "no prod entry to rewrite" in readme
    # the original's shape, so the difference is visible rather than asserted
    quoted = readme[readme.index("review comment on PR #4837") :]
    quoted = quoted[: quoted.index("A bring-up has no prod entry")]
    # Real commit ids as the comment carried them. Blurring either to a
    # placeholder loses the point: the original *rewrote* pins that were
    # already there, which is exactly what a bring-up cannot do.
    assert '-  vllm_commit: "03fa3af"' in quoted
    assert '+  vllm_commit: "b95c0501e62f"' in quoted
    assert 'tt_metal_commit: "de59f8a"' in quoted

def _sample_rate_section():
    readme = _readme()
    start = readme.index("The snippet resamples to 16 kHz")
    return readme[start : readme.index("\n**Do not use", start)]


def test_the_readme_does_not_claim_both_checkpoints_declare_the_rate():
    """Only the JA checkpoint has `sampling_rate`; the base one omits it.

    The note used to read "that is what the checkpoint's
    preprocessor_config.json declares (sampling_rate: 16000)" as if it applied
    to whatever checkpoint you had open. It does not: Qwen/Qwen3-ASR-1.7B --
    the one the reference dumps come from -- has no such key, and a reader
    grepping for it there finds nothing and doubts the whole paragraph.
    """
    body = _sample_rate_section()
    assert "absent" in body, (
        "say that the base checkpoint omits the key, or the claim overreaches"
    )
    assert "WhisperFeatureExtractor" in body, (
        "name where the base checkpoint's 16000 actually comes from"
    )
    # The arithmetic, which is the part that holds for both files. Assert the
    # equations, not the digits: "480000" survives on its own in prose like
    # "the sample count is ... = 480000", which loses the derivation that makes
    # the fallback usable.
    collapsed = " ".join(body.split())
    for equation in (
        "`n_samples` 480000 = `chunk_length` 30 x 16000",
        "`nb_max_frames` 3000 x `hop_length` 160 = 480000",
    ):
        assert equation in collapsed, f"spell out {equation}, not just the numbers"
    assert "present in both files" in body, (
        "point the reader at the check that works regardless of checkpoint"
    )


@pytest.mark.parametrize(
    "repo,declares",
    [("models--neosophie--Qwen3-ASR-1.7B-JA", True), ("models--Qwen--Qwen3-ASR-1.7B", False)],
)
def test_the_preprocessor_configs_match_what_the_readme_says(repo, declares):
    """Check the files, not the prose. Skips where the cache is absent.

    QWEN3ASR_SNAP points at a HF hub cache (the tt-metal tests use the same
    variable); without it there is nothing to compare against and asserting
    would only fail on machines that never downloaded the weights.
    """
    import glob
    import json

    cache = os.environ.get("QWEN3ASR_HF_CACHE") or os.path.expanduser(
        "~/.cache/huggingface/hub"
    )
    found = glob.glob(os.path.join(cache, repo, "snapshots", "*", "preprocessor_config.json"))
    if not found:
        pytest.skip(f"no cached preprocessor_config.json for {repo}")
    cfg = json.load(open(found[0]))
    assert ("sampling_rate" in cfg) is declares, (
        f"{repo}: sampling_rate presence changed; the README table needs updating"
    )
    if declares:
        assert cfg["sampling_rate"] == 16000
    # the geometry the README tells you to fall back on
    assert cfg["n_samples"] == cfg["chunk_length"] * 16000
    assert cfg["nb_max_frames"] * cfg["hop_length"] == cfg["n_samples"]


def _arch_name_section():
    readme = _readme()
    start = readme.index("#### `ARCH_NAME` is absent from the engine")
    return readme[start : readme.index("\n#### ", start + 1)]


def _table_row(body, leading_cell):
    for line in body.splitlines():
        if line.startswith(f"| {leading_cell}"):
            return line
    raise AssertionError(f"no table row for {leading_cell} in the ARCH_NAME section")


def test_the_readme_records_arch_name_missing_from_the_engine():
    """The startup log says "overriding with blackhole"; the worker disagrees.

    Measured per process rather than from that log line: ARCH_NAME reaches the
    APIServer but is absent from the EngineCore environ, while MESH_DEVICE and
    the offline flags reach both. A reader who only sees the entrypoint log
    concludes the variable is in effect on the worker, and then explains an
    unrelated failure with it.
    """
    body = _arch_name_section()
    assert "EngineCore" in body and "APIServer" in body, (
        "name the two processes, or the distinction the measurement makes is lost"
    )

    # The measurement lives in the table, so assert on the row. "absent"
    # anywhere in the section is also satisfied by this section's own heading,
    # which would let the row be flipped to "set" without failing anything.
    row = _table_row(body, "`ARCH_NAME`")
    assert "absent" in row, (
        "the EngineCore cell is the measurement; keep it in the row, not just the prose"
    )
    assert "wormhole_b0" in row and "blackhole" in row, (
        "show both values, or the row does not say what was overridden with what"
    )

    # the variables that *do* arrive, so "absent" is a contrast and not a
    # blanket claim that the spec's env_vars do not work
    arrives = _table_row(body, "`MESH_DEVICE`")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        assert name in arrives, f"{name} does reach the engine; show it"
    assert "absent" not in arrives, (
        "these three were measured present in both processes"
    )


def test_the_readme_explains_why_arch_name_absence_is_harmless():
    """Otherwise this reads as an open bug and the next reader chases it."""
    body = _arch_name_section()
    assert "UMD | Creating TopologyDiscovery for architecture: blackhole" in body, (
        "quote the UMD line, which is the evidence that arch comes off PCIe"
    )
    assert "PCIe" in body


def test_the_readme_quotes_the_todo_that_model_spec_actually_carries():
    """The "transitional" claim has to come from the source, not from memory."""
    todo = "TODO: Remove once all model specs are uplifted to tt-metal >= 0.60.0"
    spec_src = open(
        os.path.join(os.path.dirname(__file__), "..", "workflows", "model_spec.py")
    ).read()
    assert todo in spec_src, (
        "the comment moved or changed; requote it in the README before relying on it"
    )
    assert todo in _arch_name_section()


def test_the_readme_warns_against_exporting_arch_name_globally():
    """The obvious "fix" is the one that would put wormhole_b0 on a worker.

    _infer_env_vars derives ARCH_NAME from the device, so a global export in
    the shell is not how this value is meant to be set, and the base image
    already carries the wrong one.
    """
    body = _arch_name_section()
    assert "wormhole_b0" in body, "name the wrong value the base image ships"
    lowered = body.lower()
    assert "do not" in lowered and "global" in lowered, (
        "say plainly not to export it globally, or the note reads as an invitation"
    )
