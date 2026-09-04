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


def test_the_readme_clones_from_the_forks_only():
    """The branches are pushed, so the recipe must be reproducible as written.

    While they were local-only the images were built against a git daemon on
    the docker host, and the README documented that detour. An image built from
    a daemon on one machine is not reproducible by anyone else, so the detour
    had to go the moment the branches were pushed -- otherwise a reader would
    set up loopback plumbing that is no longer needed, and the recipe would
    never be exercised in the form it is delivered in.
    """
    readme = _readme()
    for leftover in (
        "If a branch is not pushed yet",
        "git daemon",
        "git://172.17.0.1:9418",
        "/tmp/ttmetal-src.git",
        "/tmp/vllmttplugin-src.git",
    ):
        assert leftover not in readme, (
            f"{leftover!r} is loopback plumbing from before the branches were "
            "pushed; the recipe must clone from the forks"
        )

    # what must remain: both clones aimed at the pushed forks
    assert "nyoshifujiTT/tt-metal.git" in readme
    assert "nyoshifujiTT/vllm-tt-plugin.git" in readme


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
    assert "MODEL_SPECS_ENV=dev" in sh, (
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
    assert "lsof -t /dev/tenstorrent" in sh, (
        "verify the device is actually free before relaunching"
    )
    assert "kill -9" in sh, "escalate for anything that still holds the device"


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
    assert "device_ok" in sh, "health must be decided by something observable"
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
