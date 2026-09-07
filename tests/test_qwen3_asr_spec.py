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


def _spec_yaml():
    """The catalog text, for the declarations that do not survive parsing."""
    path = (
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "audio_tts.yaml"
    )
    return path.read_text()

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
    "e7929dc",  # before the served decoder took its dtype from the shared helper
)


def test_no_superseded_commit_is_referenced_anywhere():
    """A superseded SHA must not be *used* as a pin -- but may be discussed.

    The rule this enforces is that no build or run resolves to a tree the repo
    no longer tests. Bare textual absence is a proxy for that, and it broke on
    the first pin whose own story is worth telling: the short-pin trap section
    quotes `e7929dc` precisely because it once resolved to an unrelated
    upstream object. Deleting that example to satisfy a substring check would
    remove the evidence for the full-SHA rule sitting directly above it.

    So the check is scoped to the load-bearing positions: the patch's
    `tt_metal_commit:` line, the bake tag, the `--build-metal-commit`
    argument, and the image tag the runbook starts.
    """
    root = os.path.join(os.path.dirname(__file__), "..")

    def _pin_positions(text):
        """Every occurrence that would actually drive a build or a run."""
        import re

        return (
            re.findall(r'tt_metal_commit:\s*"([0-9a-f]+)"', text)
            + re.findall(r"--build-metal-commit\s+([0-9a-f]+)", text)
            + re.findall(r"ubuntu-22\.04-amd64:([0-9a-f]+)", text)
            + re.findall(r"amd64:[0-9.]+-([0-9a-f]+)-", text)
        )

    for rel in ("workflows/model_spec.py", "scripts/qwen3_asr/README.md"):
        text = open(os.path.join(root, rel)).read()
        used = _pin_positions(text)
        for stale in SUPERSEDED_TT_METAL_COMMITS:
            hits = [p for p in used if p.startswith(stale)]
            assert not hits, (
                f"{rel} still pins the superseded tt-metal commit {stale} "
                f"(found in {hits}); a build from it would use a tree the repo "
                f"no longer tests"
            )


def test_the_pin_check_looks_at_positions_that_drive_a_build():
    """Guard the guard: it must not degrade back to a bare substring test.

    Scoping it to pin positions is what lets the short-pin trap keep its worked
    example. If someone re-tightens this to "not in text", that example has to
    go, and the full-SHA rule loses its evidence.
    """
    src = open(os.path.join(os.path.dirname(__file__), "test_qwen3_asr_spec.py")).read()
    body = src[src.index("def test_no_superseded_commit_is_referenced_anywhere") :]
    body = body[: body.index("\ndef test_the_pin_check_looks_at_positions")]

    assert "tt_metal_commit:" in body and "--build-metal-commit" in body, (
        "the check must name the positions that actually pin a commit"
    )
    assert "assert stale not in text" not in body, (
        "a bare substring check forbids discussing a superseded SHA at all"
    )

    # and the example it exists to protect must still be present
    readme = _readme()
    assert "`e7929dc` matched upstream's `7e7929dcd898...`" in readme, (
        "the short-pin trap's worked example must survive the pin bump"
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
    "c0c4842",  # before the compilation_config pin was restored
)


def test_no_superseded_vllm_fork_commit_is_referenced_anywhere():
    """Same scoping as the tt-metal check: forbid *pinning*, not discussing.

    c0c4842 is now superseded, but the results table names the image it built
    (0.21.0-e7929dcf5dcf...-c0c4842) as the artifact the current numbers came
    from. That attribution is the honest part of the table, so a bare substring
    ban would force deleting it.
    """
    root = os.path.join(os.path.dirname(__file__), "..")

    def _pin_positions(text):
        import re

        return re.findall(r'vllm_commit:\s*"([0-9a-f]+)"', text) + re.findall(
            r"amd64:[0-9.]+-[0-9a-f]+-([0-9a-f]+)", text
        )

    for rel in ("workflows/model_spec.py", "scripts/qwen3_asr/README.md"):
        text = open(os.path.join(root, rel)).read()
        used = _pin_positions(text)
        for stale in SUPERSEDED_VLLM_FORK_COMMITS:
            hits = [p for p in used if p.startswith(stale)]
            assert not hits, (
                f"{rel} still pins the superseded vLLM commit {stale} "
                f"(found in {hits})"
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


def _dropped_work_section():
    readme = _readme()
    start = readme.index("#### After any upstream merge, check for silently dropped work")
    return readme[start : readme.index("\nDisk:", start)]


def test_the_readme_tells_you_to_look_for_work_a_merge_dropped():
    """A green suite cannot detect a change and its test leaving together.

    Concrete case: vllm-tt-plugin's first upstream merge kept
    enforce_eager = True, dropped the compilation_config pin that has to
    accompany it, and dropped the covering test in the same commit. Nothing
    failed, and the branch spent ~209 s per start compiling a graph the ttnn
    path never uses until it was found by hand.

    So the runbook has to name the check, not just the risk -- and has to say
    that a long output is expected, or the next reader dismisses 33 lines as
    noise.
    """
    body = _dropped_work_section()

    # the mechanism, so the check is understood rather than copied blindly
    assert "the assertion left with the code" in body, (
        "say why a passing suite proves nothing here"
    )
    # the worked example, with its cost and its fix
    assert "enforce_eager" in body and "compilation_config" in body
    assert "209 s" in body, "quantify what the omission cost"
    assert "acae5aa" in body, "name the commit that restored it"

    # the check itself, runnable
    assert "git log --format='%h %an'" in body and "def $fn" in body, (
        "give the command, not a description of it"
    )
    # and the discipline it needs to be worth anything
    assert "Every line has to be accounted for" in body
    for category in ("renamed", "replaced", "withdrawn"):
        assert category in body, f"name the '{category}' disposition"
    # The counts, as the sentence that reports them. A bare `"33" in body`
    # also matched `6m33.591s` elsewhere in the runbook, so deleting this
    # sentence outright left the test green.
    flat = " ".join(body.split())
    assert "33 lines here, 2 in tt-metal and 0 in the plugin" in flat, (
        "record that this repo's output was long and still clean, or a long "
        "list looks like a failure of the check"
    )
    # and that every line was dispositioned, which is what makes it clean
    assert "every one of the 35 resolved to the first three" in flat, (
        "a count without a disposition is just a number"
    )
    # The total must be the sum of the per-repo counts the same sentence gives,
    # read back out of the runbook rather than restated here.
    per_repo = re.search(
        r"(\d+) lines here, (\d+) in tt-metal and (\d+) in the plugin", flat
    )
    total = re.search(r"every one of the (\d+) resolved", flat)
    assert sum(int(g) for g in per_repo.groups()) == int(total.group(1)), (
        f"the per-repo counts {per_repo.groups()} do not add up to "
        f"{total.group(1)}"
    )


def test_the_dropped_work_scan_is_usable_in_every_repo():
    """`tests/` is not the test root everywhere, and a wrong root finds nothing.

    tt-metal keeps this bring-up's tests under
    models/demos/audio/qwen3_asr/tests, so a scan hardcoded to `tests/` there
    matches no files and prints nothing -- indistinguishable from a clean
    result. The scan is only trustworthy if the root is set on purpose.
    """
    body = _dropped_work_section()

    # Both roots, as assignments: naming only tt-metal's leaves the reader to
    # guess what the other two repos use, and a bare "TESTS=" is satisfied by
    # either line alone.
    assert "TESTS=tests" in body, "give the root the other two repos use"
    assert "TESTS=models/demos/audio/qwen3_asr/tests" in body, (
        "give tt-metal's root, which is the one that differs"
    )
    assert "silently finds nothing if it is wrong" in body, (
        "warn that a wrong root looks like success"
    )
    # the scan must use the variable rather than the literal
    assert 'grep -rq "def $fn" "$TESTS/"' in body, (
        "the grep has to honour TESTS, or parameterising the filter is pointless"
    )
    # and the tt-metal-specific range, since its full log is unusable
    assert "yito/qwen3_asr_pr" in body, (
        "say how to bound the history on tt-metal"
    )
    # resolved by ref, not by a hardcoded remote name: the branch is under
    # upstream/ in one checkout and origin/ in another, and the wrong one
    # aborts the scan with "fatal: ambiguous argument" rather than skipping
    assert "git rev-parse --verify -q" in body, (
        "resolve the base defensively; a bad revision aborts the scan"
    )
    # Both spellings must appear in the prose that explains why, not only
    # inside the snippet: the explanation is what stops someone "simplifying"
    # the rev-parse fallback back to a single hardcoded ref.
    reason = body[body.index("Resolve the") :]
    reason = reason[: reason.index("```")]
    for ref in ("upstream/yito/qwen3_asr_pr", "origin/yito/qwen3_asr_pr"):
        assert ref in reason, f"say that {ref} is one of the spellings seen"
    assert "fatal: ambiguous" in reason, (
        "name the failure a wrong ref produces, which is an abort not a skip"
    )


def test_the_readme_does_not_present_librispeech_wer_as_runnable():
    """WER 6.7288 came from a config upstream deleted.

    Our lmms-eval entry lived in evals/eval_config.py, which went with the v1
    workflows (#4678 / #4630). The successor catalog carries whisper entries
    but no Qwen3-ASR one, and evals/lmms_eval_models/qwen3_asr_openai.py is now
    an orphan -- nothing outside that directory references it.

    Quoting the number without that context invites someone to try to
    reproduce it and conclude the tree is broken.
    """
    readme = _readme()
    body = readme[readme.index("An older run also had LibriSpeech WER") :]
    body = body[: body.index("This table is a record")]
    flat = " ".join(body.split())

    assert "not reproducible on this tree" in flat, (
        "say the figure cannot be re-measured here"
    )
    assert "evals/eval_config.py" in flat and "reference_config/evals/eval_config.py" in flat, (
        "name both the deleted config and its successor"
    )
    assert "qwen3_asr_openai" in flat, "name the adapter that is now orphaned"
    # and what running it again would take, so the note is actionable. The
    # detail lives in test_the_readme_names_what_actually_blocks_the_
    # librispeech_eval; here just require that a route back is described.
    assert "Re-enabling it" in flat, (
        "say what re-enablement involves, or the note is a dead end"
    )


def _build_script():
    return open(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "build_docker_images.py")
    ).read()


def test_the_readme_gives_the_real_reason_the_base_is_built_by_hand():
    """"the script issues a plain docker build" is false, and misleads.

    build_tt_metal_base_image() runs `docker buildx bake ... ci-build` itself,
    carrying the same FROM-scratch explanation. A reader who checked would find
    the stated reason contradicted and could reasonably drop the manual step.

    The step is still required, for a different reason: the script clones
    https://github.com/tenstorrent/tt-metal.git -- upstream, which does not
    have this branch -- so the checkout of our pin fails. Pre-building the tag
    makes the function return before it ever clones.
    """
    script = _build_script()
    readme = _readme()

    # the premise: the script really does use bake, and really does clone upstream
    assert '"bake",' in script, "the script bakes; the old README claim is stale"
    assert "https://github.com/tenstorrent/tt-metal.git" in script
    assert "if check_image_exists_local(tt_metal_base_tag):" in script, (
        "the early return is what makes the manual build effective"
    )

    body = readme[readme.index("tt-metal's `dockerfile/Dockerfile` declares") :]
    body = body[: body.index("The script then sees the base locally")]
    flat = " ".join(body.split())

    assert "issues a plain `docker build` for the base" not in flat, (
        "the script bakes; do not state the opposite"
    )
    # Require it where the baking is asserted, not merely somewhere in the
    # section: the name recurs in the early-return paragraph, so a section-wide
    # check still passed with the attribution reduced to "it".
    bakes = flat[: flat.index("The reason is where it clones from")]
    assert "`build_tt_metal_base_image()` runs `docker buildx bake" in bakes, (
        "attribute the bake to the function, so the claim can be checked"
    )
    assert "upstream" in flat and "does not carry this bring-up's branch" in flat, (
        "give the real reason: the clone is from upstream"
    )
    assert "check_image_exists_local" in flat, (
        "explain why a pre-built tag suppresses the clone"
    )


def test_the_readme_explains_the_single_threaded_flag():
    """The build command passes it and nothing said why.

    It is not a parallelism knob here: the patch adds one prod entry, so there
    is one combination either way. What it selects is the execution path --
    the alternative is a ProcessPoolExecutor gated on host resources, and that
    gate wants DISK_PER_BUILD_GB (40) free on Docker's data-root, which `/`
    does not have. Without the note, someone tidying the command line drops the
    flag and the single build never gets admitted.
    """
    script = _build_script()
    readme = _readme()

    # the premise, from the script
    assert "if single_threaded:" in script
    assert "_run_resource_aware_queue" in script
    assert "DISK_PER_BUILD_GB = 40" in script, (
        "the reserve changed; requote it in the README"
    )

    body = readme[readme.index("`--single-threaded` is not about parallelism") :]
    body = body[: body.index("\nWithout the tt-metal half")]
    flat = " ".join(body.split())

    assert "one combination" in flat, "say why it is not a parallelism question"
    assert "_run_resource_aware_queue" in flat, "name the path it avoids"
    assert "DISK_PER_BUILD_GB = 40" in flat, "give the reserve that blocks it"
    assert "does not have 40 GB spare" in flat, (
        "connect the reserve to this host, or the flag looks optional"
    )


def _unit_file():
    return open(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scripts",
            "qwen3_asr",
            "qwen3asr-supervisor.service",
        )
    ).read()


def test_the_readme_reconciles_the_service_port_with_its_own_commands():
    """Installing the unit serves 8101; every command here targets 8110.

    PORT="${1:-8101}" in the script, `... asr_supervisor.sh 8101` in the unit,
    and 8110 in every curl/eval/benchmark invocation in this runbook -- because
    those were measured against the --docker-server deployment. Someone who
    follows Install and then pastes a verification command gets connection
    refused and no hint why.
    """
    readme = _readme()
    supervisor = _supervisor()
    unit = _unit_file()

    # the mismatch is real, or this note is stale
    assert 'PORT="${1:-8101}"' in supervisor
    assert "asr_supervisor.sh 8101" in unit
    assert "http://127.0.0.1:8110" in readme

    body = readme[readme.index("The service serves 8101") :]
    body = body[: body.index("### What the supervisor reads")]
    flat = " ".join(body.split())

    assert "not the 8110 used everywhere above" in flat, (
        "name both ports, or the reader cannot see the mismatch"
    )
    assert 'PORT="${1:-8101}"' in flat, "show where the default comes from"
    # both ways out, so the reader can pick
    assert "edit `ExecStart` to pass `8110`" in flat
    assert "substitute the port in the verification commands" in flat


def test_the_readme_says_the_two_serving_modes_cannot_coexist():
    """--local-server and --docker-server both want /dev/tenstorrent/0.

    Retargeting the unit to 8110 without stopping the container gives a device
    conflict, not a working service, so the instruction to change the port has
    to carry that warning.
    """
    readme = _readme()
    body = readme[readme.index("The service serves 8101") :]
    body = body[: body.index("### What the supervisor reads")]
    flat = " ".join(body.split())

    assert "--local-server" in flat and "--docker-server" in flat, (
        "name the two modes; the port is not the only difference"
    )
    assert "Only one of the two can own `/dev/tenstorrent/0`" in flat


def test_the_port_is_documented_as_positional_not_an_env_var():
    """The environment table listed every knob except the one that is not one.

    PORT is $1, so exporting PORT= does nothing. Readers scanning the table for
    how to change the port would find nothing and reasonably assume it is not
    configurable.
    """
    readme = _readme()
    row = _readme_row(readme, "*(positional `$1`)*")
    assert "8101" in row
    assert "Not** an environment variable" in row, (
        "say it cannot be set through the environment, unlike every other row"
    )


def test_the_readme_does_not_call_the_recovery_loop_fully_self_sustaining():
    """It contradicted the BMC finding two paragraphs above it.

    "making the recovery loop fully self-sustaining" was written before the
    power-cycle stage was known to be unavailable here. Left in place, the
    reader gets both claims and no way to tell which is current.
    """
    readme = _readme()
    assert "fully self-sustaining" not in readme, (
        "the power-cycle stage cannot run on this host; scope the claim"
    )
    body = readme[readme.index("`qwen3asr-supervisor.service` runs the supervisor") :]
    body = body[: body.index("\n### ")]
    flat = " ".join(body.split())
    assert "not self-sustaining for a wedge that needs a hardware reset" in flat, (
        "say which case is not covered"
    )
    # and what it does still cover, so this is not read as 'no recovery at all'
    assert "self-sustaining for anything `tt-smi -r` clears" in flat


def test_the_readme_says_the_power_cycle_fallback_is_unavailable_here():
    """Step 4's second stage cannot run on the delivery host.

    Measured: ipmitool is installed, but there is no BMC --

        $ sudo ipmitool mc info
        Could not open device at /dev/ipmi0 or ...: No such file or directory
        $ ls /dev/ipmi*
        ls: cannot access '/dev/ipmi*'

    Presenting a hardware reset as part of the recovery loop overstates what
    self-recovers. It does degrade safely -- the branch falls through to a wait
    loop and relaunches regardless -- but a wedge that survives tt-smi -r needs
    a human here, and that is the operationally important part.
    """
    readme = _readme()
    body = readme[readme.index("The power-cycle fallback does not work") :]
    body = body[: body.index("`qwen3asr-supervisor.service` runs")]
    flat = " ".join(body.split())

    assert "no BMC device" in flat or "no BMC" in flat, (
        "say why it cannot run, not just that it does not"
    )
    assert "/dev/ipmi" in flat, "quote the device that is absent"
    # The safe-degradation, so this does not read as "recovery is broken".
    # Require both halves: what the script does instead, and that the wedge is
    # still retried. An either-or check passed with the retry claim deleted.
    assert "relaunches anyway" in flat, (
        "say what the script does when the power cycle is refused"
    )
    assert "retried rather than abandoned" in flat, (
        "say the wedge is still retried, or this reads as recovery giving up"
    )
    # and the honest limit
    assert "needs a human" in flat, (
        "state that an unrecoverable wedge is not self-healing on this host"
    )


def test_the_supervisor_relaunches_even_when_the_power_cycle_is_refused():
    """The README's "degrades safely" claim has to be true of the script.

    The main loop must keep retrying when the power cycle cannot run. This test
    used to demand that recover_device "must not return non-zero" on that path,
    which conflated two things: the loop continuing, and the function claiming
    success. It got the second one wrong -- reporting a recovery that did not
    happen is exactly the failure this suite is meant to catch -- so the
    requirement is now stated as "every caller tolerates the failure", which is
    what actually keeps the relaunch going.
    """
    sh = _supervisor()
    start = sh.index('log "tt-smi -r insufficient')
    branch = sh[start : sh.index("\n}", start)]

    assert "ipmitool chassis power cycle" in branch
    assert "exit" not in branch, (
        "the power-cycle path must fall through to a relaunch, not exit"
    )
    # a bounded wait, so an accepted-but-deferred power cycle cannot hang us
    assert "seq 1 40" in branch and "sleep 30" in branch, (
        "keep the wait bounded; 40 x 30 s = 20 min matches the startup budget"
    )
    # and the loop keeps going: every call site tolerates a non-zero return
    main = sh[sh.index('log "=== supervisor start') :]
    calls = [
        ln.strip()
        for ln in main.splitlines()
        if "recover_device" in ln and not ln.lstrip().startswith("#")
    ]
    assert calls, "the main loop must still attempt recovery"
    for call in calls:
        assert call.endswith("|| true"), (
            f"a failed recovery must not stop the relaunch loop: {call}"
        )


def test_the_readme_names_what_actually_blocks_the_librispeech_eval():
    """"rehome the adapter" was too vague, and wrong about the mechanism.

    Traced through the tree: our adapter reached the venv via
    evals/lmms_eval_models/install.py, driven by setup_evals_audio(). Upstream
    deleted that hook -- EVALS_AUDIO now has no setup_function -- and gets
    whisper_tt from a TT fork of lmms-eval pinned in
    requirements/evals-audio.txt instead. So adding a catalog entry alone would
    fail at model resolution, which is the part worth writing down.
    """
    readme = _readme()
    body = readme[readme.index("An older run also had LibriSpeech WER") :]
    body = body[: body.index("This table is a record")]
    flat = " ".join(body.split())

    # the template that does work, so the entry can be copied
    assert "whisper_tt" in flat and "EVALS_AUDIO" in flat
    # the mechanism that broke, named precisely
    # Both strings occur again in the closing aside about the stale
    # requirements comment, so assert on the bullet that explains the
    # mechanism rather than on the section as a whole.
    bullet = flat[flat.index("But our adapter reached the venv") :]
    bullet = bullet[: bullet.index("- Upstream gets")]
    assert "driven by `setup_evals_audio()`" in bullet, (
        "name the hook that drove the install, not just the function name"
    )
    assert "upstream deleted that hook" in bullet, (
        "say it was removed upstream, or the reader looks for a local mistake"
    )
    assert "no `setup_function`" in bullet, (
        "say what EVALS_AUDIO looks like now, so the claim is checkable"
    )
    assert "bgoelTT/lmms-eval" in flat, (
        "name where whisper_tt comes from now; that is the pattern to follow"
    )
    # and the consequence of doing only half of it
    assert "fail at model resolution" in flat


def test_the_removed_venv_hook_is_really_gone():
    """Assert against the tree, so this note fails if the hook comes back.

    Also pins the stale comment: requirements/evals-audio.txt still points at
    setup_evals_audio(). If someone fixes that comment upstream, the README's
    aside about it should go too.
    """
    root = os.path.join(os.path.dirname(__file__), "..")

    venvs = open(os.path.join(root, "workflows", "workflow_venvs.py")).read()
    assert "setup_evals_audio" not in venvs, (
        "the hook is back; the README's re-enablement note is stale"
    )
    # EVALS_AUDIO must still be declared, or the whole paragraph is moot
    assert "WorkflowVenvType.EVALS_AUDIO" in venvs

    reqs = open(os.path.join(root, "requirements", "evals-audio.txt")).read()
    assert "bgoelTT/lmms-eval" in reqs, (
        "the lmms-eval source moved; the README names this pin"
    )
    assert "setup_evals_audio()" in reqs, (
        "the stale comment was corrected; drop the README aside about it"
    )


def test_the_librispeech_adapter_really_is_orphaned():
    """Check the tree, not the prose -- and fail here if it gets rehomed.

    If someone wires the adapter back into the successor catalog, the README's
    "not reproducible" note becomes wrong and must be retired. Asserting the
    orphan state makes that a test failure rather than stale documentation.
    """
    root = os.path.join(os.path.dirname(__file__), "..")

    adapter = os.path.join(root, "evals", "lmms_eval_models", "qwen3_asr_openai.py")
    if not os.path.isfile(adapter):
        pytest.skip("the adapter has been moved or removed; revisit the README note")

    successor = os.path.join(root, "reference_config", "evals", "eval_config.py")
    catalog = open(successor).read()
    assert "qwen3_asr_openai" not in catalog, (
        "the adapter is referenced by the successor catalog now; the README's "
        "'not reproducible' note is stale"
    )


def test_the_scan_covers_the_implementation_side_too():
    """Scanning tests alone misses a file of ours deleted upstream.

    The plugin's loss happened to take its test with it, so a test scan found
    it. The reverse is possible and did occur here: upstream deleted
    workflows/run_reports.py in #4630, taking our eval-only ttft fix with it,
    and nothing failed because the v1 tests went too.

    That one turned out to be obsoleted rather than lost -- functional_ttft is
    gone from the repo entirely and the v2 audio path holds ttft as
    Optional[float] behind an `is not None` filter instead of a dict subscript
    -- but the scan had to exist to reach that conclusion at all.
    """
    body = _dropped_work_section()

    flat = " ".join(body.split())

    assert "The test scan alone is not enough" in flat, (
        "say why a second scan is needed"
    )
    # the implementation scan, and it must exclude the test root it already covered
    assert "FILE GONE" in body
    assert 'grep -vE "^$TESTS/"' in body, (
        "the implementation scan must skip what the test scan already did"
    )
    assert "[ -e \"$f\" ]" in body, "give the existence check, not a description"

    # the worked example, with the verdict and the evidence behind it
    assert "workflows/run_reports.py" in body
    assert "functional_ttft" in body and "Optional[float]" in body, (
        "record what was checked before writing the file off"
    )
    assert "obsoleted by the rewrite, not lost" in flat
    assert "Do not restore the file" in flat, (
        "state the action, or the next reader re-adds a file upstream deleted"
    )
    # results from all three repos, so 'clean' is a claim about the whole set
    assert "2 in tt-metal" in body and "0 in the plugin" in body


def test_the_readme_records_the_compilation_cost_on_the_pinned_image():
    """209 s of the startup window goes into an unused compiled graph.

    Read off the serving container's own log at the pinned plugin commit:

      init engine ... took 214.11 s (compilation: 208.97 s)
      compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, ...}

    enforce_eager is set, but VllmConfig.__post_init__ derives the compilation
    mode before the platform hook runs, so the mode survives as VLLM_COMPILE.
    The plugin fix (acae5aa) pins it to NONE.

    Worth documenting precisely because the obvious reading is wrong twice: it
    is not part of the kernel-cache story above (a warm cache does not avoid
    it), and it does not call the throughput numbers into question (upstream's
    later check still disables torch.compile, and cudagraph_mode is already
    NONE, so execution was eager regardless).
    """
    readme = _readme()
    body = readme[readme.index("A second, separate cost sits inside that window") :]
    body = body[: body.index("Requests use the HF repo id")]
    flat = " ".join(body.split())

    assert "208.97 s" in flat, "quote the measured compilation time"
    assert "CompilationMode.VLLM_COMPILE" in flat, (
        "show the mode that survived, or the cause is not identifiable"
    )
    assert "__post_init__" in flat, "name why enforce_eager alone is not enough"
    assert "acae5aa" in flat, "point at the fix, so the note can be retired"
    # and the scope limit, so this is not read as invalidating the benchmarks
    assert "does **not** invalidate" in body
    assert "startup time, not steady-state" in flat


def _comment_only_filter():
    """The comment-only grep the README tells you to run, as a Python predicate.

    Mirrors `grep -vE '^[+-][[:space:]]*(#|$)'`: keep a diff line only if it is
    neither an indented comment nor blank.
    """
    import re

    # Built from the README's own pattern rather than restated, so a change to
    # the documented grep is what this test exercises. BRE character classes
    # translate directly enough for the two we use.
    readme = _readme()
    start = readme.index("| grep -vE '^[+-][[:space:]]")
    quoted = readme[readme.index("'", start) + 1 :]
    quoted = quoted[: quoted.index("'")]
    pattern = re.compile(quoted.replace("[[:space:]]", "[ \\t]"))
    return lambda line: not pattern.match(line)


def test_the_comment_only_check_recognises_indented_comments():
    """The documented grep anchored # to column 0, so it never fired in practice.

    Every comment inside a function is indented, so

        grep -vE '^[+-]#|^(\\+\\+\\+|---)'

    let an indented comment-only diff through and reported it as a real code
    change. That happened for real: tt/qwen3_asr_decoder.py's comment edit was
    printed by the check, which -- taken at face value -- orders a ~7 h rebuild
    for a diff that changes no executed byte.

    The fixed form allows leading whitespace, and drops blank-line changes for
    the same reason.
    """
    readme = _readme()
    assert "[[:space:]]*" in readme, (
        "the comment pattern must tolerate indentation, or it only matches column 0"
    )
    assert "^[+-]#|" not in readme, "the column-0-only form must not come back"
    assert "load-bearing" in readme, (
        "say why the whitespace class is there, or it gets 'simplified' away"
    )

    keep = _comment_only_filter()
    # an indented comment-only diff must be filtered out entirely
    for line in ("-        # old wording", "+        # new wording", "+\t# tab-indented", "+"):
        assert not keep(line), f"{line!r} is a comment/blank change; it must be dropped"
    # while real code must survive, indented or not
    for line in ("+        S_pad = 1024", "-    return None", "+x = 1"):
        assert keep(line), f"{line!r} is executable; it must be reported"


def test_the_readme_lists_every_file_the_pin_check_prints_today():
    """One name was listed while the command prints two.

    Run at the current pins the filename filter prints both
    tt/generator_vllm.py and tt/qwen3_asr_decoder.py. Naming only the first
    leaves a reader who sees two files unsure whether the second is the known
    case or a new one.
    """
    readme = _readme()
    body = readme[readme.index("At the current pins the tt-metal command") :]
    body = body[: body.index("One exception the filename filter")]
    for name in ("tt/generator_vllm.py", "tt/qwen3_asr_decoder.py"):
        assert name in body, f"{name} is printed by the check; name it"
    assert "the pin stays" in body


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
    # ...and that naming it is not by itself what makes it be used. The row
    # said only "passed as MODEL_WEIGHTS_DIR", which was true and insufficient:
    # that variable is read on the local-source branch alone.
    assert "MODEL_SOURCE=local" in row and "--host-weights-dir" in row, (
        "name every part the pin needs; MODEL_WEIGHTS_DIR alone did nothing"
    )
    assert "defaults to `huggingface`" in row, (
        "say why the other two are required, or they look redundant"
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
    # The sentence, not the word. `"both" in body` was also satisfied by "both
    # commits rewrote comment blocks" two paragraphs down, so removing the
    # statement that the checks are a sequence left this green.
    flat = " ".join(body.split())
    assert "Only a file that survives *both* forces the pin to move." in flat, (
        "state that a file must survive both checks, not just one"
    )
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
    # And the executor still has to be justified, since upstream ships none.
    #
    # Not `"0.26.0" in readme`: the version appears three times in this file
    # (the release the plugin pins, the range the merge moved through, and the
    # executor's justification), so deleting the justification left this green.
    # Require the sentence that carries the reason instead.
    assert "TTUniProcExecutor" in readme
    flat = " ".join(readme.split())
    assert "async output thread was removed in 0.24.0 and has not returned in 0.26.0" in flat, (
        "the executor exists because upstream dropped the async output thread "
        "and has not restored it; say so, or it reads as an unexplained fork"
    )


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
    """Directories the merge removed must not be presented as current.

    Checked against the tree rather than from memory, because the original
    list was wrong on all three counts:

      evals/                  still exists -- reduced to lmms_eval_models/,
                              which holds our orphaned qwen3_asr_openai
                              adapter. Naming it is required, not forbidden.
      benchmarking/           only a stale __pycache__ remains; no source.
      workflows/model_spec.py still exists and is imported by
                              reference_config/evals/eval_config.py and
                              workflow_module/model_catalog.py.

    So the rule cannot be "these strings must be absent". What matters is that
    the README does not present the *old eval/benchmark layout* as the place
    to run things, which the paths below cover.
    """
    readme = _readme()
    root = os.path.join(os.path.dirname(__file__), "..")

    # Paths with no source left: mentioning them as somewhere to run is stale.
    for gone in ("`benchmarking/asr_openai_benchmark.py`", "`evals/run_evals.py`"):
        assert gone not in readme, f"{gone} has no source in the tree"
        assert not os.path.exists(
            os.path.join(root, gone.strip("`"))
        ), f"{gone} exists again; this assertion is now wrong"

    # The current homes must be the ones quoted.
    assert "reference_config/evals/asr_ja_eval.py" in readme
    assert "reference_config/benchmarking/asr_openai_benchmark.py" in readme

    # And model_spec.py is current, so requiring its absence was simply false.
    assert os.path.isfile(os.path.join(root, "workflows", "model_spec.py"))


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

    Scoped to the evidence section. Both CERs appear six and four times across
    the file -- results table, per-round records, the upgrade note -- so a
    whole-file check was satisfied by any of them and said nothing about the
    switch. What has to be here is the side-by-side, which is the only place
    the two routes are compared.
    """
    readme = _readme()
    section = readme[readme.index("Evidence that the move changed no output") :]
    section = section[: section.index("\n#### ")]
    for name, cer in (("TED 509 CER", "0.1002"), ("MagicHub 600 CER", "0.1668")):
        row = next(
            (ln for ln in section.splitlines() if ln.startswith(f"| {name} |")), None
        )
        assert row, f"the {name} row must be in the evidence table"
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert len(cells) == 3, f"both routes must be shown: {row}"
        assert cer in cells[1] and cer in cells[2], (
            f"{name} must be quoted for BOTH routes, or the table does not show "
            f"the switch preserved it: {row}"
        )


def test_the_readme_keeps_the_migration_evidence():
    """"Nothing was lost" is a claim about accuracy, so it needs both columns.

    The fork run is kept as the record that the move preserved output. Quoting
    only the plugin numbers would leave the claim uncheckable.
    """
    readme = _readme()
    # Bound by the next heading rather than by a character count: the fixed
    # 2500-char window silently excluded the closing sentence as soon as the
    # LibriSpeech note above it grew.
    section = readme[readme.index("Evidence that the move changed no output") :]
    section = section[: section.index("\n#### ")]
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

    # Scoped to the "### 4." chapter -- its subsections included, since that is
    # where the commands and their results live. A whole-file check said
    # nothing (both CERs occur several times elsewhere), and bounding at the
    # next `#### ` was too tight: `### 4.` opens by explaining why the upstream
    # harnesses do not fit, and only its subsections name ours and quote the
    # numbers.
    chapter = readme[readme.index("### 4. Eval and benchmark") :]
    next_chapter = re.search(r"\n## ", chapter)
    assert next_chapter, "the chapter must be bounded by a following ## heading"
    chapter = chapter[: next_chapter.start()]

    assert "asr_ja_eval.py" in chapter, "the corpus CER harness must be named"
    assert "asr_openai_benchmark.py" in chapter, (
        "the throughput harness must be named"
    )
    # the measured results, so a rerun can be compared against something
    assert "0.1002" in chapter and "0.1668" in chapter, (
        "the chapter that documents the harnesses must also record what they "
        "measured, or the commands produce numbers with nothing to compare to"
    )


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

    Measured, 60 requests at concurrency 4 on the FLEURS clip, with the
    streaming probe counting tokens rather than SSE frames:
      non-streaming  TTFT 1.170 s, decode TPS/user 23.31, aggregate 42.54
      streaming      TTFT 1.324 s, decode TPS/user 24.24, aggregate 41.50
    """
    readme = _readme()
    section = readme[readme.index("Serving-level timings") :]
    section = section[: section.index("**No image exists at these pins yet.**")]

    # both probes' headline numbers, not just one column
    for value in ("1.170", "1.324", "23.31", "24.24", "42.54", "41.50"):
        assert value in section, (
            f"{value} was measured; record it or a rerun has no baseline"
        )
    assert "60 / 60" in section, "say how many requests the numbers came from"
    # the superseded frame-counted figures must not linger as if current
    for stale in ("22.22", "39.52"):
        assert stale not in section, (
            f"{stale} was measured in SSE frames, not tokens; it is not a baseline"
        )


def test_the_readme_explains_the_gap_between_the_two_probes():
    """Two columns that differ with no stated reason read as a contradiction.

    The streaming column is consistently the slower one on latency, which is
    expected: SSE framing and scheduling land on the client's clock but not on
    the decode counters.

    This test used to also REQUIRE the sentence "the final chunk carries no new
    token" as the explanation for the one-token gap in tokens-per-request. That
    was a false claim being held in place by its own test: the streaming
    figure was a count of SSE frames, not tokens, so there was no off-by-one to
    explain -- just two different units. Requiring the explanation is what kept
    the unit bug invisible, which is exactly the failure mode a test is
    supposed to prevent.
    """
    readme = _readme()
    section = readme[readme.index("Serving-level timings") :]
    section = section[: section.index("**No image exists at these pins yet.**")]
    assert "SSE framing" in section, "say why the client-side number is higher"
    # and a threshold, so "agrees" is not left to taste
    assert "tens of percent" in section, (
        "state how large a gap stops being framing overhead"
    )
    # the retracted explanation may only survive as the quotation inside its
    # own correction
    flat = " ".join(section.split())
    if "final chunk carries no new token" in flat:
        i = flat.index("final chunk carries no new token")
        assert "described a coincidence" in flat[i : i + 200], (
            "that sentence is retracted; it may only appear where it is retracted"
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


def test_the_readme_says_how_to_restart_the_server():
    """A second run.py over a live container fails, and /health hides it.

    The documented launch publishes 8110. Run again without stopping the old
    container and run.py raises "Docker container failed to start." while the
    previous container keeps answering, so /health stays 200 and the restart
    looks successful. Observed during a bring-up rerun: the eval suite was then
    driven against the very container that was being restarted.
    """
    readme = _readme()
    section = readme[readme.index("### 3. Run") :]
    section = section[: section.index("### 4.")]
    assert "Docker container failed to start." in section, (
        "quote the error, or the reader cannot recognise it"
    )
    assert "docker stop $(docker ps -q)" in section, "give the command that fixes it"
    assert "not confirmed by `/health`" in section or "not confirmed\nby `/health`" in section, (
        "say that /health cannot distinguish a restart from the old container"
    )
    assert "docker ps" in section, "and name the check that can"


def test_the_readme_quantifies_the_first_transcription():
    """"can take minutes" is not enough to size a timeout against.

    Measured after a device reset on this host: 6m33.591s for the first
    transcription and 1.223s for the second. A curl --max-time 300 -- an
    entirely reasonable-looking choice against a "minutes" warning -- cannot
    survive it, and when curl gives up the request looks like it disappeared.
    That misreading cost three container restarts during this bring-up before
    the request was simply waited out.
    """
    readme = _readme()
    section = readme[readme.index("The very first transcription JIT-compiles") :]
    section = section[: section.index("non-deterministic device hang")]
    assert "6m33" in section, "quote the measured worst case, not just 'minutes'"
    assert "0m1.223s" in section, "and the second request, so the gap is visible"
    # The instruction itself, not merely the words somewhere in the section:
    # both "--max-time" and "7 minutes" also occur further down in the same
    # block, so a bare containment check survived deleting the instruction.
    flat = " ".join(section.split())
    assert "Budget 7 minutes for it, and do not put a shorter `--max-time` on that request." in flat, (
        "state the budget and the timeout warning as one instruction"
    )
    assert "cannot survive that" in flat, (
        "say what happens when the timeout is shorter -- curl gives up and the "
        "request looks like it vanished"
    )


def test_the_readme_separates_startup_warmth_from_the_first_request():
    """A warm cache gets /health up fast and says nothing about the first clip.

    The startup table quotes 140 s for a warm kernel cache, which reads as "the
    machine is ready". A device reset still leaves the first transcription to
    recompile, so the two costs have to be stated as separate.
    """
    readme = _readme()
    section = readme[readme.index("The very first transcription JIT-compiles") :]
    section = section[: section.index("non-deterministic device hang")]
    assert "separate" in section, "say the two costs are distinct"
    assert "does not imply" in section, "and that one does not predict the other"


def test_the_readme_says_why_the_process_view_cannot_settle_it():
    """Both the log and the CPU look the same during a compile and a wedge."""
    readme = _readme()
    section = readme[readme.index("The very first transcription JIT-compiles") :]
    section = section[: section.index("non-deterministic device hang")]
    assert "Running: 0 reqs" in section, "the scheduler shows nothing either way"
    assert "is a warning, not a stopping point" in section, (
        "the trace-allocator line is the last thing logged; say it is benign"
    )
    assert "neither reading distinguishes it" in section, (
        "spinning vs idle CPU does not separate the two cases"
    )


def test_the_restart_note_uses_the_holder_check_that_works():
    """lsof misses a containerised holder; the runbook must not recommend it.

    asr_supervisor.sh walks /proc for exactly this reason -- a container has
    its own device node, so `lsof -t /dev/tenstorrent/*` reports the device
    free while the fd is open. A restart note that told the reader to use lsof
    would send them to the one check that cannot see the blocker.
    """
    readme = _readme()
    section = readme[readme.index("### 3. Run") : readme.index("### 4.")]
    assert "/proc/[0-9]*/fd" in section, "the holder check must walk /proc"
    assert "Starting devices in cluster" in section, (
        "name the symptom a surviving holder produces"
    )
    supervisor = _supervisor()
    assert "/proc/[0-9]*/fd" in supervisor, (
        "the supervisor's holder check moved; the runbook points at it"
    )


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


def test_the_supervisor_actually_pins_the_snapshot_it_names():
    """MODEL_WEIGHTS_DIR alone is read on one branch the supervisor never took.

    SNAP names a revision (.../snapshots/987bda16...), so the launch reads as
    "start on this revision". It was not: setup_host.py reads MODEL_WEIGHTS_DIR
    only under `model_source == local`, and model_source defaults to
    `huggingface` (os.getenv("MODEL_SOURCE", HUGGINGFACE)). The supervisor set
    neither MODEL_SOURCE nor --host-weights-dir/--host-hf-cache, so run.py
    resolved the repo through the HF cache and the pinned revision was
    decorative -- the existence check on SNAP passed while a different snapshot
    could be served.
    """
    sh = _supervisor()
    launch = sh[sh.index("launch_server()") : sh.index("wait_healthy()")]
    code = [ln for ln in launch.splitlines() if not ln.strip().startswith("#")]

    assert any("MODEL_WEIGHTS_DIR=" in ln for ln in code), "the weights dir must be passed"
    assert any("MODEL_SOURCE=local" in ln for ln in code), (
        "MODEL_WEIGHTS_DIR is only read on the local branch; select it"
    )
    assert any("--host-weights-dir" in ln for ln in code), (
        "and run.py has to be told the directory too, or it re-resolves the repo"
    )


def test_the_local_source_branch_is_the_one_that_reads_the_weights_dir():
    """Check the premise against the real code, not against this docstring.

    If setup_host.py ever reads MODEL_WEIGHTS_DIR unconditionally, the
    MODEL_SOURCE=local above becomes unnecessary rather than wrong -- but until
    then it is load-bearing, and this is what says so.
    """
    setup = (get_repo_root_path() / "workflows" / "setup_host.py").read_text()
    # the default really is huggingface
    assert 'os.getenv(\n        "MODEL_SOURCE", ModelSource.HUGGINGFACE.value\n    )' in setup or (
        '"MODEL_SOURCE", ModelSource.HUGGINGFACE.value' in setup
    ), "model_source no longer defaults to huggingface; re-check the supervisor"
    # and every read of MODEL_WEIGHTS_DIR sits under a LOCAL branch
    for idx, line in enumerate(setup.splitlines()):
        if 'getenv("MODEL_WEIGHTS_DIR")' in line:
            before = "\n".join(setup.splitlines()[max(0, idx - 25) : idx])
            assert "ModelSource.LOCAL.value" in before, (
                f"line {idx + 1} reads MODEL_WEIGHTS_DIR outside a local-source "
                f"branch; the supervisor's MODEL_SOURCE=local may be redundant"
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

    Checked against code rather than the whole file. Two comments now explain
    why the pattern is handled the way it is, so a containment check on the
    file passed even with the actual kill deleted -- verified by removing
    `kill_ours "VLLM::EngineCore"` and watching this test stay green.
    """
    sh = _supervisor()
    code = "\n".join(
        line for line in sh.splitlines() if not line.lstrip().startswith("#")
    )
    assert "VLLM::EngineCore" in code, (
        "the engine process must be killed, not just its run.py parent"
    )
    assert "device_holders" in code, (
        "verify the device is actually free before relaunching"
    )
    assert "kill -9" in code, "escalate for anything that still holds the device"
    # and the engine pattern must be stopped, not merely named somewhere
    stop = sh[sh.index("stop_server() {") : sh.index("launch_server() {")]
    stop_code = "\n".join(
        line for line in stop.splitlines() if not line.lstrip().startswith("#")
    )
    assert "VLLM::EngineCore" in stop_code, (
        "stop_server must be the place that stops the engine"
    )


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
    # In code, not in the comment that explains it: a cgroup pattern only
    # spares anything if in_container actually matches on it.
    sh_code = "\n".join(
        line for line in sh.splitlines() if not line.lstrip().startswith("#")
    )
    assert "/docker-" in sh_code, "identify container processes by cgroup"
    # and it must actually be consulted on both paths
    assert sh.count("in_container ") >= 2, (
        "in_container must gate both the engine kill and the device-holder "
        "escalation"
    )


def test_no_kill_in_the_supervisor_bypasses_that_filter():
    """The guard was on one pattern; the other two killed the container.

    This test used to ban only `pkill -f "VLLM::EngineCore"`, so

        pkill -f "run.py --model Qwen3-ASR"
        pkill -f "run_vllm_api_server.py"

    sat right above it, unguarded, and passed. Host /proc lists processes
    inside containers, so on a host serving via --docker-server the second one
    matches the live server -- measured: `pgrep -af run_vllm_api_server.py` ->
    pid 2714943, `/proc/2714943/cgroup` ->
    /system.slice/docker-9c2677b2....scope, equal to the container's
    .State.Pid. Sparing the engine while killing the API server in front of it
    is not sparing anything.

    So the rule is not "guard the engine pattern"; it is "every kill in this
    script goes through in_container".
    """
    sh = _supervisor()
    code = [ln for ln in sh.splitlines() if not ln.lstrip().startswith("#")]

    # pkill cannot be filtered per-pid at all, so it may not appear in code.
    offenders = [ln.strip() for ln in code if "pkill" in ln]
    assert not offenders, (
        f"pkill kills every match, including containerised ones: {offenders}"
    )

    # Every kill must sit inside the one helper that consults in_container,
    # or be the escalation that already filtered its pid list.
    helper = sh[sh.index("kill_ours() {") : sh.index("stop_server() {")]
    assert "in_container" in helper, "kill_ours must consult in_container"
    assert 'kill "$pid"' in helper, "and it is the helper that does the killing"

    outside = sh.replace(helper, "")
    outside_code = [ln for ln in outside.splitlines() if not ln.lstrip().startswith("#")]
    stray = [
        ln.strip()
        for ln in outside_code
        # the -9 escalation in stop_server kills $stubborn, which device_holders
        # + in_container already filtered; anything else is unguarded
        if re.search(r"\bkill\b", ln) and "$stubborn" not in ln and "kill_ours" not in ln
    ]
    assert not stray, f"these kills do not go through in_container: {stray}"


def test_every_process_pattern_the_supervisor_stops_is_routed_through_it():
    """All three patterns must be stopped, and all three via the helper."""
    sh = _supervisor()
    stop = sh[sh.index("stop_server() {") : sh.index("launch_server() {")]
    for pattern in ("run.py --model Qwen3-ASR", "run_vllm_api_server.py", "VLLM::EngineCore"):
        assert f'kill_ours "{pattern}"' in stop, (
            f"{pattern} must be stopped through the filtered helper"
        )


def test_the_supervisor_will_not_launch_while_a_container_owns_the_chip():
    """Sparing the container's processes is not enough on its own.

    Once stop_server leaves them alone the device stays held, and launching
    anyway does not fail cleanly: run.py hangs in "Starting devices in
    cluster", wait_healthy spends its full 20 minutes (watching 8101 while the
    container serves 8110), and recover_device then runs `tt-smi -r` -- a reset
    of the chip the deployment is serving on. device_ok only asks whether
    tt-smi can read the board, so the reset reports success and the loop
    repeats: a healthy production server wedged every 20 minutes indefinitely.

    The main loop therefore has to refuse to launch while a containerised pid
    holds the device, rather than discovering it 20 minutes later.
    """
    sh = _supervisor()
    main = sh[sh.index('log "=== supervisor start') :]
    guard = main[: main.index("  launch_server")]

    assert "device_holders" in guard, "the check must run before the launch"
    assert "in_container" in guard, "and only containerised holders may block it"
    assert "not launching" in guard, "say what it is doing instead"
    # it must wait, not fall through
    assert "sleep" in guard, "the guard must block rather than proceed"
    # and the launch must be genuinely after it
    assert guard.index("device_holders") < guard.index("sleep")


def test_the_guard_does_not_block_on_our_own_leftovers():
    """A stale local-server pid is ours to kill; only containers gate us."""
    sh = _supervisor()
    main = sh[sh.index('log "=== supervisor start') :]
    guard = main[: main.index("  launch_server")]
    # the holder list must be filtered by in_container before it blocks
    assert 'in_container "$pid" && held=' in guard, (
        "blocking on every holder would deadlock against our own stale process, "
        "which stop_server is there to clean up"
    )


def test_recover_device_refuses_to_reset_a_chip_a_container_is_serving():
    """The pre-launch guard is not the only way into tt-smi -r.

    recover_device is also reached from the monitor loop, so a container
    started underneath us -- or a canary failing for an unrelated reason --
    lands on `tt-smi -r` with the deployment still serving. tt-smi does not
    care who holds the chip, and device_ok reports success afterwards because
    the board itself reads fine, so the reset is both destructive and invisible.
    """
    sh = _supervisor()
    body = sh[sh.index("recover_device() {") : sh.index("kill_ours() {")]

    assert "device_holders" in body, "recover_device must check who holds the chip"
    assert "in_container" in body, "and only containerised holders may stop it"
    assert "return 1" in body, "it must decline rather than reset"
    # the check has to come before the reset, not after it
    assert body.index("device_holders") < body.index("$TTSMI"), (
        "the holder check must precede tt-smi -r"
    )


def test_a_declined_recovery_does_not_abort_the_supervisor():
    """`set -u` is on and the callers ignore the status, so make that explicit.

    Both call sites continue/break back to the guard, which then waits for the
    container. Writing `recover_device || true` says the non-zero return is an
    expected outcome rather than an oversight.
    """
    sh = _supervisor()
    main = sh[sh.index('log "=== supervisor start') :]
    calls = [ln.strip() for ln in main.splitlines() if "recover_device" in ln and not ln.lstrip().startswith("#")]
    assert calls, "the main loop must still attempt recovery"
    for call in calls:
        assert call.endswith("|| true"), (
            f"a declined recovery must not be read as a script error: {call}"
        )


def test_the_device_chmod_does_not_widen_the_by_id_directory():
    """`chmod 666 /dev/tenstorrent/*` also hits by-id/ and breaks traversal.

    udev creates /dev/tenstorrent/by-id/ to hold the stable
    `blackhole-<asic_id>` symlinks, and the glob matches that directory. 666 on
    a directory drops its execute bit, so nothing non-root can traverse it, and
    it stays that way until udev recreates it at the next boot.

    Measured on the delivery host after the supervisor had run:

        /dev/tenstorrent/by-id  drw-rw-rw-   ctime 2026-09-04 01:19
        stat /dev/tenstorrent/by-id/*  ->  Permission denied

    udev's own rule is `SUBSYSTEM=="tenstorrent", MODE="0666"`, which applies
    to the device nodes only -- the scope the supervisor wanted.
    """
    sh = _supervisor()
    code = [ln for ln in sh.splitlines() if not ln.lstrip().startswith("#")]

    offenders = [ln.strip() for ln in code if "chmod 666 /dev/tenstorrent/*" in ln]
    assert not offenders, (
        f"this glob includes the by-id directory: {offenders}"
    )

    # the replacement must exist and must test for a character device
    assert "\nrelax_device_perms() {" in sh, "the chmod belongs in one helper"
    helper = sh[sh.index("relax_device_perms() {") : sh.index("recover_device() {")]
    assert '[ -c "$node" ]' in helper, (
        "only character devices may be chmod'ed; by-id is a directory"
    )


def test_every_device_chmod_goes_through_that_helper():
    """Three call sites had the glob; a fourth must not reintroduce it."""
    sh = _supervisor()
    helper = sh[sh.index("relax_device_perms() {") : sh.index("recover_device() {")]
    outside = sh.replace(helper, "")
    code = [ln for ln in outside.splitlines() if not ln.lstrip().startswith("#")]
    stray = [ln.strip() for ln in code if "chmod" in ln and "relax_device_perms" not in ln]
    assert not stray, f"these chmods bypass the helper: {stray}"
    # and the helper is actually used on every path that leaves the device
    # freshly reset. recover_device has two of them -- the tt-smi -r success
    # return and the fall-through after the power cycle -- so counting once per
    # function let either be dropped silently.
    # Count the exits rather than hardcoding a number: recover_device gained a
    # third one when the unavailable-power-cycle path started returning
    # failure, and a fixed 2 would have had to be edited rather than checked.
    recover = sh[sh.index("recover_device() {") : sh.index("# Kill a previous run")]
    exits = len(re.findall(r"^\s+return\b", recover, re.M))
    assert exits >= 3, f"recover_device should have several exits, found {exits}"
    assert recover.count("relax_device_perms") == exits - 1, (
        "every exit that leaves the device reset must relax the nodes; the only "
        "exception is the containerised-holder refusal, which never touched it. "
        f"exits={exits}, relax calls={recover.count('relax_device_perms')}"
    )
    launch = sh[sh.index("launch_server() {") : sh.index("wait_healthy() {")]
    assert "relax_device_perms" in launch, "launch_server must relax the nodes"


def test_the_runbook_tells_you_how_to_repair_a_widened_by_id():
    """Hosts that ran the older script are still broken until someone fixes it.

    The bad chmod persists across supervisor restarts -- only a reboot (udev
    recreating the directory) or an explicit chmod clears it -- so a fix in the
    script does not fix the machines it already ran on. The delivery host was
    found in that state days later.
    """
    readme = _readme()
    row = _readme_row(readme, "`relax_device_perms`")
    assert "chip nodes only" in row, "say what the helper's scope is"
    assert "drw-rw-rw-" in row, "and what the broken state looks like"

    section = readme[readme.index("`relax_device_perms`") :]
    flat = " ".join(section.split())
    assert "chmod 755 /dev/tenstorrent/by-id" in flat, (
        "give the repair command; the script fix does not reach hosts it already ran on"
    )
    assert "next boot" in flat, "say that it does not clear itself"


def _canary_timings(sh):
    """(steady-state canary timeout, monitor sleep, fails before recovery)."""
    canary = sh[sh.index("canary_ok() {") : sh.index("CANARY_FIRST_TIMEOUT=")]
    default = re.search(r'local timeout="\$\{1:-(\d+)\}"', canary)
    assert default, "canary_ok must take its timeout as an argument with a default"
    monitor = sh[sh.index("# monitor loop") :]
    nap = re.search(r"^\s*sleep (\d+)$", monitor, re.M)
    fails = re.search(r'\[ "\$fails" -ge (\d+) \]', monitor)
    assert nap and fails
    return int(default.group(1)), int(nap.group(1)), int(fails.group(1))


def test_the_monitor_is_not_asked_about_a_server_that_never_served():
    """/health 200 does not mean a transcription can complete yet.

    The first transcription JIT-compiles kernels: measured 6m27s-6m45s across
    fourteen runs, and on this run the route was published at 02:35:52 while the
    first transcription finished at 02:45 -- 9.1 minutes later. The monitor
    loop starts 20 s after wait_healthy returns and gives the canary 45 s, so
    two failures arrive 2.2 minutes in and declare a wedge. recover_device then
    resets the device mid-compile and the loop relaunches, so the supervisor
    could never bring the service up by itself.

    The launch path therefore has to spend the compile budget before the
    monitor's short canary is used at all.
    """
    sh = _supervisor()
    main = sh[sh.index('log "=== supervisor start') :]

    assert "warm_first_transcription" in main, (
        "the launch path must complete one transcription before monitoring"
    )
    # and it must come after wait_healthy but before the monitor loop
    assert main.index("wait_healthy") < main.index("warm_first_transcription") < main.index(
        "# monitor loop"
    ), "the warm-up belongs between the health gate and the monitor"


def test_the_warm_up_budget_covers_the_measured_compile():
    """A budget shorter than the measurement reintroduces the same loop."""
    sh = _supervisor()
    budget = re.search(r'CANARY_FIRST_TIMEOUT="\$\{CANARY_FIRST_TIMEOUT:-(\d+)\}"', sh)
    assert budget, "the first-transcription budget must be a named, overridable value"
    seconds = int(budget.group(1))
    # Slowest first transcription observed here is 6m44.8s (of fourteen), and
    # the runbook tells readers to budget 7 minutes. 7*60 = 420s clears the
    # slowest by 15s, which is why the shipped default is 600s rather than 420.
    slowest_measured = 6 * 60 + 45
    assert seconds >= slowest_measured, (
        f"{seconds}s is under the slowest measured compile ({slowest_measured}s); "
        f"the monitor would call a still-compiling server wedged"
    )
    assert seconds >= 7 * 60, (
        f"{seconds}s is under the 7 minutes the runbook tells readers to budget"
    )


def test_the_steady_state_canary_stays_short():
    """The long budget is for the first request only.

    If the monitor also waited minutes, a genuinely wedged server would go
    unnoticed for that long -- which is what the canary exists to catch.
    """
    default, nap, fails = _canary_timings(_supervisor())
    assert default <= 60, f"the steady-state canary must stay short, got {default}s"
    # and the wedge verdict must still be reached in a couple of minutes
    worst = fails * (nap + default)
    assert worst <= 5 * 60, f"a wedge would take {worst}s to notice"


def test_a_failed_warm_up_recovers_instead_of_monitoring():
    """If the first transcription never returns, that IS the wedge."""
    sh = _supervisor()
    main = sh[sh.index('log "=== supervisor start') :]
    block = main[main.index("warm_first_transcription") :]
    block = block[: block.index("# monitor loop")]
    assert "recover_device" in block, "a failed warm-up must recover, not proceed"
    assert "continue" in block, "and restart the launch rather than monitor"


def test_a_failed_power_cycle_is_not_reported_as_a_recovery():
    """No BMC here, and the old form logged success anyway.

    `sudo ipmitool chassis power cycle >/dev/null 2>&1` discarded both the
    output and the status. The wait loop that followed breaks as soon as
    /dev/tenstorrent/0 exists and device_ok passes -- both already true on this
    host -- so 30 s later it logged "device back after power cycle" and
    returned 0, having neither power-cycled nor done anything past the tt-smi
    -r above. Measured, with ipmitool stubbed to fail as it really does here:

        04:14:22 ... ipmitool chassis power cycle (host will reboot)
        04:14:52 device back after power cycle        <- false

    An operator reading the log would conclude the board was recovered.
    """
    sh = _supervisor()
    body = sh[sh.index("recover_device() {") : sh.index("# Kill a previous run")]

    # the status must be checked, and the error kept
    assert ">/dev/null 2>&1" not in body.split("ipmitool")[1].split("\n")[0], (
        "discarding ipmitool's status is what hid the failure"
    )
    assert "ipmi_err=$(sudo ipmitool chassis power cycle 2>&1)" in body, (
        "capture stderr so the real reason can be logged"
    )
    assert "power cycle UNAVAILABLE" in body, "say plainly that it did not happen"
    assert "a human has to power-cycle it" in body, (
        "and what the operator has to do instead"
    )


def test_recover_device_returns_failure_when_it_did_not_recover():
    """Callers treat 0 as recovered; only actual recovery may return 0."""
    sh = _supervisor()
    body = sh[sh.index("recover_device() {") : sh.index("# Kill a previous run")]

    # the tt-smi -r success path returns 0
    assert 'log "device recovered by tt-smi -r"' in body
    # the no-BMC path must return non-zero
    unavailable = body[body.index("power cycle UNAVAILABLE") :]
    assert "return 1" in unavailable[: unavailable.index("for _ in")], (
        "a power cycle that could not run is not a recovery"
    )
    # and a power cycle that ran but did not bring the board back
    assert "did not come back within 20 minutes" in body, (
        "an accepted-but-ineffective power cycle must be recorded too"
    )
    assert 'return "$came_back"' in body, "and reported to the caller"


def test_the_no_bmc_note_and_the_script_agree():
    """The runbook already said the power-cycle path cannot run here.

    That note and a log line claiming success were both in the tree at once.
    Keep the note, now that the script agrees with it.
    """
    readme = _readme()
    assert "ipmitool" in readme, "the runbook must still name the escalation"
    flat = " ".join(readme.split())
    assert "no BMC" in flat or "BMC" in flat, (
        "and that this host has none, or the script's refusal looks like a bug"
    )


def test_the_runbook_says_the_unit_waits_rather_than_taking_over():
    """Otherwise "enable the service" reads as "the service now runs".

    Someone installing the unit on a host that is already serving via
    --docker-server needs to know the supervisor will sit and log rather than
    start, and why launching anyway would have been worse than useless.
    """
    readme = _readme()
    section = readme[readme.index("why the holder check exists") :]
    section = section[: section.index("### What the supervisor reads")]
    flat = " ".join(section.split())
    assert "does not take the service over -- it waits" in flat, (
        "say what enabling the unit does while a container holds the chip"
    )
    assert "not launching" in flat, "quote the log line, so it is recognisable"
    # and the reason the guard is not merely tidy
    assert "tt-smi -r" in flat and "resetting the chip" in flat, (
        "say that launching anyway resets a chip that is serving traffic"
    )
    assert "every 20 minutes" in flat, "and that it repeats, rather than failing once"


def test_the_runbook_does_not_narrow_the_guard_to_the_engine():
    """"spares the engine" understated it and matched the old broken code.

    The review table said in_container "spares the engine of a running
    --docker-server", which described exactly the state where the two pkills
    above it killed that deployment's API server. A reader auditing whether the
    supervisor is safe to install next to a container would have read that row
    and stopped.
    """
    readme = _readme()
    row = _readme_row(readme, "`in_container`")
    assert "spares the engine" not in row, (
        "the guard is not engine-specific; that wording matched the bug"
    )
    helper_row = _readme_row(readme, "`kill_ours`")
    assert "every" in helper_row.lower(), "say the guard covers every process"
    assert "2714943" in helper_row, (
        "quote the pid it was exercised against, so the claim is checkable"
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
    # and the correct node has to be used, not merely mentioned: three of the
    # four occurrences are comments explaining the device-holder machinery.
    node_code = "\n".join(
        line for line in sh.splitlines() if not line.lstrip().startswith("#")
    )
    assert "/dev/tenstorrent/0" in node_code, (
        "the post-power-cycle wait must test the node that exists on this host"
    )


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
def test_the_readme_says_why_completions_answers_on_an_asr_model():
    """"the ASR model answers that endpoint too" reads as an accident.

    It is a setting: the adapter declares supports_transcription = True and
    supports_transcription_only = False, and the plugin's get_supported_tasks
    branches on exactly that -- True alone returns ["transcription"] and drops
    "generate". Verified live: /v1/completions and /v1/audio/transcriptions
    both return 200.

    Worth pinning because this suite has no other entry point. Flipping the
    flag would remove its access to the server, and a reader who thought the
    behaviour was incidental would not connect the two.
    """
    readme = _readme()
    body = readme[readme.index("### The plugin's server-facing tests") :]
    body = body[: body.index("```")]
    flat = " ".join(body.split())

    assert "supports_transcription_only = False" in flat, (
        "name the flag that keeps /v1/completions available"
    )
    assert "would return `[\"transcription\"]`" in flat or 'return `["transcription"]`' in flat, (
        "say what the other setting does, or the flag looks decorative"
    )
    assert "both return 200" in flat, "record that this was checked, not assumed"
    # the consequence for these tests specifically
    assert "only entry point" in flat


def test_the_readme_covers_a_docstring_only_change():
    """The comment-only grep only knows `#`, so a docstring edit over-reports.

    That is the safe direction -- it never lets a real change through -- but
    without guidance the reader spends a ~7 h rebuild on a diff that changed no
    executed byte. It happened immediately: vllm-tt-plugin aec8563 edits only
    executor.py's module docstring, and the check prints the file.

    The follow-up is cheap and decisive: if every hunk header falls inside the
    docstring, the pin stays.
    """
    readme = _readme()
    body = readme[readme.index("It still only understands `#` comments") :]
    body = body[: body.index("The extra tt-metal exclusions")]
    flat = " ".join(body.split())

    # Name the case in the sentence that introduces it, not merely somewhere in
    # the section -- "docstring" recurs in the worked example below, so a
    # section-wide check passed with the opening reduced to "some edits".
    assert "a **docstring**-only edit prints and looks like a code change" in flat, (
        "name the case the grep cannot classify"
    )
    assert "over-reports, never under-reports" in flat, (
        "say which way it errs, or this reads as the check being unsafe"
    )
    assert "grep '^@@'" in flat, "give the hunk-header check, not a description"
    # the worked example, so the current output is recognisable
    assert "aec8563" in flat and "@@ -6,8 +6,16 @@" in flat
    assert "`vllm_commit` stays at `acae5aa`" in flat, (
        "state the verdict, or the reader still does not know whether to bump"
    )


def test_the_results_table_does_not_credit_the_unbuilt_image():
    """"Measured ... with the image above" became false when the pin moved.

    The pins above now name 60166e19d45, and no image has been built from it
    (verified on the host: nothing in `docker images` carries that tag, and the
    serving container runs 0.21.0-e7929dcf5dcf...-c0c4842). Attributing the
    numbers to "the image above" credits an artifact that does not exist.

    The figures are still good -- the difference is the served decoder's
    weight-dtype plumbing, which does not move any default -- but the table has
    to say which image produced them.
    """
    readme = _readme()
    body = readme[readme.index("Measured on the delivery p150") :]
    body = body[: body.index("| TED 509 clips")]
    flat = " ".join(body.split())

    assert "with the image above" not in flat or "Not with the image" in flat, (
        "the pins above name an image that has not been built"
    )
    assert "e7929dcf5dcf" in flat, "name the image the numbers came from"
    assert "same defaults" in flat, (
        "say why the figures still stand, or this reads as invalidating them"
    )


def test_the_readme_admits_no_image_exists_at_the_current_pins():
    """The pin moved for a code change; the measurements predate it.

    The old wording said only that the branches were unpushed, which was true
    but no longer the whole story: bumping tt_metal_commit to pick up the
    served-decoder dtype fix means the image the numbers came from was built at
    the *previous* pin and does not contain it. Reporting figures under a pin
    nothing was built from, without saying so, is the sort of thing a reader
    reasonably assumes has been checked.
    """
    readme = _readme()
    body = readme[readme.index("**No image exists at these pins yet.**") :]
    body = body[: body.index("\n## Install")]
    flat = " ".join(body.split())

    assert "No image exists at these pins yet" in flat
    # which pin the numbers actually came from
    assert "e7929dcf5dcf" in flat, (
        "name the pin the measurements were taken at, or 'predates' is unfalsifiable"
    )
    # why the figures still stand, so this does not read as invalidating them
    assert "moves no default" in flat
    # and the ordering of what is left
    assert "push the branches, then" in flat, (
        "the rebuild depends on the push; give the order"
    )


@pytest.mark.parametrize("spec_id", ASR_SPEC_IDS)
def test_asr_spec_serves_the_batch_width_the_comment_measured(spec_id):
    """max_concurrency was the one declaration nothing asserted.

    The comment beside it justifies 4 with a throughput sweep (1->2->4 scales
    2.56 -> 4.59 -> 7.60 audio-s/s, and 8 regresses to 3.06 because prefill is
    run one user at a time), and it is also the customer's ASR_CONCURRENCY. It
    reaches vLLM as max_num_seqs, so changing it changes the operating point
    every number in the runbook was taken at -- and the harness defaults are
    pinned to it by tests/test_asr_harness_defaults_agree.py.
    """
    assert MODEL_SPECS[spec_id].device_model_spec.max_concurrency == 4


@pytest.mark.parametrize("spec_id", ASR_SPEC_IDS)
def test_asr_spec_caps_the_context_at_the_kv_budget_it_was_sized_for(spec_id):
    """max_context reaches vLLM as max_model_len and sizes the KV allocation.

    The adapter's get_max_tokens_all_users() returns max_model_len *
    max_num_seqs; at 2048 x 4 / block 64 that is the 128 (+padding) blocks the
    device is given. Before that method existed the plugin fell back to 131072
    tokens and overrode the block count to 2052 -- a 15x over-allocation, all
    of it written to the device as zeros at startup, i.e. pure start-up time.
    Raising max_context here scales that allocation linearly.
    """
    assert MODEL_SPECS[spec_id].device_model_spec.max_context == 2048


def test_the_multi_weight_template_still_pins_its_display_name():
    """Both weights must resolve to one display name, and it must be pinned.

    The template lists two weights, and without an explicit
    model_display_name the name is derived from whichever weight happens to be
    first -- so reordering the list would rename the model. The comment in the
    spec says exactly this; assert it rather than trusting the comment.
    """
    names = {MODEL_SPECS[spec_id].model_name for spec_id in ASR_SPEC_IDS}
    assert names == {"Qwen3-ASR-1.7B", "Qwen3-ASR-1.7B-JA"}, names

    spec_text = _spec_yaml()
    qwen = spec_text[spec_text.index("Qwen3-ASR served through the TT vLLM backend") :]
    assert "model_display_name: Qwen3-ASR-1.7B" in qwen, (
        "the display name is no longer pinned; it would follow the weight order"
    )


@pytest.mark.parametrize("spec_id", ASR_SPEC_IDS)
def test_asr_spec_declares_both_offline_switches(spec_id):
    """HF_HUB_OFFLINE alone does not stop transformers from reaching the hub.

    The weights are staged by the runbook and the container has no business
    contacting huggingface.co at start-up; the two variables cover the two
    libraries that would. Only HF_HUB_OFFLINE was ever asserted, so dropping
    the transformers one would have gone unnoticed.
    """
    env_vars = MODEL_SPECS[spec_id].env_vars
    assert env_vars["HF_HUB_OFFLINE"] == "1"
    assert env_vars["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("spec_id", ASR_SPEC_IDS)
def test_asr_spec_states_its_own_footprint(spec_id):
    """min_disk_gb / min_ram_gb must be declared, not inferred.

    ModelSpec derives them from param_count when they are absent, and
    infer_param_count() truncates "1.7B" to 1 -- so an inferred budget would be
    computed from a 1-billion-parameter model. The declarations are what keep
    that wrong number out of the disk and RAM guards.
    """
    spec = MODEL_SPECS[spec_id]
    assert spec.min_disk_gb == 15
    assert spec.min_ram_gb == 6


def test_the_footprint_is_declared_because_the_inference_would_be_wrong():
    """Guard the premise of the test above.

    If infer_param_count() ever learns to read 1.7, the declarations stop being
    load-bearing and this test says so instead of leaving the reasoning stale.
    """
    from workflows.model_spec import ModelSpec

    assert ModelSpec.infer_param_count("neosophie/Qwen3-ASR-1.7B-JA") == 1, (
        "the inference no longer truncates 1.7B to 1; revisit why the spec "
        "declares min_disk_gb / min_ram_gb explicitly"
    )


@pytest.mark.parametrize("spec_id", ASR_SPEC_IDS)
def test_asr_spec_is_the_default_impl_for_p150(spec_id):
    """Nothing else serves Qwen3-ASR, so it has to be the default.

    Without default_impl the resolver has no leaf to pick for P150 and the
    documented run command cannot name the model by weight alone.
    """
    assert MODEL_SPECS[spec_id].device_model_spec.default_impl is True


def _supervisor_env_reads():
    """Every ${VAR:-default} the supervisor script expands, with its default."""
    import re

    found = {}
    for var, default in re.findall(
        r"\$\{([A-Z_][A-Z0-9_]*):-([^}]*)\}", _supervisor()
    ):
        # first expansion wins; later ones reuse the resolved value
        found.setdefault(var, default)
    return found


def _supervisor_env_table():
    readme = _readme()
    start = readme.index("| variable | default | note |")
    table = readme[start:]
    return table[: table.index("\n\n")]


def test_the_runbook_documents_every_variable_the_supervisor_reads():
    """The table is the only place an operator learns what can be overridden.

    It listed nine of the ten and there was no test that it listed any
    particular number, so HF_TOKEN -- which the script exports into run.py's
    environment -- was absent with nothing to notice. Scan the script instead
    of restating its variables, so the eleventh cannot be missed either.
    """
    reads = _supervisor_env_reads()
    assert reads, "the scan found no ${VAR:-default} expansions; pattern is stale"

    table = _supervisor_env_table()
    missing = [var for var in sorted(reads) if f"`{var}`" not in table]
    assert not missing, (
        f"the supervisor reads {missing} but the runbook table does not list them"
    )


def test_the_supervisor_table_has_one_row_per_variable():
    """Two rows for one variable let a stale one survive beside a correct one."""
    table = _supervisor_env_table()
    for var in sorted(_supervisor_env_reads()):
        rows = [line for line in table.splitlines() if f"`{var}`" in line.split("|")[1]]
        assert len(rows) == 1, f"{var} has {len(rows)} rows in the table"


def test_the_first_transcription_range_covers_every_run_we_recorded():
    """The quoted range is a claim about our own measurements.

    It read "6m27s-6m35s over five runs" long after fourteen had been taken,
    two of which (6m40.0s and 6m44.8s) fell outside it -- the second measured
    in the very session that left the sentence alone. A range narrower than
    the observations tells the reader 6m36s is abnormal when it is not, and it
    is the stated basis for CANARY_FIRST_TIMEOUT.

    The runbook's own numbers are the evidence here: the worked example quotes
    6m33.591s, and the range has to contain it with room for the spread.
    """
    readme = _readme()
    row = _readme_row(readme, "`CANARY_FIRST_TIMEOUT`")

    bounds = re.search(r"measured (\d+)m(\d+)s[\u2013-](\d+)m(\d+)s", row)
    assert bounds, f"the row must state a measured range: {row[:200]}"
    low = int(bounds.group(1)) * 60 + int(bounds.group(2))
    high = int(bounds.group(3)) * 60 + int(bounds.group(4))
    assert low < high, (low, high)

    # the worked example elsewhere in the runbook must sit inside the range
    example = re.search(r"real (\d+)m([\d.]+)s` for the\s*\n?first transcription", readme)
    assert example, "the runbook no longer quotes a first-transcription time"
    example_s = int(example.group(1)) * 60 + float(example.group(2))
    assert low <= example_s <= high, (
        f"the worked example {example_s}s is outside the quoted range "
        f"{low}-{high}s"
    )

    # and the slowest run named in the row must be inside it too
    slowest = [
        int(m.group(1)) * 60 + float(m.group(2))
        for m in re.finditer(r"(\d+)m([\d.]+)s", row)
    ]
    assert slowest, row
    assert max(slowest) <= high, (
        f"the row names {max(slowest)}s but claims the range tops out at {high}s"
    )


def test_the_first_transcription_budget_clears_the_quoted_range():
    """CANARY_FIRST_TIMEOUT is justified by that range, so it must exceed it.

    Reading the bound out of the README rather than restating it means
    widening the range without revisiting the budget fails here.
    """
    row = _readme_row(_readme(), "`CANARY_FIRST_TIMEOUT`")
    bounds = re.search(r"measured \d+m\d+s[\u2013-](\d+)m(\d+)s", row)
    assert bounds, row[:200]
    high = int(bounds.group(1)) * 60 + int(bounds.group(2))

    budget = re.search(
        r'CANARY_FIRST_TIMEOUT="\$\{CANARY_FIRST_TIMEOUT:-(\d+)\}"', _supervisor()
    )
    assert budget, "the budget must stay a named, overridable value"
    assert int(budget.group(1)) > high, (
        f"the budget {budget.group(1)}s does not clear the quoted worst case {high}s"
    )
