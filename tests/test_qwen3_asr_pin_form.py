# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tt_metal_commit has to be a full SHA while the branch lives on a fork.

build_docker_images.resolve_commit_to_full_sha expands the pin with

    git ls-remote https://github.com/tenstorrent/tt-metal.git | grep <pin>

and takes the first hit. That is a substring match against UPSTREAM, which does
not carry this branch. A short pin therefore resolves to whatever upstream
object happens to contain those characters: "e7929dc" matched
7e7929dcd898... (refs/pull/9507/head) and the build failed cloning a commit the
fork does not have. A full 40-char SHA survives the grep as itself.
"""

import os
import re

HERE = os.path.dirname(__file__)
README = os.path.join(HERE, "..", "scripts", "qwen3_asr", "README.md")


def _readme():
    with open(README) as fh:
        return fh.read()


def _patch_block(readme):
    start = readme.index("git apply <<'PATCH'")
    return readme[start : readme.index("\nPATCH\n")]


def test_tt_metal_commit_is_a_full_sha():
    match = re.search(r'^\+  tt_metal_commit: "([0-9a-f]+)"', _readme(), re.M)
    assert match, "the runbook patch must set tt_metal_commit"
    pin = match.group(1)
    assert len(pin) == 40, (
        f"tt_metal_commit is {len(pin)} chars; it must be the full 40-char SHA, "
        "or ls-remote|grep against upstream can resolve it to another object"
    )


def test_build_metal_commit_matches_the_pin_exactly():
    """list_image_combinations filters with ==, not a prefix match."""
    readme = _readme()
    pin = re.search(r'^\+  tt_metal_commit: "([0-9a-f]+)"', readme, re.M).group(1)
    flag = re.search(r"--build-metal-commit (\S+)", readme)
    assert flag, "the build command must be documented"
    assert flag.group(1) == pin, (
        "--build-metal-commit is an exact-equality filter over catalog entries; "
        f"it says {flag.group(1)!r} but the pin is {pin!r}"
    )


def test_the_image_tags_carry_the_same_pin():
    """get_image_tags interpolates the pin verbatim into both tags."""
    readme = _readme()
    pin = re.search(r'^\+  tt_metal_commit: "([0-9a-f]+)"', readme, re.M).group(1)

    base = re.search(r"ci-build\.tags=local/tt-metal/tt-metalium/\S+?:(\S+?)\s", readme)
    assert base, "the bake command must tag the base image"
    assert base.group(1) == pin, (
        f"base image tag is {base.group(1)!r}, the pin is {pin!r}; "
        "the dev build looks the base image up by this exact tag"
    )

    dev = re.search(r"vllm-tt-metal-src-dev-\S+?:(\S+)", readme)
    assert dev, "the run command must name the dev image"
    assert pin in dev.group(1), (
        f"dev image tag {dev.group(1)!r} does not carry the pin {pin!r}"
    )


def test_the_readme_explains_why_a_short_pin_breaks():
    """Otherwise the next reader shortens it again for readability."""
    readme = _readme()
    assert "full 40-character SHA" in readme
    assert "ls-remote" in readme
    assert "refs/pull/9507/head" in readme, "keep the observed collision on record"


def test_the_no_runtime_diff_check_matches_the_stated_rule():
    """The command has to implement "not in the import graph", not "not a test".

    The rule the section states is import reachability, but the tt-metal command
    only filtered '/tests/'. Commits touching reference/dump_reference.py or
    eval/corpus_eval.py -- golden tooling and the offline eval, neither of which
    anything under tt/ imports -- therefore appeared as runtime diffs and would
    have forced a ~7 h rebuild that cannot change the served image.
    """
    readme = _readme()
    body = readme[readme.index("Why the pin may lag the branch head") :]

    assert "absence from the import graph" in body, "the rule must still be stated"
    for excluded in (
        "dump_reference",
        "extract_text_decoder",
        "corpus_eval",
    ):
        assert excluded in body, (
            f"{excluded}.py is not reachable from tt/, so the check must not "
            "report it as a runtime diff"
        )


def test_the_readme_says_how_to_confirm_an_exclusion():
    """A hardcoded exclusion list rots the moment tt/ starts importing one.

    Give the reader the check rather than asking them to trust the list.
    """
    body = _readme()
    body = body[body.index("Why the pin may lag the branch head") :]
    assert "grep -rn" in body
    assert "models/demos/audio/qwen3_asr/tt/" in body


def test_every_shell_variable_the_runbook_cds_into_is_defined():
    """`cd $VAR` with VAR unset lands in $HOME and the next command runs there.

    The runbook cd'd into $TT_METAL_HOME, $TT_INFERENCE_SERVER and
    $VLLM_TT_PLUGIN without ever saying what they are. Following it literally
    runs `docker buildx bake` and `pytest tests/tt` from the wrong directory.
    """
    import re

    readme = _readme()
    used = set(re.findall(r"cd \$([A-Z_]+)", readme))
    assert used, "the runbook does use cd $VAR; keep this check meaningful"

    for var in sorted(used):
        assert re.search(rf"^export {var}=", readme, re.M), (
            f"${var} is used with cd but never exported in the runbook"
        )


def test_the_runbook_names_the_branch_for_each_checkout():
    """Three trees, one branch name; a reader on the wrong one gets no error."""
    readme = _readme()
    head = readme[: readme.index("## Serving with")]
    assert "nyoshifujiTT/qwen3-asr-17b_p150x1" in head
    for repo in ("tt-metal", "tt-inference-server", "vllm-tt-plugin"):
        assert repo in head, f"{repo} must be listed among the checkouts"


def test_the_readme_does_not_claim_a_fork_clone_build_at_these_pins():
    """The pinned commits are not on the forks, so that build cannot have run.

    The results table said the numbers were "reproduced across three
    independent builds (loopback, fork clone, ...)". That was true of earlier
    pins. At the current ones the fork clone would fail -- the commits are not
    pushed -- so the current image was built over a local git daemon, and
    saying otherwise claims a reproduction nobody performed.
    """
    readme = _readme()
    body = readme[readme.index("Measured on the delivery p150") :]
    head = body[: body.index("## Install")]

    assert "reproduced across three independent builds" not in head, (
        "do not claim fork-clone reproduction at pins that are not pushed"
    )
    assert "git daemon" in head, "say how the current image was actually built"
    assert "not been executed as written at these pins" in head, (
        "name the one runbook step still owed once the branches are pushed"
    )
