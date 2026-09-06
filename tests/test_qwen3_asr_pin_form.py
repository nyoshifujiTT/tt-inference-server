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

from workflows.utils import get_repo_root_path

HERE = os.path.dirname(__file__)
README = os.path.join(HERE, "..", "scripts", "qwen3_asr", "README.md")


def _readme():
    with open(README) as fh:
        return fh.read()


def _patch_block(readme):
    start = readme.index("git apply <<'PATCH'")
    return readme[start : readme.index("\nPATCH\n")]


def _pinned_metal(readme):
    """tt_metal_commit as set by the runbook's own patch block.

    Scoped deliberately: the section above quotes the original PR #4837 recipe,
    which carries its own "+  ..._commit:" lines. A whole-file search would
    report whichever came first.
    """
    match = re.search(
        r'^\+  tt_metal_commit: "([0-9a-f]+)"', _patch_block(readme), re.M
    )
    assert match, "the runbook patch must set tt_metal_commit"
    return match.group(1)


def test_tt_metal_commit_is_a_full_sha():
    pin = _pinned_metal(_readme())
    assert len(pin) == 40, (
        f"tt_metal_commit is {len(pin)} chars; it must be the full 40-char SHA, "
        "or ls-remote|grep against upstream can resolve it to another object"
    )


def test_build_metal_commit_matches_the_pin_exactly():
    """list_image_combinations filters with ==, not a prefix match."""
    readme = _readme()
    pin = _pinned_metal(readme)
    flag = re.search(r"--build-metal-commit (\S+)", readme)
    assert flag, "the build command must be documented"
    assert flag.group(1) == pin, (
        "--build-metal-commit is an exact-equality filter over catalog entries; "
        f"it says {flag.group(1)!r} but the pin is {pin!r}"
    )


def test_the_image_tags_carry_the_same_pin():
    """get_image_tags interpolates the pin verbatim into both tags."""
    readme = _readme()
    pin = _pinned_metal(readme)

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
    # The wording moved when the pin was bumped for a code change: at these
    # pins nothing has been built at all, which is a stronger statement than
    # "one step is still owed". Accept either, so long as the runbook says
    # plainly that something is outstanding here.
    assert (
        "not been executed as written at these pins" in head
        or "No image exists at these pins yet" in head
    ), (
        "name the one runbook step still owed once the branches are pushed"
    )


def test_the_readme_covers_a_comment_only_change_to_a_served_module():
    """The filename filter cannot decide this case, and it came up.

    A tt-metal commit fixed a wrong issue number in a comment inside
    tt/generator_vllm.py -- a module the server does import. The documented
    check prints the filename, which reads as "bump the pin and rebuild ~7 h",
    but the stated rule is whether the server executes something different, and
    a comment does not. Give the check that settles it.
    """
    readme = _readme()
    body = readme[readme.index("Why the pin may lag the branch head") :]
    assert "changes only comments" in body
    # The check has to strip comment-only diff lines, not just look at names.
    # Requiring the whitespace class rather than a bare '^[+-]#': the anchored
    # form only matched column 0, so indented comments -- i.e. every comment
    # inside a function -- were reported as real code changes.
    assert "grep -vE '^[+-][[:space:]]*(#|$)'" in body
    assert "Anything printed is a real code change" in body


def _pinned_vllm(readme):
    """vllm_commit as set by the runbook's own patch block."""
    match = re.search(r'^\+  vllm_commit: "([0-9a-f]+)"', _patch_block(readme), re.M)
    assert match, "the runbook patch must set vllm_commit"
    return match.group(1)


def _sibling_checkout(name):
    """Locate a co-checked-out repo without hardcoding one machine's layout.

    The bring-up host keeps them at ~/<name>; the workstation at ~/repos/<name>.
    Returning a missing path is fine: _count treats a failed git call as
    "not available here" and the check skips rather than failing spuriously.
    """
    for candidate in (f"~/{name}", f"~/repos/{name}"):
        if os.path.isdir(os.path.expanduser(os.path.join(candidate, ".git"))):
            return candidate
    return f"~/{name}"


def test_the_quoted_test_counts_match_the_pinned_trees():
    """The counts illustrate "tests ship in the image", and they go stale.

    They were captured once and then drifted: the plugin figure said 28 while
    the image built from the current pin carries 30, because the pin moved from
    50695d8 to c0c4842 and the merge brought new test files. A reader who runs
    the quoted command sees a different number and cannot tell whether the
    image is wrong or the doc is.

    Derive the expected counts from the pinned trees so the doc fails here
    rather than in front of a reader.
    """
    import subprocess

    readme = _readme()
    counts = re.findall(r"\| wc -l\n([0-9]+)\n", readme)
    assert len(counts) == 2, f"expected two quoted counts, found {counts}"
    metal_quoted, plugin_quoted = (int(c) for c in counts)

    def _count(repo, pin, path, pattern):
        # Non-recursive: the quoted commands are `ls <dir>` and `ls <dir>/*.py`,
        # neither of which descends. `-r` would fold in tests/tt/ and report 42
        # where the reader sees 30.
        out = subprocess.run(
            ["git", "ls-tree", "--name-only", f"{pin}:{path}"],
            cwd=os.path.expanduser(repo), capture_output=True, text=True,
        )
        if out.returncode != 0:
            return None  # tree not available in this checkout; skip silently
        names = [n for n in out.stdout.split() if re.search(pattern, n)]
        return len(names)

    metal = _count(
        _sibling_checkout("tt-metal"), _pinned_metal(readme),
        "models/demos/audio/qwen3_asr/tests", r".",
    )
    if metal is not None:
        assert metal == metal_quoted, (
            f"the pinned tt-metal tree has {metal} files under qwen3_asr/tests, "
            f"the README says {metal_quoted}"
        )

    plugin = _count(_sibling_checkout("vllm-tt-plugin"), _pinned_vllm(readme), "tests", r"\.py$")
    if plugin is not None:
        assert plugin == plugin_quoted, (
            f"the pinned plugin tree has {plugin} .py files under tests/, "
            f"the README says {plugin_quoted}"
        )


def test_commands_reference_repo_files_by_a_path_that_resolves():
    """`sudo cp qwen3asr-supervisor.service ...` did not resolve from anywhere.

    The unit file lives in scripts/qwen3_asr/, and no block in this runbook cds
    there -- they all work from $TT_INFERENCE_SERVER. Following the Install
    section verbatim gives "No such file or directory".

    Generalised: every repo-relative file a command names must exist at the
    path given, resolved from the repository root.
    """
    import re

    root = get_repo_root_path()
    readme = _readme()

    # Arguments that look like paths in THIS repo. Deliberately narrow:
    #  - tests/tt/... belongs to vllm-tt-plugin, which the runbook also drives;
    #  - the extension must be a full suffix, or "....src.dev.Dockerfile"
    #    truncates to a name that does not exist.
    candidates = set(
        re.findall(
            r"(?:^|\s)(?:\$TT_INFERENCE_SERVER/)?"
            r"((?:scripts|workflows|reference_config|evals|vllm-tt-metal)"
            r"/[A-Za-z0-9_./-]+?\.(?:py|ya?ml|json|sh|service|Dockerfile|md))"
            r"(?=[\s\\)`]|$)",
            readme,
            re.M,
        )
    )
    assert candidates, "the runbook does name in-repo files; keep this meaningful"

    missing = sorted(p for p in candidates if not (root / p).exists())
    assert not missing, f"named in the runbook but absent from the repo: {missing}"


def test_the_unit_file_is_copied_from_where_it_lives():
    readme = _readme()
    assert "scripts/qwen3_asr/qwen3asr-supervisor.service" in readme, (
        "give the path the file is actually at; a bare filename resolves "
        "nowhere the runbook has cd'd to"
    )


def test_the_patch_targets_files_that_exist():
    """`git apply` fails on a path that is not in the tree.

    The generic path check cannot catch a typo here: the Dockerfile name
    appears four times in the patch (diff/---/+++/hunk context), so mutating
    one leaves three valid and the set-based check still passes. Take the
    targets from the diff headers, where each one must resolve.
    """
    import re

    root = get_repo_root_path()
    block = _patch_block(_readme())

    targets = set(re.findall(r"^diff --git a/(\S+) b/(\S+)$", block, re.M))
    assert targets, "the runbook patch must name the files it edits"

    for a, b in sorted(targets):
        assert a == b, f"the patch renames {a} -> {b}; that is not intended here"
        assert (root / a).exists(), (
            f"the patch edits {a}, which is not in the repository; git apply "
            "would fail before the build starts"
        )
