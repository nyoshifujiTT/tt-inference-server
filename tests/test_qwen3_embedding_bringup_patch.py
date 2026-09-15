# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Guard rails for the Qwen3-Embedding p150x1 verification image build.

Building a verification image from our fork branches needs a temporary,
PR #4837-style manual patch (``git apply`` -> build -> ``git checkout``).  The
patch itself deliberately lives outside the repo, but the *properties it relies
on* are repo state, and those are what silently rot:

* ``build_docker_images.py`` must keep resolving/cloning tt-metal from a URL the
  patch can repoint.  If a future refactor moves the clone elsewhere, the patch
  stops steering the build and we would quietly build upstream code while
  believing we built ours.
* the builder must keep reading the *prod* catalog, and must keep queueing a
  build only for entries that pin both ``tt_metal_commit`` and ``vllm_commit``.
  That pair is precisely what the patch adds, so it is the mechanism that makes
  this model buildable at all.

These tests assert the checked-in tree keeps those hooks, and that no fork URL
or verification pin was ever committed by accident.
"""
from pathlib import Path

from workflows.utils import get_repo_root_path

REPO_ROOT = get_repo_root_path()
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_docker_images.py"
SRC_DEV_DOCKERFILE = REPO_ROOT / "vllm-tt-metal" / "vllm.tt-metal.src.dev.Dockerfile"
PROD_EMBEDDING_YAML = REPO_ROOT / "workflows" / "model_specs" / "prod" / "embedding.yaml"

FORK_OWNER = "nyoshifujiTT"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_builder_clones_tt_metal_from_a_single_patchable_url():
    """The patch repoints tt-metal in two places; keep both discoverable."""
    body = _read(BUILD_SCRIPT)
    assert body.count("https://github.com/tenstorrent/tt-metal.git") == 2, (
        "build_docker_images.py no longer has exactly the two upstream tt-metal "
        "references (ls-remote probe + clone) that bringup_v5.patch repoints. "
        "Re-derive the patch before trusting any image built with it."
    )


def test_src_dev_dockerfile_clones_the_plugin_from_a_patchable_url():
    body = _read(SRC_DEV_DOCKERFILE)
    assert "git clone https://github.com/tenstorrent/vllm-tt-plugin.git" in body, (
        "the src.dev Dockerfile no longer clones vllm-tt-plugin from the upstream "
        "URL that bringup_v5.patch repoints; re-derive the patch."
    )


def test_builder_reads_the_prod_catalog():
    """MODEL_SPECS is prod-only, which is why the patch edits the prod catalog."""
    from workflows.model_spec import MODEL_SPECS

    assert MODEL_SPECS, "MODEL_SPECS is empty; the builder has no specs to build"
    prod_weights = _read(PROD_EMBEDDING_YAML)
    assert "Qwen/Qwen3-Embedding-0.6B" in prod_weights, (
        "the prod catalog no longer carries a Qwen3-Embedding-0.6B entry for the "
        "patch to borrow as the build trigger"
    )


def test_a_build_is_only_queued_when_both_commits_are_pinned():
    """The pinned pair is the patch's mechanism; assert the builder still needs it."""
    from workflows.model_spec import MODEL_SPECS
    import importlib.util

    spec = importlib.util.spec_from_file_location("_bdi", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    combos = module.list_image_combinations(MODEL_SPECS)
    for combo in combos:
        assert all(combo), f"a build was queued with an unpinned commit: {combo!r}"

    unpinned = [k for k, c in MODEL_SPECS.items() if c.vllm_commit is None]
    assert unpinned, (
        "every spec now pins vllm_commit, so adding it no longer distinguishes the "
        "entry we build; revisit how bringup_v5.patch triggers the build"
    )


def test_no_fork_url_or_verification_pin_is_committed():
    """The manual patch must never land in the tree."""
    for path in (BUILD_SCRIPT, SRC_DEV_DOCKERFILE, PROD_EMBEDDING_YAML):
        body = _read(path)
        assert FORK_OWNER not in body, (
            f"{path.name} references the {FORK_OWNER} fork; the temporary bringup "
            "patch was committed instead of being reverted with git checkout"
        )
