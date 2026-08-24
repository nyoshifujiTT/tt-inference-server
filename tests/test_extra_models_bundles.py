# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
"""Validate the EXTRA_MODELS_DIR bundles shipped for the canonical vLLM plugin.

The standalone ``vllm-tt-plugin`` (upstream vLLM 0.24) discovers models by
scanning ``EXTRA_MODELS_DIR`` for ``<name>/vllm_metadata.json`` files, each of
which must carry an ``arch`` (HF architecture) and a ``main_class``
(``"module:Class"``). These tests keep the shipped bundles well-formed so the
plugin can register them without editing plugin source.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest

BUNDLE_ROOT = Path(__file__).resolve().parents[1] / "vllm-tt-metal" / "extra_models"

_MODULE_CLASS_RE = re.compile(r"^[\w.]+:[\w]+$")


def _bundle_dirs():
    if not BUNDLE_ROOT.is_dir():
        return []
    return sorted(p for p in BUNDLE_ROOT.iterdir() if p.is_dir())


def test_bundle_root_exists():
    assert BUNDLE_ROOT.is_dir(), f"missing bundle root {BUNDLE_ROOT}"


def test_at_least_reranker_bundle_present():
    names = {p.name for p in _bundle_dirs()}
    assert "bge-reranker-v2-m3" in names


@pytest.mark.parametrize("bundle", _bundle_dirs(), ids=lambda p: p.name)
def test_bundle_metadata_wellformed(bundle):
    meta_path = bundle / "vllm_metadata.json"
    assert meta_path.is_file(), f"{bundle.name}: missing vllm_metadata.json"
    data = json.loads(meta_path.read_text())
    assert isinstance(data.get("arch"), str) and data["arch"], (
        f"{bundle.name}: 'arch' must be a non-empty string"
    )
    main_class = data.get("main_class")
    assert isinstance(main_class, str) and _MODULE_CLASS_RE.match(main_class), (
        f"{bundle.name}: 'main_class' must be 'module.path:ClassName', got {main_class!r}"
    )


def test_reranker_bundle_targets_expected_arch_and_class():
    meta = json.loads(
        (BUNDLE_ROOT / "bge-reranker-v2-m3" / "vllm_metadata.json").read_text()
    )
    assert meta["arch"] == "XLMRobertaForSequenceClassification"
    assert meta["main_class"] == (
        "models.demos.bge_reranker_v2_m3.demo.generator_vllm:BgeRerankerV2M3"
    )


class TestRerankerBuildingDoc:
    """BUILDING.md carries the patch used to build the image from an unmerged
    branch. A patch in prose rots silently, so check it still applies and that
    no fork pin has leaked into the committed catalog."""

    DOC = BUNDLE_ROOT / "bge-reranker-v2-m3" / "BUILDING.md"
    REPO_ROOT = BUNDLE_ROOT.parents[1]

    def _documented_patch(self) -> str:
        text = self.DOC.read_text()
        start = text.index("git apply <<'PATCH'\n") + len("git apply <<'PATCH'\n")
        return text[start : text.index("\nPATCH\n", start) + 1]

    def test_documented_patch_still_applies(self, tmp_path):
        # The placeholder stands in for the reader's own fork; any owner works.
        patch = (
            self._documented_patch()
            .replace("<you>", "someorg")
            .replace("<your-branch>", "some-branch")
        )
        patch_file = tmp_path / "fork.diff"
        patch_file.write_text(patch)

        result = subprocess.run(
            ["git", "apply", "--check", str(patch_file)],
            cwd=self.REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "BUILDING.md's patch no longer applies to "
            f"vllm.tt-metal.src.dev.Dockerfile:\n{result.stderr}"
        )

    def test_dockerfile_clones_upstream_not_a_fork(self):
        dockerfile = (
            self.REPO_ROOT / "vllm-tt-metal" / "vllm.tt-metal.src.dev.Dockerfile"
        ).read_text()
        clone_lines = [ln for ln in dockerfile.splitlines() if "git clone" in ln]
        assert clone_lines
        for line in clone_lines:
            assert "nyoshifuji" not in line.lower(), (
                f"fork clone URL committed to the Dockerfile: {line.strip()}"
            )

    def test_documented_patch_switches_branch_too(self):
        # `git clone --depth 1` only fetches the default branch, and GitHub does
        # not serve arbitrary SHAs, so pointing the clone at a fork without also
        # selecting the branch leaves the pinned commit unreachable
        # ("couldn't find remote ref"). The patch must set --branch.
        added = [
            ln
            for ln in self._documented_patch().splitlines()
            if ln.startswith("+") and "git clone" in ln
        ]
        assert added, "the documented patch no longer changes any clone line"
        for line in added:
            assert "--branch" in line, (
                f"documented clone line does not select a branch: {line.strip()}"
            )


class TestBundlesReachTheImage:
    """A bundle only registers a model if it is actually in the image and
    EXTRA_MODELS_DIR points at it. Shipping the directory without the wiring
    fails at serve time with 'No TT model architecture is registered', long
    after the build looked fine."""

    DOCKERFILE = (
        BUNDLE_ROOT.parents[1] / "vllm-tt-metal" / "vllm.tt-metal.src.dev.Dockerfile"
    )

    def _dockerfile(self) -> str:
        return self.DOCKERFILE.read_text()

    def test_extra_models_dir_is_set_in_the_image(self):
        assert "EXTRA_MODELS_DIR=" in self._dockerfile(), (
            "EXTRA_MODELS_DIR is not set in the image; the plugin will not scan "
            "for bundles and every bundled model stays unregistered"
        )

    def test_bundles_are_copied_into_the_image(self):
        text = self._dockerfile()
        assert 'COPY --chown=container_app_user:container_app_user \\\n    "vllm-tt-metal/extra_models" ${EXTRA_MODELS_DIR}' in text, (
            "the extra_models directory is not COPYed into the image"
        )
