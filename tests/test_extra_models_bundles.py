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
            .replace("<full 40-char tt-metal sha>", "a" * 40)
            .replace("<full 40-char vllm-tt-plugin sha>", "b" * 40)
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

    def test_committed_pins_are_abbreviated_upstream_shas(self):
        # The clone-URL check above guards only half of the patch. The pins are
        # the other half, and a fork pin leaks just as badly: the build resolves
        # whatever SHA the catalog names, so a committed branch SHA silently
        # ships a fork build. That is exactly what happened once -- the pins were
        # refreshed to this branch's merge commits, which are not on upstream's
        # default branch, while the URL check kept passing.
        #
        # A committed pin cannot be resolved offline, so this checks the property
        # that distinguishes the two cases without the network: BUILDING.md
        # requires a *full 40-char* SHA precisely because an abbreviated one only
        # resolves when the commit is already in the Dockerfile's shallow clone of
        # the default branch. So an abbreviated pin is by construction an upstream
        # default-branch commit, and a full 40-char pin in a committed file is the
        # signature of a fork pin that should have stayed a local patch.
        pins = self._committed_reranker_pins()
        assert pins, "no reranker pins found in the prod catalog"
        for key, value in pins.items():
            assert len(value) < 40, (
                f"{key} is pinned by full SHA ({value}) in a committed file. "
                "Full SHAs are only needed for commits that are not on the "
                "upstream default branch, i.e. fork pins, which BUILDING.md says "
                "must be applied as a local patch and reverted after the build."
            )

    def _committed_reranker_pins(self) -> dict[str, str]:
        spec = (
            self.REPO_ROOT / "workflows" / "model_specs" / "prod" / "embedding.yaml"
        ).read_text()
        start = spec.index("BAAI/bge-reranker-v2-m3")
        pins = {}
        for line in spec[start:].splitlines():
            match = re.match(r'\s*(tt_metal_commit|vllm_commit):\s*"([^"]+)"', line)
            if match:
                pins[match.group(1)] = match.group(2)
            if len(pins) == 2:
                break
        return pins

    def test_documented_patch_pins_full_shas(self):
        # The build resolves the pin with `git fetch --depth 1 origin <pin>`.
        # GitHub serves any full SHA that way, including one reachable only from
        # a topic branch, but an abbreviated SHA is not a ref and fails with
        # "couldn't find remote ref". Verified against this repo: 7, 8 and 12
        # character forms all fail; the 40 character form fetches and checks out.
        added = [
            ln
            for ln in self._documented_patch().splitlines()
            if ln.startswith("+") and ("tt_metal_commit" in ln or "vllm_commit" in ln)
        ]
        assert len(added) == 2, "the documented patch no longer sets both pins"
        for line in added:
            assert "full 40-char" in line, (
                f"documented pin does not call for a full SHA: {line.strip()}"
            )

    def test_documented_patch_covers_the_commit_pins(self):
        # Repointing the clones is only half of it: the build still resolves
        # whatever SHA the prod catalog pins, so the pins have to move with the
        # branch. Keeping them in the same patch means one `git apply` and one
        # `git checkout` cover everything, and no fork pin can be left behind in
        # a committed file.
        patch = self._documented_patch()
        assert "workflows/model_specs/prod/embedding.yaml" in patch, (
            "the documented patch does not touch the prod commit pins"
        )
        for key in ("tt_metal_commit", "vllm_commit"):
            assert any(
                ln.startswith("+") and key in ln for ln in patch.splitlines()
            ), f"the documented patch does not set {key}"

    def test_documented_build_produces_the_image_the_catalog_resolves(self):
        """The build step must ask for the release image.

        `build_docker_images.py` only builds the release image when `--release`
        is passed; otherwise it logs "Skipping release image build" and leaves
        just the dev one. But a vLLM model resolves to the *release* repo
        (model_spec.py::get_default_docker_image), so a doc that omits the flag
        sends the reader to serve an image the catalog never names -- or to no
        image at all.
        """
        doc = self.DOC.read_text()
        build_lines = [ln for ln in doc.splitlines() if "build_docker_images.py" in ln]
        assert build_lines, "BUILDING.md no longer documents a build command"
        for line in build_lines:
            assert "--release" in line, (
                f"documented build does not pass --release, so it never produces "
                f"the release image the catalog resolves to: {line.strip()}"
            )

    def test_documented_serve_uses_the_release_image(self):
        """The serve step must point at the release tag, for the same reason."""
        doc = self.DOC.read_text()
        serve_lines = [ln for ln in doc.splitlines() if "--override-docker-image" in ln]
        assert serve_lines, "BUILDING.md no longer documents serving"
        # Check the argument itself, not the prose around it: an earlier version
        # of this test looked at the following 120 characters and passed on the
        # wrong command purely because a later sentence said "release".
        for line in serve_lines:
            argument = line.split("--override-docker-image", 1)[1].strip()
            assert "release" in argument, (
                f"the documented serve command does not pass the release tag: {line.strip()}"
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

    def test_the_reranker_is_registered_only_through_its_bundle(self):
        """The in-tree tt-vllm-plugin must not also register the reranker.

        That package is a different plugin: this image clones the standalone
        tenstorrent/vllm-tt-plugin, and the only Dockerfiles that install
        ``tt-vllm-plugin/`` belong to tt-media-server. Registering the reranker
        there therefore ran nowhere, while looking like the wiring that makes
        serving work -- and it duplicated what the bundle already does, so the
        two could silently disagree about which class serves the architecture.
        """
        registry = BUNDLE_ROOT.parents[1] / "tt-vllm-plugin" / "tt_vllm_plugin" / "__init__.py"
        if not registry.is_file():
            pytest.skip("in-tree tt-vllm-plugin is not present")
        assert "bge_reranker" not in registry.read_text(), (
            "the reranker is registered in the in-tree tt-vllm-plugin, which this "
            "image does not install; EXTRA_MODELS_DIR is the single registration point"
        )
