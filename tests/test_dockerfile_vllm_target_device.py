# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Guards how the dev image obtains vLLM and the TT platform.

The image installs the standalone ``tenstorrent/vllm-tt-plugin``, which owns the
vLLM version pin and its dependency overrides in ``docs/install-vllm-tt.sh``.
``TT_VLLM_COMMIT_SHA_OR_TAG`` therefore names a *plugin* commit.

An earlier bring-up shape cloned the ``tenstorrent/vllm`` fork instead and
installed its bundled ``plugins/vllm-tt-plugin``. These tests keep that shape
from creeping back: the fork carried no Qwen3-ASR change that the standalone
plugin lacks, and restating the install here would duplicate (and drift from)
the pin the plugin owns.
"""

import os
import re

from workflows.utils import get_repo_root_path

DOCKERFILE = (
    get_repo_root_path() / "vllm-tt-metal" / "vllm.tt-metal.src.dev.Dockerfile"
)


def _pinned_vllm_version():
    """The vLLM version the plugin's install script pins, read from the script.

    Not spelled out here on purpose. These tests exist to say "the plugin owns
    the pin, do not restate it", and they restated it: three docstrings claimed
    0.24.0 long after the plugin moved to 0.26.0, which is what the running
    server reports. Read it from the co-checked-out plugin when available, and
    skip rather than assert a stale literal when it is not.
    """
    import pytest

    for candidate in ("~/vllm-tt-plugin", "~/repos/vllm-tt-plugin"):
        script = os.path.expanduser(os.path.join(candidate, "docs/install-vllm-tt.sh"))
        if os.path.isfile(script):
            with open(script) as fh:
                match = re.search(r"vllm==([0-9][0-9.]*)", fh.read())
            if match:
                return match.group(1)
    pytest.skip("vllm-tt-plugin is not checked out beside this repo")


def _vllm_install_step() -> str:
    text = DOCKERFILE.read_text()
    match = re.search(
        r"RUN /bin/bash -c \"git clone "
        r"https://github\.com/tenstorrent/vllm-tt-plugin\.git.*?\"\n",
        text,
        re.DOTALL,
    )
    assert match, "vllm-tt-plugin install RUN step not found in dev Dockerfile"
    return match.group(0)


def _tt_metal_build_step() -> str:
    text = DOCKERFILE.read_text()
    match = re.search(
        r"RUN /bin/bash -c \"git clone (?:--depth 1 )?"
        r"https://github\.com/tenstorrent-metal/tt-metal\.git.*?\"\n",
        text,
        re.DOTALL,
    )
    assert match, "tt-metal build RUN step not found in dev Dockerfile"
    return match.group(0)


def test_vllm_install_is_delegated_to_the_plugin_script():
    """The plugin owns the vLLM pin; the Dockerfile must not restate it.

    ``docs/install-vllm-tt.sh`` installs the pinned vLLM with
    ``VLLM_TARGET_DEVICE=empty`` and removes the unusable torchaudio wheel.
    Spelling the version out here would let the two drift apart -- which is
    exactly what happened: this docstring said 0.24.0 while the plugin had
    moved to 0.26.0 and the running server reported v0.26.0.
    """
    step = _vllm_install_step()
    assert "source docs/install-vllm-tt.sh" in step, (
        "the vLLM install must be delegated to the plugin's install script"
    )
    tail = step.split("install-vllm-tt.sh", 1)[1]
    assert "uv pip install" not in tail, (
        "nothing may be installed after the plugin script; it owns the "
        "resolved environment"
    )
    # and the version must not be restated anywhere in the Dockerfile
    assert not re.search(r"vllm==[0-9]", DOCKERFILE.read_text()), (
        "the Dockerfile must not pin a vLLM version; the plugin script does"
    )


def test_no_test_here_hardcodes_the_pinned_vllm_version():
    """The version lives in the plugin script, so read it from there.

    Three docstrings in this file quoted 0.24.0 as the pinned release. The
    plugin now pins a different one, so the stated reasons for two of these
    rules -- "the fork's fix ships in the release the plugin pins" and "that
    release imports processors lazily" -- were arguments about a version no
    longer in use.
    """
    here = os.path.abspath(__file__)
    with open(here) as fh:
        lines = fh.read().splitlines()
    pinned = _pinned_vllm_version()

    # A superseded version may be named where the text is about the change
    # itself ("said 0.24.0 while the plugin had moved to..."). What may not
    # happen is a rule resting on it as the current pin, so require every
    # mention to carry a marker that it is historical.
    historical = (
        "moved to",
        "used to",
        "was established",
        "said ",
        "quoted ",
        "Stated as",
        "justified by",
    )
    offenders = []
    for i, line in enumerate(lines, 1):
        for match in re.finditer(r"(?:vllm==|vLLM )([0-9]+\.[0-9]+(?:\.[0-9]+)?)", line):
            version = match.group(1)
            if version == pinned:
                continue
            context = " ".join(lines[max(0, i - 4) : i + 1])
            if not any(marker in context for marker in historical):
                offenders.append(f"line {i}: {line.strip()}")
    assert not offenders, (
        f"the plugin pins {pinned}; these mentions read as the current pin "
        f"rather than as history: {offenders}"
    )


def test_runtime_target_device_stays_tt():
    text = DOCKERFILE.read_text()
    assert "VLLM_TARGET_DEVICE=tt" in text, (
        "the runtime ENV must keep VLLM_TARGET_DEVICE=tt so the TT platform is "
        "selected when the server runs"
    )


def test_the_tt_platform_comes_from_the_standalone_plugin():
    """The tt platform is advertised by the plugin's entry point.

    Without the plugin installed the container starts and then dies with
    "RuntimeError: Failed to infer device type".
    """
    step = _vllm_install_step()
    assert "vllm-tt-plugin.git ${vllm_tt_plugin_dir}" in step, (
        "the standalone tenstorrent/vllm-tt-plugin must be cloned"
    )
    assert "git checkout ${TT_VLLM_COMMIT_SHA_OR_TAG}" in step, (
        "TT_VLLM_COMMIT_SHA_OR_TAG must pin the plugin commit"
    )


def test_the_vllm_fork_is_not_cloned():
    """The fork shape must not come back.

    Every Qwen3-ASR change that lived on the fork branch has an equivalent on
    the standalone plugin, and the fork's HF-config fix ships in the upstream
    vLLM release the plugin pins (see ``docs/install-vllm-tt.sh``; it was
    0.24.0 when this was established and the fix is still present at the
    version pinned today, checked by
    ``test_the_pinned_release_still_carries_the_forks_fix``).
    """
    text = DOCKERFILE.read_text()
    assert "TT_VLLM_REPO_URL" not in text, (
        "no build arg may redirect a vLLM clone; the plugin owns the vLLM pin"
    )
    assert "tenstorrent/vllm.git" not in text, "the vLLM fork must not be cloned"
    assert "-e plugins/vllm-tt-plugin" not in text, (
        "the fork-bundled plugin copy must not be installed"
    )


def test_torchaudio_is_not_reinstalled_behind_the_plugin_script():
    """The plugin script uninstalls torchaudio on purpose.

    Its CUDA wheel cannot load next to the CPU torch tt-metal installs, and
    transformers>=5.12 imports it if it is merely present. An earlier shape
    installed it explicitly because vLLM used to import funasr_processor (and
    thus torchaudio) unconditionally; current vLLM imports processors lazily
    via ``__getattr__``, so that reason is gone. The lazy import is what makes
    the removal safe, so it is checked against the installed vLLM by
    ``test_the_pinned_release_still_imports_processors_lazily`` rather than
    asserted from a version number.
    """
    text = DOCKERFILE.read_text()
    assert "torchaudio" not in _vllm_install_step(), (
        "torchaudio must not be reinstalled after install-vllm-tt.sh removed it"
    )
    assert "TORCH_VERSION" not in text


def test_the_plugin_source_tree_is_copied_into_the_runtime_stage():
    """The plugin is an editable install, so its tree must survive the copy.

    It has to land at the same absolute path as in the builder or the .pth link
    dangles and ``import vllm_tt_plugin`` fails at runtime.
    """
    text = DOCKERFILE.read_text()
    assert "${vllm_tt_plugin_dir} ${vllm_tt_plugin_dir}" in text, (
        "the vllm-tt-plugin tree must be COPYed to the same path"
    )
    assert "${vllm_dir} ${vllm_dir}" not in text, (
        "the fork's vLLM tree must no longer be copied"
    )


def test_no_clone_url_is_a_build_arg():
    """Redirecting a clone belongs in a temporary patch, not a build arg.

    A bring-up whose commits are not upstream yet follows the recipe PR#4837
    established: ``git apply`` the clone-URL line, build, then ``git checkout``
    to restore it. Build args for this were added during bring-up and
    withdrawn -- they let an image that came from a fork claim to have come
    from the committed Dockerfile.
    """
    text = DOCKERFILE.read_text()
    for arg in ("TT_METAL_REPO_URL", "TT_VLLM_REPO_URL"):
        assert arg not in text, f"{arg} must not exist; patch the clone instead"


def test_the_clone_urls_are_the_upstream_ones():
    """The committed Dockerfile must always describe an upstream build."""
    step = _tt_metal_build_step()
    assert "https://github.com/tenstorrent-metal/tt-metal.git" in step, (
        "tt-metal must be cloned from upstream in the committed file"
    )
    # upstream clones shallow and then fetches just the pinned commit: a
    # full-history clone has taken over an hour on CI and dropped with
    # "fatal: early EOF"
    assert "--depth 1" in step, "the shallow clone must not be undone"
    assert (
        "git clone https://github.com/tenstorrent/vllm-tt-plugin.git"
        in _vllm_install_step()
    ), "the plugin must be cloned from upstream in the committed file"


def test_the_pinned_release_still_imports_processors_lazily():
    """The torchaudio removal is only safe while processors import lazily.

    The rule above used to be justified by "vLLM 0.24.0 imports processors
    lazily", which stops being an argument the moment the pin moves. Check the
    consequence rather than the shape: importing the processors package must
    not pull in funasr_processor, and must not pull in torchaudio, because
    that is what makes removing the torchaudio wheel safe.

    Checking for a ``__getattr__`` in the source was not enough -- pointing the
    same check at ``vllm.envs``, which also has one, still passed.
    """
    import subprocess
    import sys

    import pytest

    try:
        import vllm  # noqa: F401
    except ImportError:
        pytest.skip("vLLM is not importable in this interpreter")

    # A fresh interpreter, or an earlier test in this session may already have
    # imported the module under test and made the check vacuous. Naming the
    # module in the probe is also what makes the check specific: pointing the
    # import at an unrelated lazy module (vllm.envs) passed the earlier form.
    probe = (
        "import sys, vllm.transformers_utils.processors as p;"
        "assert p.__name__ == 'vllm.transformers_utils.processors';"
        "print('funasr' if any(m.endswith('processors.funasr_processor')"
        " for m in sys.modules) else 'no-funasr');"
        "print('torchaudio' if 'torchaudio' in sys.modules else 'no-torchaudio');"
        "print('lazy' if hasattr(p, '__getattr__') else 'eager')"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-2000:]
    lines = out.stdout.split()
    assert "no-funasr" in lines, (
        "funasr_processor is imported eagerly; removing torchaudio would break it"
    )
    assert "no-torchaudio" in lines, (
        "importing the processors package pulled in torchaudio, which the "
        "plugin script uninstalls"
    )
    assert "lazy" in lines, "the processors package no longer defers its imports"


def test_the_pinned_release_still_carries_the_forks_fix():
    """The fork was retired because its HF-config fix landed upstream.

    Stated as "ships in the 0.24.0 release the plugin pins", which says nothing
    about the release pinned today. The fix is that an audio model's HF config
    resolves without the fork's patch, so name the classes that have to be
    there: a bare ``import`` of the module passed even when pointed at a module
    that does not exist, because ImportError is the skip condition.
    """
    import pytest

    try:
        import vllm  # noqa: F401
    except ImportError:
        pytest.skip("vLLM is not importable in this interpreter")

    from vllm.transformers_utils.configs import qwen3_asr

    for name in ("Qwen3ASRConfig", "Qwen3ASRTextConfig", "Qwen3ASRAudioEncoderConfig"):
        assert hasattr(qwen3_asr, name), (
            f"upstream vLLM no longer defines {name}; the fork's HF-config fix "
            f"is not in the pinned release"
        )
