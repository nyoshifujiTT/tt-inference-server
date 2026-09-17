# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Run evals/lmms_eval_models/install.py, which nothing else executes.

This script is ours and it is what put the Qwen3-ASR lmms-eval adapter into the
eval venv. Every reference to it elsewhere in the suite is either prose about
the venv hook that upstream deleted or a containment check on the README, so
the code itself -- the copy, the registry edit, and the refusals it documents
-- had no coverage at all.

It edits a *third-party* package in place (lmms_eval/models/__init__.py), so
the two properties that matter are that the edit lands next to the anchor it
claims to anchor on, and that a second run does not duplicate the entry.
"""

import ast
import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
INSTALL = os.path.join(HERE, "..", "evals", "lmms_eval_models", "install.py")
ADAPTER = os.path.join(HERE, "..", "evals", "lmms_eval_models", "qwen3_asr_openai.py")


def _install_module():
    spec = importlib.util.spec_from_file_location("asr_install_under_test", INSTALL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_lmms_eval(tmp_path, monkeypatch):
    """A stand-in lmms_eval package, importable for the duration of one test.

    The script resolves the install path through ``import lmms_eval``, so the
    only way to run it without the real (large, forked) package is to put a
    minimal one on sys.path. The registry file carries the upstream anchor
    verbatim, because that string is the contract.
    """
    package = tmp_path / "lmms_eval"
    (package / "models" / "simple").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    registry = package / "models" / "__init__.py"
    registry.write_text(
        'AVAILABLE_SIMPLE_MODELS = {\n    "whisper_tt": "WhisperTT",\n}\n'
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [
        n for n in sys.modules if n == "lmms_eval" or n.startswith("lmms_eval.")
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return package


def test_the_install_copies_the_adapter_and_registers_it(fake_lmms_eval):
    module = _install_module()
    assert module.main(["install.py", ADAPTER]) == 0

    copied = fake_lmms_eval / "models" / "simple" / "qwen3_asr_openai.py"
    assert copied.exists(), "the adapter must be copied into the package"
    # the same file, not a stub: lmms-eval imports the class out of it
    with open(ADAPTER) as fh:
        assert copied.read_text() == fh.read()

    registry = (fake_lmms_eval / "models" / "__init__.py").read_text()
    assert '"qwen3_asr_openai": "Qwen3ASROpenAI"' in registry, (
        "the class name here must match the adapter, or model resolution fails"
    )
    # and it must be registered *inside* the dict, next to the anchor, not
    # appended after the closing brace where it would be a syntax error
    assert (
        registry.index('"whisper_tt"')
        < registry.index('"qwen3_asr_openai"')
        < registry.index("}")
    )


def test_the_registry_stays_importable_python(fake_lmms_eval):
    """The edit is a string replace into someone else's source file."""
    module = _install_module()
    assert module.main(["install.py", ADAPTER]) == 0
    registry = (fake_lmms_eval / "models" / "__init__.py").read_text()
    ast.parse(registry)


def test_the_registered_class_is_the_one_the_adapter_defines(fake_lmms_eval):
    """A registry entry naming a class that does not exist fails at load time."""
    module = _install_module()
    assert module.main(["install.py", ADAPTER]) == 0
    registry = (fake_lmms_eval / "models" / "__init__.py").read_text()

    with open(ADAPTER) as fh:
        classes = {
            node.name
            for node in ast.parse(fh.read()).body
            if isinstance(node, ast.ClassDef)
        }
    named = registry.split('"qwen3_asr_openai": "')[1].split('"')[0]
    assert named in classes, f"{named} is not defined in the adapter: {sorted(classes)}"


def test_a_second_install_does_not_duplicate_the_entry(fake_lmms_eval):
    """The docstring promises idempotence; setup steps get re-run."""
    module = _install_module()
    assert module.main(["install.py", ADAPTER]) == 0
    assert module.main(["install.py", ADAPTER]) == 0

    registry = (fake_lmms_eval / "models" / "__init__.py").read_text()
    assert registry.count("qwen3_asr_openai") == 1, (
        f"a re-run duplicated the entry:\n{registry}"
    )


def test_a_missing_anchor_is_refused_rather_than_silently_skipped(fake_lmms_eval):
    """If upstream drops whisper_tt, the edit must fail loudly.

    Writing the entry anyway (or returning 0) would leave the eval to fail much
    later with an unresolvable model name.
    """
    registry = fake_lmms_eval / "models" / "__init__.py"
    registry.write_text("AVAILABLE_SIMPLE_MODELS = {}\n")

    module = _install_module()
    with pytest.raises(SystemExit) as excinfo:
        module.main(["install.py", ADAPTER])
    assert "whisper_tt" in str(excinfo.value), (
        f"the message must name the anchor that moved: {excinfo.value}"
    )
    assert "qwen3_asr_openai" not in registry.read_text(), (
        "nothing may be written when the anchor is absent"
    )


def test_an_unknown_model_name_is_refused(fake_lmms_eval, tmp_path):
    """The registry entry is keyed by filename; an unknown one has no class."""
    stray = tmp_path / "whisper_xl.py"
    stray.write_text("class WhisperXL:\n    pass\n")

    module = _install_module()
    with pytest.raises(SystemExit) as excinfo:
        module.main(["install.py", str(stray)])
    assert "whisper_xl" in str(excinfo.value)


def test_a_missing_source_is_refused_before_the_package_is_touched(fake_lmms_eval):
    module = _install_module()
    with pytest.raises(SystemExit) as excinfo:
        module.main(["install.py", os.path.join(HERE, "no_such_adapter.py")])
    assert "not found" in str(excinfo.value)
    assert not list((fake_lmms_eval / "models" / "simple").iterdir()), (
        "a bad argument must not leave a partial install behind"
    )


def test_wrong_argument_count_prints_the_usage(fake_lmms_eval):
    module = _install_module()
    with pytest.raises(SystemExit) as excinfo:
        module.main(["install.py"])
    assert "usage:" in str(excinfo.value)
