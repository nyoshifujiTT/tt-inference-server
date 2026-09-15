#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Install a simple lmms-eval model into the lmms_eval package of *this* venv.

Run with the target venv's interpreter:

    <venv-python> evals/lmms_eval_models/install.py <model-source.py>

Copies the model module into lmms_eval/models/simple/ and registers it in the
simple-model registry (lmms_eval/models/__init__.py). Idempotent.

This lives in a script rather than inline in workflow_venvs so the setup step
is a single run_command call like every other setup_* step -- the resolution of
the lmms_eval install path has to happen inside the venv anyway.
"""

import os
import shutil
import sys

REGISTRY_ANCHOR = '    "whisper_tt": "WhisperTT",\n'
REGISTRY_ENTRIES = {
    "qwen3_asr_openai": '    "qwen3_asr_openai": "Qwen3ASROpenAI",\n',
}


def main(argv):
    if len(argv) != 2:
        sys.exit(f"usage: {argv[0]} <model-source.py>")
    src = argv[1]
    name = os.path.splitext(os.path.basename(src))[0]
    if not os.path.exists(src):
        sys.exit(f"model source not found: {src}")
    entry = REGISTRY_ENTRIES.get(name)
    if entry is None:
        sys.exit(f"no registry entry known for model {name!r}")

    try:
        import lmms_eval
    except ImportError as exc:
        sys.exit(f"lmms_eval is not installed in this interpreter: {exc}")

    pkg = os.path.dirname(lmms_eval.__file__)
    shutil.copyfile(src, os.path.join(pkg, "models", "simple", f"{name}.py"))

    init_path = os.path.join(pkg, "models", "__init__.py")
    with open(init_path) as fh:
        contents = fh.read()
    if name in contents:
        return 0
    if REGISTRY_ANCHOR not in contents:
        sys.exit("could not register model: whisper_tt anchor not found")
    with open(init_path, "w") as fh:
        fh.write(contents.replace(REGISTRY_ANCHOR, REGISTRY_ANCHOR + entry, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
