# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""run_uvicorn_in_place.sh starts the server from its own directory.

Launchers such as tt-model's http-server kind exec a command line from their own
working directory. The server resolves /static, .env and its job database against
the working directory and imports its packages as top-level modules, so the
script has to move there first, whatever the caller's cwd, and pass the caller's
host/port through to uvicorn.
"""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "run_uvicorn_in_place.sh"


def _run(tmp_path, *args):
    """Run the script with `python` stubbed to report its cwd and argv."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text('#!/bin/bash\necho "CWD=$(pwd)"\necho "ARGS=$*"\n')
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    result = subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_runs_from_the_server_directory_whatever_the_callers_cwd(tmp_path):
    out = _run(tmp_path, "--host", "0.0.0.0", "--port", "8000")
    assert out["CWD"] == str(SCRIPT.parent)
    assert (Path(out["CWD"]) / "main.py").is_file()


def test_serves_main_app_with_the_callers_host_and_port(tmp_path):
    out = _run(tmp_path, "--host", "0.0.0.0", "--port", "20001")
    assert out["ARGS"] == (
        "-m uvicorn main:app --lifespan on --host 0.0.0.0 --port 20001"
    )


def test_follows_a_symlink_to_the_real_tree(tmp_path):
    """A launcher may put a link to the script on its PATH; the server still has to
    start from the tree the link points at."""
    link = tmp_path / "serve.sh"
    link.symlink_to(SCRIPT)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text('#!/bin/bash\necho "CWD=$(pwd)"\n')
    (bin_dir / "python").chmod(0o755)
    result = subprocess.run(
        ["bash", str(link), "--port", "1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"CWD={SCRIPT.parent}"
