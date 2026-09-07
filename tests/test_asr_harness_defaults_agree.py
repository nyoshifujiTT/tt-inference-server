# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Every ASR harness in this bring-up must default to the runbook's server.

tests/test_asr_perf_probes_present.py already requires this of the two probes:
the runbook targets 8110 throughout, and the served name carries the
``neosophie/`` owner, so a probe that defaults elsewhere either 404s or
measures nothing when run without arguments.

That reasoning is a property of *the runbook*, not a property of those two
files -- it applies just as much to the corpus eval and the LibriSpeech
benchmark, which the runbook quotes side by side with the probes. It was
applied to only half the harnesses, and the halves that escaped had drifted:
asr_ja_eval.py defaulted to 8101 (the supervisor's port, reachable only under
the systemd unit) and asr_openai_benchmark.py to 8100 (used nowhere at all)
with a --model missing the owner prefix.

The scan below discovers the harnesses rather than listing them, so a new one
cannot be added outside the rule.
"""

import ast
import os
import re

HERE = os.path.dirname(__file__)
BENCH_DIR = os.path.join(HERE, "..", "reference_config", "benchmarking")
EVAL_DIR = os.path.join(HERE, "..", "reference_config", "evals")
README = os.path.join(HERE, "..", "scripts", "qwen3_asr", "README.md")

RUNBOOK_HOST = "http://127.0.0.1:8110"
SERVED_MODEL = "neosophie/Qwen3-ASR-1.7B-JA"
# max_num_seqs in the P150 spec, and the concurrency the customer runs.
SERVER_BATCH_WIDTH = 4

# The transcription endpoint every harness drives. Used to discover them.
ENDPOINT = "/v1/audio/transcriptions"


def _read(path):
    with open(path) as fh:
        return fh.read()


def asr_harnesses():
    """Every committed harness that POSTs to the transcription endpoint."""
    found = []
    for directory in (BENCH_DIR, EVAL_DIR):
        for name in sorted(os.listdir(directory)):
            if not name.startswith("asr_") or not name.endswith(".py"):
                continue
            path = os.path.join(directory, name)
            if ENDPOINT in _read(path):
                found.append(path)
    return found


def _argparse_defaults(src):
    """Defaults of ap.add_argument("--flag", default=...) via the AST."""
    out = {}
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                out[flag[2:].replace("-", "_")] = kw.value.value
    return out


def _positional_defaults(src):
    """Defaults of NAME=sys.argv[i] if ... else <literal> (the probes' style)."""
    out = {}
    pattern = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=.*sys\.argv.*else\s+(.+?)\s*$")
    for line in src.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        name, literal = match.groups()
        literal = literal.rstrip(")")
        try:
            out[name] = ast.literal_eval(literal)
        except (ValueError, SyntaxError):
            continue
    return out


def harness_defaults(path):
    """The host / model / concurrency a harness uses with no arguments."""
    src = _read(path)
    flags = _argparse_defaults(src)
    positional = _positional_defaults(src)
    return {
        "host": flags.get("host", positional.get("HOST")),
        "model": flags.get("model", positional.get("MODEL")),
        "concurrency": flags.get("concurrency", positional.get("C")),
    }


def test_the_scan_finds_every_harness_the_runbook_quotes():
    """Guard the discovery itself: a missed file would vacuously pass below."""
    names = {os.path.basename(p) for p in asr_harnesses()}
    assert names == {
        "asr_perf_probe.py",
        "asr_perf_stream.py",
        "asr_ja_eval.py",
        "asr_openai_benchmark.py",
    }, names


def test_every_harness_defaults_to_the_runbook_host():
    """8101 is the systemd unit's port and 8100 is nobody's; the runbook is 8110."""
    for path in asr_harnesses():
        host = harness_defaults(path)["host"]
        assert host == RUNBOOK_HOST, (
            f"{os.path.basename(path)}: default host {host!r} is not the "
            f"runbook's {RUNBOOK_HOST!r}"
        )


def test_the_runbook_still_targets_the_host_the_defaults_claim():
    """If the runbook moves off 8110, this rule has to move with it.

    Pinning the harnesses to a constant the runbook has abandoned would keep
    the tests green while making every documented command wrong.
    """
    readme = _read(README)
    assert RUNBOOK_HOST in readme, (
        f"the runbook no longer mentions {RUNBOOK_HOST}; update the rule"
    )
    for path in asr_harnesses():
        assert os.path.basename(path) in readme, (
            f"{os.path.basename(path)} is not mentioned by the runbook"
        )


