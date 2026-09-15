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


def test_every_harness_defaults_to_the_served_model_name():
    """The owner prefix is part of the served name; without it the server 404s."""
    for path in asr_harnesses():
        model = harness_defaults(path)["model"]
        assert model == SERVED_MODEL, (
            f"{os.path.basename(path)}: default model {model!r} is not the "
            f"served {SERVED_MODEL!r}"
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

def test_every_harness_defaults_to_the_server_batch_width():
    """A default of 1 measures batch-1 while the server is configured for 4.

    Every figure in the runbook is taken at concurrency 4 -- it is both
    max_num_seqs in the P150 spec and the customer's setting -- so a harness
    that defaults to 1 silently reports a different operating point than the
    one being certified.
    """
    for path in asr_harnesses():
        concurrency = harness_defaults(path)["concurrency"]
        assert concurrency == SERVER_BATCH_WIDTH, (
            f"{os.path.basename(path)}: default concurrency {concurrency!r} is "
            f"not the server's batch width {SERVER_BATCH_WIDTH}"
        )


def test_the_batch_width_is_the_one_the_spec_serves():
    """Read max_concurrency off the spec rather than restating 4 here."""
    spec = _read(
        os.path.join(HERE, "..", "workflows", "model_specs", "dev", "audio_tts.yaml")
    )
    qwen = spec[spec.index("neosophie/Qwen3-ASR-1.7B-JA") :]
    match = re.search(r"^\s*max_concurrency:\s*(\d+)", qwen, re.M)
    assert match, "the Qwen3-ASR template no longer declares max_concurrency"
    assert int(match.group(1)) == SERVER_BATCH_WIDTH, (
        f"the spec now serves {match.group(1)} concurrent requests; the "
        f"harness defaults pin {SERVER_BATCH_WIDTH}"
    )


def test_the_defaults_are_reachable_through_the_parsers_we_can_import():
    """Assert on the parsed namespace, not only on the source text.

    The AST scan would keep passing if a harness read the flag but then
    overrode it, so check the parser's own answer where the module exposes one.
    """
    import argparse
    import importlib.util

    path = os.path.join(BENCH_DIR, "asr_openai_benchmark.py")
    spec = importlib.util.spec_from_file_location("_asr_bench_defaults", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args([])
    assert isinstance(args, argparse.Namespace)
    assert args.host == RUNBOOK_HOST
    assert args.model == SERVED_MODEL
    assert args.concurrency == SERVER_BATCH_WIDTH


def _response_format_choices(path):
    """The choices= list on a harness's --response-format flag."""
    for node in ast.walk(ast.parse(_read(path))):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "add_argument":
            continue
        if not node.args or getattr(node.args[0], "value", None) != "--response-format":
            continue
        for kw in node.keywords:
            if kw.arg == "choices":
                return [elt.value for elt in kw.value.elts]
    return None


# What the served model actually accepts, measured against the running server:
#
#   json          HTTP 200
#   text          HTTP 200
#   verbose_json  HTTP 400  "Currently do not support verbose_json for
#                            neosophie/Qwen3-ASR-1.7B-JA"
#   srt / vtt     HTTP 400  "Currently only support response_format: `text`,
#                            `json` or `verbose_json`"
#
# verbose_json is in vLLM's schema but unimplemented for this model, and
# leaving it unimplemented is a settled decision, not an oversight.
SERVED_RESPONSE_FORMATS = ["json", "text"]


def test_no_harness_offers_a_response_format_the_server_rejects():
    """A choice that always 400s is worse than no choice at all.

    Both harnesses listed verbose_json, and the benchmark's help went further
    and told the reader it was how to get `duration`. Picking it fails every
    request. argparse choices= is a promise that the value works.
    """
    for path in asr_harnesses():
        choices = _response_format_choices(path)
        if choices is None:
            continue
        rejected = [c for c in choices if c not in SERVED_RESPONSE_FORMATS]
        assert not rejected, (
            f"{os.path.basename(path)} offers {rejected}, which the server "
            f"answers with HTTP 400"
        )


def test_the_harnesses_offer_the_same_response_formats():
    """Two harnesses run against one server; a format works for both or neither."""
    offered = {
        os.path.basename(path): _response_format_choices(path)
        for path in asr_harnesses()
        if _response_format_choices(path) is not None
    }
    assert len(offered) == 2, offered
    values = list(offered.values())
    assert values[0] == values[1], offered


def test_no_harness_help_promises_the_unimplemented_format():
    """The help text is read more often than the choices list.

    "verbose_json -> duration" survived in the benchmark's help after the
    format stopped being reachable, which is how a reader would still be told
    to use it.
    """
    for path in asr_harnesses():
        src = _read(path)
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", None) != "add_argument":
                continue
            for kw in node.keywords:
                if kw.arg == "help" and isinstance(kw.value, ast.Constant):
                    assert "verbose_json ->" not in kw.value.value, (
                        f"{os.path.basename(path)}: the help still advertises "
                        f"verbose_json as usable: {kw.value.value!r}"
                    )
