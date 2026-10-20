# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""DER scoring executed inside the EVALS_DIARIZATION venv.

pyannote.audio, pyannote.metrics and tt-metal's
``models.demos.audio.pyannote_diarization.accuracy`` are provisioned into their
own venv rather than the shared WORKFLOW_RUN_SCRIPT one, the same way the
embedding eval keeps mteb in EVALS_EMBEDDING. The served model is called from
the workflow process; this module only answers the questions that need those
packages, as JSON between marker lines on stdout:

``plan``
    What to send: the bundled sample (with its annotation) or, when a corpus is
    configured, every ``audio/<id>.wav`` that has an RTTM, plus the thresholds
    the verdict is made against.
``sample``
    Where the bundled 30 s reference clip is, for the benchmark.
``score``
    Given the turns the server returned (JSON on stdin), the DER computed by
    tt-metal's accuracy module, so the served model and tt-metal's own tests
    are scored by identical code.
"""

from __future__ import annotations

import argparse
import json
import sys

from models.demos.audio.pyannote_diarization import accuracy

RESULT_START = "===DIARIZATION_RESULT_JSON_START==="
RESULT_END = "===DIARIZATION_RESULT_JSON_END==="


def plan(corpus_name: str, limit) -> dict:
    result = {
        "published_der": accuracy.PUBLISHED_DER,
        "published_der_ref": accuracy.PUBLISHED_DER_REF,
        "accuracy_der_max": accuracy.ACCURACY_DER_MAX,
        "corpus_der_tolerance": accuracy.CORPUS_DER_TOLERANCE,
        "sample_audio": accuracy.sample_audio_path(),
        "corpus_root": None,
        "corpus_audio": [],
        "published_corpus_der": None,
    }
    root = accuracy.corpus_root(corpus_name)
    if root:
        result["corpus_root"] = root
        result["corpus_audio"] = [
            wav for _, wav, _ in accuracy.corpus_files(root, limit)
        ]
        result["published_corpus_der"] = accuracy.published_corpus_der(corpus_name)
    return result


def score(request: dict) -> dict:
    turns_by_audio = request["turns"]
    root = request.get("corpus_root")
    if root:
        return accuracy.corpus_der(
            lambda wav: turns_by_audio[wav], root, limit=request.get("limit")
        )
    audio = accuracy.sample_audio_path()
    reference = accuracy.load_rttm(accuracy.sample_reference_path())
    return accuracy.score_against_reference(
        accuracy.turns_to_annotation(turns_by_audio[audio]), reference
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--corpus-name", required=True)
    plan_parser.add_argument("--limit", type=int, default=None)
    sub.add_parser("sample")
    sub.add_parser("score")
    args = parser.parse_args(argv)

    if args.command == "plan":
        result = plan(args.corpus_name, args.limit)
    elif args.command == "sample":
        result = {"sample_audio": accuracy.sample_audio_path()}
    else:
        result = score(json.load(sys.stdin))

    print(RESULT_START)
    print(json.dumps(result))
    print(RESULT_END)
    return 0


if __name__ == "__main__":
    sys.exit(main())
