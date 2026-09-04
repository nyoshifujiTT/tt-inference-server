# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The corpus eval the runbook tells readers to run must live in the repo.

The accuracy numbers this bring-up is accepted on (TED CER 0.1002, MagicHub CER
0.1668) come from asr_ja_eval.py. The runbook printed a command for it while
the script existed only in a working directory on the bring-up host, so a
reader following the runbook got "No such file or directory" and no way to
reproduce the figures. Its sibling, asr_openai_benchmark.py, was committed
under reference_config/; this one belongs next to it.
"""

import ast
import os

HERE = os.path.dirname(__file__)
EVAL = os.path.join(HERE, "..", "reference_config", "evals", "asr_ja_eval.py")
README = os.path.join(HERE, "..", "scripts", "qwen3_asr", "README.md")


def _read(path):
    with open(path) as fh:
        return fh.read()


def test_the_corpus_eval_script_is_committed():
    assert os.path.exists(EVAL), (
        "asr_ja_eval.py must be in the repository; the runbook's accuracy "
        "section is unreproducible without it"
    )
    ast.parse(_read(EVAL))


def test_the_runbook_points_at_the_committed_path():
    """A bare "python3 asr_ja_eval.py" only works from an undisclosed cwd."""
    readme = _read(README)
    assert "reference_config/evals/asr_ja_eval.py" in readme
    assert "\npython3 asr_ja_eval.py" not in readme, (
        "the runbook must give the in-repo path, not a bare filename"
    )


def test_the_eval_speaks_the_multipart_transcription_api():
    """The whole reason the upstream harness is unusable here.

    The runbook explains that the upstream generic audio path POSTs a JSON body
    and gets HTTP 400 because vLLM wants multipart/form-data. The script the
    runbook substitutes has to be the one that does it correctly, otherwise the
    justification does not hold.
    """
    src = _read(EVAL)
    assert "/v1/audio/transcriptions" in src
    assert "multipart/form-data" in src


def test_the_eval_sends_the_customer_request_parameters():
    """The measured numbers are only comparable if the preset travels with it.

    Accuracy was accepted against the customer's (gbase-asr) request shape;
    dropping any of these silently changes what the CER means.
    """
    src = _read(EVAL)
    for field in (
        '"language"',
        '"to_language"',
        '"repetition_penalty"',
        '"max_completion_tokens"',
        '"temperature"',
    ):
        assert field in src, f"the customer preset field {field} must be sent"


def test_the_eval_reports_the_metrics_the_runbook_quotes():
    """corpus_cer is the figure the acceptance table cites."""
    src = _read(EVAL)
    for key in ("corpus_cer", "rtf_sum_lat_over_audio", "throughput_audio_per_s"):
        assert key in src
