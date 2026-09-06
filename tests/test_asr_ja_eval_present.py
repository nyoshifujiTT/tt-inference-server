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


def test_the_runbook_says_how_to_build_the_two_manifests():
    """--manifest <corpus>/manifest.jsonl is unusable without a recipe.

    Neither TED nor MagicHub can be redistributed, so the manifests have to be
    rebuilt by the reader. The runbook quoted the resulting CERs and printed a
    command taking a manifest, but said nothing about how either corpus is
    assembled -- which of the two headline numbers is reproducible was then a
    matter of guesswork.
    """
    readme = _read(README)
    assert "Where the two manifests come from" in readme
    # the record format, or the reader cannot write one
    assert '"wav"' in readme and '"ref"' in readme
    # TED is reconstructed, not downloaded
    assert "compose_tedxjp10k.py" in readme
    assert "segments" in readme
    # MagicHub: which dataset, and the sampling that fixes 600
    assert "MagicHub/Japanese_Spontaneous_Conversation_Training_Dataset" in readme
    assert "seed 42" in readme


def test_the_runbook_records_why_a_mixed_track_corpus_is_excluded():
    """Otherwise the next reader repeats the CABank Sakura attempt.

    Its per-clip CER measured 2.0 -- not a model result but an artefact of one
    mixed track holding every speaker while the reference holds one line.
    """
    readme = _read(README)
    assert "CABank" in readme
    assert "mixed track" in readme


def test_the_runbook_says_to_discard_the_first_corpus_run():
    """Otherwise a warm-up artefact reads as a regression.

    The first TED pass on a freshly healthy server reported 19 failures; the
    second on the same server reported the steady 15. What moves is p99
    (16.637 s vs 8.847 s) -- kernel compilation and cache warm-up push the tail
    past the eval's 120 s timeout. corpus_cer is unaffected (0.1000 vs 0.1002)
    because it is computed over the clips that returned.
    """
    readme = _read(README)
    assert "Discard the first corpus run" in readme
    # the sentence wraps in the source, so normalise whitespace before matching
    flat = " ".join(readme.split())
    assert "run one measurement at a time" in flat
    # the evidence, so the next reader can tell warm-up from a real regression
    assert "16.637" in readme and "8.847" in readme, "keep the p99 pair on record"
    assert "490 / 19" in readme and "494 / 15" in readme


def test_the_runbook_does_not_promise_the_first_run_will_be_bad():
    """19 is what the first pass *can* be, not what it will be.

    Measured on a later restart: the first TED pass came in at 494 / 15, CER
    0.1002, p99 4.924 s -- the steady numbers -- because one golden clip had
    been transcribed before it, which pays the JIT compilation the first corpus
    run otherwise absorbs.

    Read as a promise, the 490/19 row makes a correct first pass look wrong,
    and hides the cheaper option: warm with one request instead of spending a
    six-minute corpus pass to do it.
    """
    readme = _read(README)
    flat = " ".join(readme.split())
    assert "The first run *can* land on the steady numbers" in flat, (
        "say the first pass is not necessarily the bad one"
    )
    # the counter-example, with its own measured numbers
    assert "4.924" in readme, "record the p99 of the good first pass"
    # and the cheaper alternative to burning a corpus pass
    assert "warm the server with one request" in flat, (
        "offer the one-request warm-up, not only 'discard the first pass'"
    )


def test_the_runbook_explains_the_fifteen_expected_ted_failures():
    """"494 ok / 15 download artifacts" was asserted, never evidenced.

    The 15 are zero-length wavs in the manifest; the server rejects them with
    HTTP 400. Recording the check keeps the next reader from chasing them.
    """
    readme = _read(README)
    assert "zero-length wavs" in readme
    assert "HTTP Error 400" in readme
    assert "frames 0" in readme
