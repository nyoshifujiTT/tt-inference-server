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
import re

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
    """The eval must emit every figure the acceptance table cites."""
    src = _read(EVAL)
    for key in ("corpus_cer", "rtf_sum_lat_over_audio", "throughput_audio_per_s"):
        assert key in src


def test_the_results_table_records_corpus_speed_not_only_accuracy():
    """The corpora were accepted on CER alone; their speed went unrecorded.

    asr_ja_eval.py reports throughput_audio_per_s for every run and the table
    kept only corpus_cer, so a rerun that transcribed correctly at half the
    speed matched the runbook exactly. The one speed figure that was recorded --
    LibriSpeech rtfx -- comes from a different harness on a different workload,
    so it cannot stand in for the corpora.
    """
    readme = _read(README)
    row_ted = [ln for ln in readme.splitlines() if ln.startswith("| TED 509 clips")]
    row_magic = [ln for ln in readme.splitlines() if ln.startswith("| MagicHub 600 clips")]
    assert row_ted and row_magic, "both corpus rows must be in the results table"
    for row in (row_ted[0], row_magic[0]):
        assert "audio-s/s" in row, f"record the throughput for this corpus: {row}"
        assert "p50" in row, f"and a latency percentile: {row}"


def test_the_recorded_corpus_throughput_is_consistent_with_its_own_inputs():
    """audio-s / wall-s must equal the quoted rate, or one of them is stale."""
    readme = _read(README)
    for name in ("TED 509 clips", "MagicHub 600 clips"):
        row = next(ln for ln in readme.splitlines() if ln.startswith(f"| {name}"))
        got = re.search(
            r"([\d.]+) audio-s in ([\d.]+) s wall = \*\*([\d.]+) audio-s/s\*\*", row
        )
        assert got, f"quote the inputs alongside the rate: {row}"
        audio, wall, rate = (float(g) for g in got.groups())
        assert abs(audio / wall - rate) < 0.01, (
            f"{name}: {audio}/{wall} = {audio / wall:.3f}, table says {rate}"
        )


def test_the_table_says_the_three_speed_figures_are_not_comparable():
    """They differ in harness and in workload, so a reader will compare them.

    LibriSpeech re-sends 32 downloaded clips to fill 128 requests
    (``samples[idx % len(samples)]``), while the corpora make one pass over 509
    and 600 distinct clips. Putting 3.96 next to 12.58 without that note reads
    as a threefold regression.
    """
    readme = _read(README)
    body = readme[readme.index("| TED 509 clips") :]
    body = body[: body.index("And the serving-level timings")]
    flat = " ".join(body.split())
    assert "not comparable to each other" in flat, "say the three rows are not one series"
    assert "re-sends the same audio" in flat, "name why the LibriSpeech row is higher"
    assert "Compare a rerun against the same row" in flat, "and what to do instead"


def test_the_benchmark_really_reuses_its_downloaded_clips():
    """The note above is only true while the benchmark cycles its samples."""
    src = _read(
        os.path.join(HERE, "..", "reference_config", "benchmarking", "asr_openai_benchmark.py")
    )
    assert "samples[idx % len(samples)]" in src, (
        "the runbook explains the LibriSpeech row by this reuse; if it stops "
        "cycling, the explanation stops holding"
    )


def test_the_table_warns_that_pre_fix_speed_figures_are_high():
    """Older runs are quoted in worklogs and will be compared against these."""
    readme = _read(README)
    body = readme[readme.index("| TED 509 clips") :]
    body = body[: body.index("And the serving-level timings")]
    flat = " ".join(body.split())
    assert "1892.0" in flat and "2212.0" in flat, "name the superseded totals"
    assert "1649.4" in flat and "1927.7" in flat, "and the measured ones"
    assert "CERs are unaffected" in flat, (
        "say the accuracy conclusions did not move, or the fix reads as invalidating them"
    )


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


def test_the_runbook_separates_transient_failures_from_the_permanent_fifteen():
    """The fail count is not fixed, and 15 was documented as though it were.

    asr_ja_eval.py records a failure and moves on -- it does not retry -- so a
    single network hiccup changes the number. Measured on this host: one pass
    returned 490 ok / 19 fail (corpus_cer 0.1000, audio_s 1636.1) and an
    immediate rerun against the same server returned 494 / 15, CER 0.1002,
    audio_s 1649.4, with the server's error/abort/length/repetition counters at
    0.0 the whole time. Reading 19 against a documented 15 looks like a
    regression and is not one.
    """
    readme = _read(README)
    body = readme[readme.index("494 ok / 15 download artifacts") :]
    body = body[: body.index("#### Where the two manifests come from")]
    flat = " ".join(body.split())

    assert "A higher fail count is not automatically a regression" in flat, (
        "say that the count moves, or 19 reads as a model failure"
    )
    assert "it does not retry" in flat, "name the reason the count moves"
    # the measured pair, so the reader can recognise the shape
    assert "490 ok / 19 fail" in flat and "494 / 15" in flat, (
        "quote both passes; one number alone does not show the count moving"
    )
    # and how to tell them apart -- by error text, not by counting
    assert "by their error text rather than by the count" in flat, (
        "give the discriminator"
    )


def test_the_runbook_explains_why_audio_s_moves_with_the_fail_count():
    """Otherwise the audio total looks like independent corroboration.

    audio_s sums the clips that succeeded, so it drops by exactly the audio of
    the extra failures (1649.4 - 1636.1 = 13.3 s over 4). A reader who treats
    it as a second signal concludes two things went wrong instead of one.
    """
    readme = _read(README)
    body = readme[readme.index("494 ok / 15 download artifacts") :]
    body = body[: body.index("#### Where the two manifests come from")]
    flat = " ".join(body.split())
    assert "sums the clips that succeeded" in flat
    assert "symptom of the same thing rather than separate evidence" in flat, (
        "say it is not independent corroboration"
    )
    # and the arithmetic, read back out of the runbook
    got = re.search(r"1649\.4 - 1636\.1 = ([\d.]+) s over the (\d+) extra failures", flat)
    assert got, "quote the arithmetic that ties the two together"
    assert abs(float(got.group(1)) - (1649.4 - 1636.1)) < 0.05


def test_the_harness_really_does_not_retry():
    """The runbook's explanation depends on this, so pin it.

    If a retry is ever added the guidance above becomes wrong -- a transient
    failure would no longer show up in the count at all.
    """
    src = _read(EVAL)
    body = src[src.index("def transcribe(") : src.index("# --- Japanese text normalization")]
    assert "for attempt" not in body and "retries" not in body, (
        "a retry loop would change what a non-zero fail count means; update the "
        "runbook's transient-vs-permanent guidance if one is added"
    )
    assert "ERROR: {e}" in body, (
        "a failure must be recorded with its reason, since the reason is what "
        "separates a transient failure from an empty wav"
    )


def _resolve_secs():
    """Compile resolve_secs out of the eval, so the real function is exercised."""
    tree = ast.parse(_read(EVAL))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_secs":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {}
            exec(compile(ast.fix_missing_locations(module), EVAL, "exec"), namespace)  # noqa: S102
            return namespace["resolve_secs"]
    raise AssertionError("resolve_secs not found")


def test_audio_duration_prefers_our_own_measurement():
    """usage.seconds is billing, in whole seconds, so it rounds every clip up.

    This eval computed wav_dur() for every clip and then threw it away, passing
    None as the fallback so the server's rounded figure always won. On TED-509
    that reported 1892.0 s of audio for files that measure 1649.4 s -- a 14.7%
    overstatement that flattered throughput_audio_per_s and
    rtf_sum_lat_over_audio by the same factor. corpus_cer is unaffected, and the
    per-clip durations written to the samples file stayed correct, so only the
    aggregate was wrong.

    The same defect was fixed in asr_openai_benchmark.py (a61fb3cdf) five days
    before this script was committed; it arrived carrying the pre-fix form.
    """
    resolve = _resolve_secs()
    # measured wins even when the server answers
    assert resolve({"usage": {"seconds": 12}}, 11.34) == 11.34
    assert resolve({"duration": 12.0}, 11.34) == 11.34
    # and the fallbacks keep their old precedence when we could not measure
    assert resolve({"duration": 12.0, "usage": {"seconds": 13}}, None) == 12.0
    assert resolve({"usage": {"seconds": 13}}, None) == 13.0
    assert resolve({}, None) is None


def test_the_measured_duration_is_actually_handed_to_the_resolver():
    """The fix is in the call site as much as in the function.

    resolve_secs could prefer its argument perfectly and still be useless while
    transcribe() passes None, which is exactly how this shipped.
    """
    src = _read(EVAL)
    body = src[src.index("def transcribe(") : src.index("# --- Japanese text normalization")]
    assert "resolve_secs(data,dur)" in body.replace(" ", ""), (
        "transcribe() must pass the duration wav_dur() already measured"
    )
    assert "resolve_secs(data,None)" not in body.replace(" ", ""), (
        "passing None discards the local measurement"
    )


def test_the_aggregate_uses_that_duration():
    src = _read(EVAL)
    lines = [ln.replace(" ", "") for ln in src.splitlines()]
    assert any("audio+=(asecord)" in ln for ln in lines), (
        "the corpus total must come from the per-clip duration"
    )


def test_both_harnesses_state_the_same_rule():
    """One rule, two clients: a divergence here is how this bug survived."""
    bench = _read(
        os.path.join(HERE, "..", "reference_config", "benchmarking", "asr_openai_benchmark.py")
    )
    eval_src = _read(EVAL)
    for src, name in ((bench, "asr_openai_benchmark.py"), (eval_src, "asr_ja_eval.py")):
        assert "billing quantity" in src, f"{name} must say why the server figure is not a measurement"
        assert "1649.4" in src and "1892.0" in src, f"{name} must cite the measured overstatement"
