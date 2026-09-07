# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The TTFT / decode-TPS probes must ship with the runbook that quotes them.

E2E, prefill TTFT, decode TPS and TPS/user were reported for this bring-up, and
they do not come from asr_openai_benchmark.py -- that one reports corpus-level
rtfx and latency percentiles. They come from two probes that lived only in a
working directory on the bring-up host, so those figures were unreproducible
for anyone else, the same defect as the missing corpus eval.
"""

import ast
import os
import re

HERE = os.path.dirname(__file__)
BENCH_DIR = os.path.join(HERE, "..", "reference_config", "benchmarking")
PROBE = os.path.join(BENCH_DIR, "asr_perf_probe.py")
STREAM = os.path.join(BENCH_DIR, "asr_perf_stream.py")
README = os.path.join(HERE, "..", "scripts", "qwen3_asr", "README.md")


def _read(path):
    with open(path) as fh:
        return fh.read()


def test_both_probes_are_committed():
    for path in (PROBE, STREAM):
        assert os.path.exists(path), f"{os.path.basename(path)} must be in the repo"
        ast.parse(_read(path))


def test_the_non_streaming_probe_reads_the_server_counters():
    """stream=false is the customer setting, so timings come off /metrics.

    A client cannot see per-token timings on a non-streaming response; the
    numbers have to be differenced from vLLM's own counters. If this probe ever
    stopped doing that it would be silently reporting client-side wall time.
    """
    src = _read(PROBE)
    # the endpoint must actually be requested. A bare '"/metrics" in src' was
    # satisfied by the file's own header comment, so pointing the request at
    # another endpoint left this green.
    assert 'HOST+"/metrics"' in src.replace(" ", ""), (
        "the probe must fetch HOST + /metrics, not merely mention it"
    )
    # and it has to be sampled twice, or there is no delta to report
    assert src.count("=metrics()") >= 2, (
        "counters must be read before and after the workload"
    )
    for counter in (
        "time_to_first_token_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "generation_tokens_total",
    ):
        assert counter in src, f"{counter} must be sampled"


def test_every_counter_the_probe_scrapes_is_actually_stored():
    """A counter that is matched but not assigned is not sampled at all.

    metrics() used to run a loop over six counter names whose result was
    computed into a local and dropped, then re-parse everything below it. The
    loop looked like the sampling code and was not: had a later edit deleted
    the second parse instead, every reported number would have been 0.0 with no
    error. Require each counter name to appear on a line that writes into the
    returned dict.
    """
    src = _read(PROBE)
    body = src[src.index("def metrics():") : src.index("body=open(WAV")]
    stored = [ln for ln in body.splitlines() if 'd["' in ln]
    for counter in (
        "time_to_first_token_seconds",
        "e2e_request_latency_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "generation_tokens_total",
        "request_success_total",
    ):
        assert any(counter in ln for ln in stored), (
            f"{counter} is parsed but never stored in the returned dict"
        )
    # and nothing may be parsed into a variable that goes nowhere. The shared
    # helpers g()/gsum() legitimately match into a local they return, so the
    # check is on the counter patterns instead: a vllm: pattern outside the
    # helpers must be on a line that stores into the returned dict.
    helper = body[body.index("def g(pat") : body.index('d["ttft_sum"]')]
    outside = body.replace(helper, "")
    dropped = [
        ln.strip()
        for ln in outside.splitlines()
        if "vllm:" in ln and 'd["' not in ln and not ln.lstrip().startswith("#")
    ]
    assert not dropped, f"these matches are computed and discarded: {dropped}"


def test_the_streaming_probe_times_the_sse_stream():
    """First chunk = TTFT, inter-chunk gap = TPOT; it must actually stream."""
    src = _read(STREAM)
    assert '"stream":"true"' in src or '"stream": "true"' in src
    assert "data:" in src, "the SSE frames have to be parsed"
    assert "tpot" in src.lower()
    assert "decode_tps_per_user" in src


def test_the_streaming_probe_does_not_equate_a_frame_with_a_token():
    """An SSE frame is not a token, so frames may not be the throughput numerator.

    vLLM's stream generator (entrypoints/speech_to_text/base/serving.py) emits
    one frame per non-empty post-processed delta -- a delta can carry several
    tokens -- and appends a usage-only frame with ``choices=[]`` when
    ``stream_include_usage`` is set. Counting frames therefore both undercounts
    and overcounts, and the resulting "decode TPS" is not the quantity
    ``vllm:generation_tokens_total`` gives the non-streaming probe, so the two
    columns in the runbook were not comparable despite being printed side by
    side under the same names.
    """
    src = _read(STREAM)
    # the usage frame must be requested, and by the field the server accepts:
    # TranscriptionRequest declares a FLATTENED stream_include_usage (form data
    # cannot nest), and setting it without stream=true is rejected outright.
    assert '"stream_include_usage"' in src, (
        "ask for the usage frame; without it there is no token count at all"
    )
    assert '"stream_options"' not in src, (
        "the transcription endpoint takes the flattened stream_include_usage, "
        "not a nested stream_options object"
    )
    # tokens must come from that frame, not from a frame tally
    assert 'usage.get("completion_tokens")' in src, (
        "the token count must be read off the usage frame"
    )
    assert "nframe" in src, "frames must be counted separately from tokens"
    assert 'tot_tok=sum(r["nframe"]' not in src, "frames may not stand in for tokens"


def test_the_streaming_probe_skips_the_usage_frame_when_timing():
    """The usage frame carries no delta, so timing it inflates the frame count."""
    src = _read(STREAM)
    assert "if not choices:" in src, "a frame with choices=[] is not a content frame"
    assert 'if not choices[0].get("delta",{}).get("content"):' in src, (
        "only frames that actually carry text may enter the TPOT series"
    )


def test_the_streaming_probe_scales_tpot_to_tokens():
    """TPOT is a frame gap; per-user TPS is in tokens, so the two differ.

    1/TPOT is frames per second. Reporting it as decode_tps_per_user next to
    the non-streaming probe's token-based figure understates throughput by
    exactly the tokens-per-frame ratio.
    """
    src = _read(STREAM)
    assert "tok_per_frame" in src
    assert "round(tok_per_frame/mean_tpot,2)" in src, (
        "per-user TPS must be tokens per second, not frames per second"
    )
    assert "round(1.0/m(\"tpot_mean\"),2)" not in src, "the frame-rate form must be gone"


def test_the_streaming_probe_refuses_to_guess_when_usage_is_absent():
    """No usage frame means no token count; report that instead of frames."""
    src = _read(STREAM)
    assert "server sent no usage frame" in src
    assert "raise SystemExit(1)" in src, "a run with no token count must not look successful"


def test_both_probes_send_the_customer_request_parameters():
    """Same preset as the corpus eval, or the timings are not comparable."""
    for path in (PROBE, STREAM):
        src = _read(path)
        for field in ('"language"', '"to_language"', '"max_completion_tokens"'):
            assert field in src, f"{os.path.basename(path)} must send {field}"


def test_the_runbook_documents_both_probes():
    readme = _read(README)
    assert "reference_config/benchmarking/asr_perf_probe.py" in readme
    assert "reference_config/benchmarking/asr_perf_stream.py" in readme
    # the positional interface, or the commands cannot be adapted
    assert "<host> <model> <wav> <requests> <concurrency> <max_tokens>" in readme
    # and why there are two of them
    assert "stream=false" in readme


def test_the_runbook_does_not_explain_the_gap_with_the_disproved_reason():
    """"the final chunk carries no new token" was a guess, and it was wrong.

    The streaming column's 23.0 "tokens per request" was a frame tally, not a
    token count, so the one-token gap against the non-streaming 24.0 had no
    mechanism behind it -- the two quantities were simply different units that
    happened to land close together. Leaving that sentence in place would let a
    reader treat the columns as calibrated when they were not.
    """
    readme = _read(README)
    flat = " ".join(readme.split())
    # The phrase may survive only as a quotation of the retracted claim, which
    # is how the correction names what it is correcting. It must not stand as
    # the runbook's own explanation.
    assert flat.count("final chunk carries no new token") <= 1, (
        "the retracted explanation must appear at most once, as a quotation"
    )
    if "final chunk carries no new token" in flat:
        i = flat.index("final chunk carries no new token")
        assert "described a coincidence" in flat[i : i + 200], (
            "the phrase may only appear where it is being retracted"
        )


def test_the_two_columns_agree_on_tokens_per_request():
    """The one row that must match exactly, because it is the same quantity.

    Both columns now count tokens: /metrics differences
    vllm:generation_tokens_total, the streaming probe reads completion_tokens
    off the usage frame. Latencies may differ (framing lands on the client's
    clock) but the token count may not, so a mismatch here means one side has
    gone back to measuring something else -- the exact failure this row exists
    to catch.
    """
    readme = _read(README)
    section = readme[readme.index("| requests ok |") :]
    row = [ln for ln in section.splitlines() if ln.startswith("| tokens per request |")]
    assert row, "the runbook must report tokens per request for both probes"
    cells = [c.strip() for c in row[0].strip("|").split("|")]
    assert cells[1] == cells[2] != "", (
        f"the two probes must report the same tokens per request, got {cells[1:]}"
    )


def test_the_runbook_records_the_frame_count_separately():
    """Frames are still worth reporting -- as frames, next to the tokens."""
    readme = _read(README)
    section = readme[readme.index("| requests ok |") :]
    assert "| SSE frames per request |" in section, (
        "report frames as their own row so they cannot be read as tokens"
    )
    assert "A frame is not a token" in readme
    assert "24 tokens arrive in 22 frames" in readme, (
        "quote the measurement that shows the two are different quantities"
    )


def _defaults(src):
    """The literal each positional argument falls back to, keyed by variable."""
    out = {}
    for line in src.splitlines():
        line = line.strip()
        for name in ("HOST", "MODEL", "N", "C", "MAXTOK"):
            if line.startswith(f"{name}=") and "sys.argv" in line and "else" in line:
                out[name] = line.split("else", 1)[1].strip()
    return out


def test_the_two_probes_default_to_the_same_workload():
    """They are printed as two columns of one comparison, so they must agree.

    The streaming probe defaulted to port 8100 and served name
    "Qwen3-ASR-1.7B" while the non-streaming one used 8101 and
    "neosophie/Qwen3-ASR-1.7B-JA", and they defaulted to 40 vs 60 requests. Run
    without arguments the streaming probe therefore 404s on a name the server
    does not serve, and even with the name fixed it would have measured a
    different workload than the column beside it.
    """
    probe, stream = _defaults(_read(PROBE)), _defaults(_read(STREAM))
    for name in ("HOST", "MODEL", "N", "C", "MAXTOK"):
        assert probe.get(name) == stream.get(name), (
            f"{name} differs: probe {probe.get(name)!r} vs stream {stream.get(name)!r}"
        )


def test_the_default_port_is_the_one_the_runbook_uses():
    """Every command in the runbook targets 8110; a probe default must not differ."""
    for path in (PROBE, STREAM):
        assert _defaults(_read(path))["HOST"] == '"http://127.0.0.1:8110"', (
            f"{os.path.basename(path)}: default host must be the runbook's 8110"
        )


def test_the_default_model_is_the_served_name():
    served = '"neosophie/Qwen3-ASR-1.7B-JA"'
    for path in (PROBE, STREAM):
        assert _defaults(_read(path))["MODEL"] == served, (
            f"{os.path.basename(path)}: default model must be the served name"
        )


def test_neither_probe_defaults_to_a_scratch_directory_wav():
    """The runbook says to fetch the clip, not to copy one from a scratch dir.

    Both probes defaulted to /home/ubuntu/ttwork/real_ja.wav -- a path on the
    bring-up host only -- which is exactly what the runbook's "fetch it rather
    than copying a wav from someone's scratch directory" warns against. On any
    other machine the default is a FileNotFoundError; the clip has to be named.
    """
    for path in (PROBE, STREAM):
        src = _read(path)
        # comments may name the path they warn about; only code may not use it
        code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
        offenders = [ln for ln in code if "ttwork" in ln]
        assert not offenders, f"{os.path.basename(path)}: no scratch-dir default: {offenders}"
        assert "WAV=sys.argv[3]\n" in src, "the clip must be a required argument"
        assert "the clip is required" in src, "and saying so must be the failure mode"


def test_the_readme_separates_a_download_failure_from_a_model_failure():
    """The benchmark downloads before it measures, so it can fail with no result.

    Observed: "Fetching HF dataset metadata: ... TimeoutError: The read
    operation timed out" -- a traceback and no bench.json, with not one request
    having reached the server. Without this noted, a rerun looks like flaky
    inference rather than a flaky fetch.
    """
    readme = _read(README)
    body = readme[readme.index("Throughput (LibriSpeech") :]
    assert "datasets-server.huggingface.co" in body
    assert "before\nit touches the server" in body or "before it touches the server" in body
    assert "Rerun it." in body


def test_the_readme_covers_the_http_form_of_that_failure_too():
    """It arrives as a status as well as a timeout, and only one was recorded.

    Measured on this host: `urllib.error.HTTPError: HTTP Error 502: Bad Gateway`
    from the same datasets-server URL. A reader who has only the TimeoutError
    example in front of them cannot tell whether a 502 is the same benign case.
    """
    readme = _read(README)
    body = readme[readme.index("Throughput (LibriSpeech") :]
    assert "HTTP Error 502" in body, "record the status form of the fetch failure"
    assert "openslr" in body, "and that it comes from the LibriSpeech fetch"


def test_the_readme_gives_the_counter_delta_that_proves_it_was_the_download():
    """"no request reached the server" is checkable, so make it checked.

    A full pass adds 1349 to vllm:request_success_total; a pass whose benchmark
    never downloaded adds 1221, exactly the 128 benchmark requests fewer.
    Measured on the 502 above: 1221, with error/abort at 0.0.
    """
    readme = _read(README)
    body = readme[readme.index("Throughput (LibriSpeech") :]
    flat = " ".join(body.split())

    got = re.search(
        r"`1 \+ 494 \+ 600 \+ 128 \+ 63 \+ 63 = (\d+)`.*?"
        r"`1 \+ 494 \+ 600 \+ 63 \+ 63 = (\d+)`, exactly (\d+) fewer",
        flat,
    )
    assert got, "give both sums and the difference between them"
    full, without_bench, fewer = (int(g) for g in got.groups())
    # Evaluate the terms as written. Comparing the totals to restated constants
    # let "= 1400" through for the same terms, because the regex then simply
    # captured 1400 and every check agreed with itself.
    for label, total, terms in (
        ("full pass", full, "1 + 494 + 600 + 128 + 63 + 63"),
        ("no benchmark", without_bench, "1 + 494 + 600 + 63 + 63"),
    ):
        assert f"`{terms} = {total}`" in flat, f"the {label} sum must be written out"
        computed = sum(int(t) for t in terms.split("+"))
        assert computed == total, (
            f"the {label} sum does not add up: {terms} = {computed}, "
            f"but the runbook says {total}"
        )
    assert full - without_bench == fewer == 128, (
        f"the difference must be the 128 benchmark requests: "
        f"{full} - {without_bench} != {fewer}"
    )
    assert "Measured on the 502 above: **1221**" in flat, (
        "quote the delta actually observed, not only the prediction"
    )


def test_the_benchmark_really_downloads_before_measuring():
    """The advice above is only sound while the order is fetch-then-measure."""
    src = _read(os.path.join(BENCH_DIR, "asr_openai_benchmark.py"))
    assert "datasets-server.huggingface.co" in src
    download_at = src.index("def download_samples")
    # the request loop must come after the downloader in the call flow
    call_at = src.index("samples = download_samples(")
    post_at = src.rindex("/v1/audio/transcriptions")
    assert download_at < post_at and call_at < post_at, (
        "clips are fetched before any request is posted; if that changes, the "
        "README's 'no request reached the server' claim stops holding"
    )


def test_the_runbook_says_the_probes_need_an_idle_server():
    """The /metrics probe cannot exclude other traffic, and nothing said so.

    The counters it differences are server-wide, so concurrent requests land in
    its result, and everything shares one device so they slow it down too.
    Done by accident during verification here: a corpus eval overlapped the
    benchmark and probe window, and the pass reported rtfx 4.476 (normal
    11.6-13.0), gen_tokens 1689 (fixed at 1440) and TPS/user 10.79 (~25).
    Nothing had regressed; the measurement was meaningless, and without this
    note the numbers read as a serious performance loss.
    """
    readme = _read(README)
    body = readme[readme.index("The non-streaming probe exists because") :]
    body = body[: body.index("Measured on the delivery p150")]
    flat = " ".join(body.split())

    assert "Run these alone." in flat, "state the precondition"
    assert "cannot be attributed to a client" in flat, (
        "say why concurrent traffic corrupts the /metrics difference"
    )
    # the measured evidence, so the shape is recognisable
    assert "4.476" in flat and "1689" in flat, (
        "quote what a contaminated pass looked like"
    )
    # and the self-check that detects it
    assert "gen_tokens" in flat and "requests x tokens-per-request" in flat, (
        "give the invariant that reveals foreign traffic in the counters"
    )


def test_the_probe_really_sends_a_fixed_token_budget():
    """The gen_tokens self-check only works while the budget is fixed."""
    src = _read(PROBE)
    assert '"max_completion_tokens":MAXTOK' in src.replace(" ", ""), (
        "the probe must cap tokens per request, or gen_tokens cannot be "
        "predicted and the contamination check above is unusable"
    )
    assert "MAXTOK=int(sys.argv[6])" in src.replace(" ", ""), (
        "and the cap must be an argument, so the expected product is known"
    )


def test_the_metrics_reader_refuses_a_counter_it_cannot_find():
    """A missing counter must abort, not become a zero.

    Every figure the non-streaming probe prints is m1 - m0 of a /metrics
    counter. The reader returned 0.0 on a regex miss, so a renamed or
    relabelled vLLM counter made both reads 0.0 and the difference a
    plausible zero: gen_tokens 0, decode_tps_aggregate 0.0, req_per_s 0.0,
    all printed as if measured. The runbook's "gen_tokens must be 1440"
    self-check only catches that if a human reads the line.
    """
    src = _read(PROBE)

    assert "return float(m.group(1)) if m else 0.0" not in src, (
        "the silent-zero fallback is back; a lost counter would be reported "
        "as a measurement"
    )
    assert re.search(r"if not m:\s*\n\s*sys\.exit\(", src), (
        "a regex miss must exit; see the counters this probe differences"
    )


def test_every_counter_the_probe_differences_is_named_in_its_error():
    """The abort has to say which counter went missing.

    Seven counters go through one helper; "a counter is missing" would send
    the reader back to the source to work out which. Passing the metric name
    to the helper is what makes the message actionable.
    """
    src = _read(PROBE)

    counters = re.findall(r"=gs?u?m?\(r'(vllm:[a-z0-9_]+)", src)
    assert len(counters) >= 7, f"expected the seven differenced counters, got {counters}"

    for counter in counters:
        assert f'"{counter}"' in src, (
            f"{counter} is read but its name is not passed to the reader, so "
            f"the abort cannot say which one is missing"
        )


def test_the_success_counter_is_guarded_like_the_others():
    """findall returns [] and sum([]) is 0 -- the same substitution.

    request_success_total is read with findall rather than search, so the
    guard above does not cover it; without its own check it would reintroduce
    exactly the behaviour that check exists to forbid.
    """
    src = _read(PROBE)
    assert "vllm:request_success_total" in src
    assert re.search(r"def gsum\(pat, name\):", src), (
        "the findall-based read needs its own guard helper"
    )
    assert re.search(r"if not vals:\s*\n\s*sys\.exit\(", src), (
        "the success counter must abort when absent, like the differenced ones"
    )
    assert "gsum(r'vllm:request_success_total" in src, (
        "request_success_total must go through the guarded helper"
    )
