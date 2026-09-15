# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The ASR benchmark must report the standard aggregate speed metrics.

RTFx = audio_duration / processing_time and RTF = its reciprocal, both over the
ORIGINAL waveform duration. vLLM's own ASR benchmark computes
``rtfx = input_audio_duration / duration`` and the Open ASR Leaderboard reports
RTFx, so emitting the same names lets these numbers be compared with published
ones - and with the tt-metal side eval, which now reports the same pair.

The pre-existing avg_rtf / avg_rtr are per-request averages and answer a
different question: they ignore concurrency, so at concurrency > 1 rtfx rises
while avg_rtf does not move. Both are kept; they must not be conflated.
"""

import os

HERE = os.path.dirname(__file__)
# upstream moved benchmarking/ under reference_config/ (the directory rename
# merged in from upstream/main); this harness moved with it.
BENCH = os.path.join(
    HERE, "..", "reference_config", "benchmarking", "asr_openai_benchmark.py"
)


def _read(path):
    with open(path) as fh:
        return fh.read()


def test_report_exposes_the_standard_pair():
    src = _read(BENCH)
    assert '"rtfx": round(total_audio / wall, 3)' in src
    assert '"rtf": round(wall / total_audio, 4)' in src


def test_per_request_averages_are_kept_and_distinguished():
    # Removing them would lose the per-request view; conflating them with rtfx
    # would misreport concurrent runs.
    src = _read(BENCH)
    assert '"avg_rtf"' in src and '"avg_rtr"' in src
    assert "ignore concurrency" in src, "the difference must be documented in place"


def test_duration_is_the_submitted_audio_not_a_padded_length():
    # total_audio accumulates the per-request audio duration reported by the
    # client, i.e. the real clip length.
    src = _read(BENCH)
    assert "total_audio = sum(r[2] for r in ok_results if r[2])" in src


def test_audio_duration_prefers_our_own_measurement():
    # The server's usage.seconds is a BILLING quantity in whole seconds, so it
    # rounds every clip up. Trusting it inflated TED-509 from the true 1649.4 s
    # to 1892.0 s (+14.7%) and made rtfx look correspondingly better, while the
    # per-request durations recorded alongside stayed correct - so the error was
    # only visible in the aggregate.
    src = _read(BENCH)
    assert "def resolve_audio_seconds(resp: dict, measured: Optional[float])" in src
    assert "if measured is not None:\n        return measured" in src
    assert "resolve_audio_seconds(data, wav_duration_seconds(file_bytes))" in src


def test_server_reported_duration_is_still_the_fallback():
    # A non-WAV container cannot be measured locally; fall back rather than drop
    # the request from the aggregate.
    src = _read(BENCH)
    body = src[src.index("def resolve_audio_seconds") : src.index("def transcribe_once")]
    assert 'resp.get("duration")' in body
    assert 'usage.get("seconds")' in body


def _load(path, name):
    """Import a harness so its resolver can be run, not only grepped."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = os.path.join(HERE, "..", "reference_config", "evals", "asr_ja_eval.py")


def _resolvers():
    """(name, callable) for the duration resolver in each harness."""
    return [
        ("asr_openai_benchmark.py", _load(BENCH, "_bench_dur").resolve_audio_seconds),
        ("asr_ja_eval.py", _load(EVAL, "_eval_dur").resolve_secs),
    ]


def test_the_measured_duration_wins_over_the_servers():
    """The rule this function exists for, checked by running it.

    Both harnesses were only grepped for the source text of the guard. The
    bug it fixes was worth +14.7% on the TED aggregate, and a rewrite that
    kept the same first two lines while reordering what follows would still
    have matched every assertion above.
    """
    server_says = {"duration": 99.0, "usage": {"type": "duration", "seconds": 99}}

    for name, resolve in _resolvers():
        assert resolve(server_says, 3.24) == 3.24, (
            f"{name}: the locally measured duration must win"
        )
        # zero is a measurement, not an absence
        assert resolve(server_says, 0.0) == 0.0, (
            f"{name}: 0.0 is a measured value; `if measured is not None` must "
            f"not become a truthiness test"
        )


def test_the_server_figure_is_used_only_when_nothing_was_measured():
    """The fallback keeps an unmeasurable clip in the aggregate."""
    for name, resolve in _resolvers():
        assert resolve({"duration": 4.5}, None) == 4.5, name
        assert resolve({"usage": {"seconds": 7}}, None) == 7.0, name
        # duration is preferred over usage.seconds: it is not rounded
        assert resolve({"duration": 4.5, "usage": {"seconds": 5}}, None) == 4.5, name


def test_an_unusable_server_figure_does_not_become_a_number():
    """Returning None drops the clip from the speed aggregate; 0 would corrupt it."""
    for name, resolve in _resolvers():
        assert resolve({}, None) is None, name
        assert resolve({"duration": "not-a-number"}, None) is None, name
        assert resolve({"usage": "not-a-dict"}, None) is None, name
        assert resolve({"usage": {"seconds": None}}, None) is None, name


def test_both_harnesses_resolve_identically():
    """They report the same quantity under two names, so they must agree.

    The runbook holds their outputs side by side (throughput_audio_per_s and
    rtfx are the same measurement), which only works if the denominator is
    computed the same way on both sides.
    """
    cases = [
        ({"duration": 9.0, "usage": {"seconds": 9}}, 3.24),
        ({"duration": 9.0}, None),
        ({"usage": {"seconds": 7}}, None),
        ({}, None),
        ({"duration": "x"}, None),
        ({"duration": 4.5, "usage": {"seconds": 5}}, None),
    ]
    (_, bench), (_, eval_) = _resolvers()
    for resp, measured in cases:
        assert bench(resp, measured) == eval_(resp, measured), (
            f"the harnesses disagree on {resp!r} / {measured!r}: "
            f"{bench(resp, measured)!r} vs {eval_(resp, measured)!r}"
        )


def test_the_wav_duration_is_frames_over_rate_and_none_on_failure():
    """The measurement the guard prefers, and its failure mode.

    If this returned 0.0 instead of None for an unparseable container, the
    resolver would take it as a measurement and score that clip against a zero
    denominator.
    """
    import io
    import wave

    module = _load(BENCH, "_bench_wav")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 24000)  # 1.5 s at 16 kHz
    assert module.wav_duration_seconds(buf.getvalue()) == 1.5

    assert module.wav_duration_seconds(b"not a wav at all") is None
