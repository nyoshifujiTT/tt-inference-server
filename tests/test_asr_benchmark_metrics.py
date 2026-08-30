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
BENCH = os.path.join(HERE, "..", "benchmarking", "asr_openai_benchmark.py")


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
