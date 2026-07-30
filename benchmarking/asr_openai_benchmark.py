#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# vLLM OpenAI-compatible ASR benchmark client.
#
# Purpose
# -------
# The existing audio benchmark (test_module/benchmark_tests/audio_benchmark_tests.py)
# targets the tt-media-server whisper endpoint: it POSTs a base64 JSON body and
# reads a top-level ``duration`` key that only the media server returns. Models
# served through the *vLLM* OpenAI-compatible endpoint (e.g. Qwen3-ASR on the TT
# vLLM backend) instead speak the standard OpenAI transcription API: multipart
# form upload, and audio length reported as ``usage.seconds`` (json) or
# ``duration`` (verbose_json). This client benchmarks that standard path.
#
# It downloads N LibriSpeech samples from the HF Datasets Server (stdlib only,
# same source the media example uses), sends multipart POSTs to
# /v1/audio/transcriptions, and reports per-request latency plus aggregate
# RTF (elapsed / audio_seconds), RTR (audio_seconds / elapsed) and TTFT.
#
# Duration resolution order (OpenAI/vLLM standard first, media fallback):
#   1. verbose_json  -> top-level "duration"
#   2. json          -> "usage":{"type":"duration","seconds":N}
#   3. media server  -> top-level "duration"
#
# stdlib only; no extra dependencies.

import argparse
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

DATASET_REPO = "openslr/librispeech_asr"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
USER_AGENT = "tt-inference-server-asr-benchmark/1.0"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_data_asr")


def eprint(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def ensure_data_dir() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


# ----------------------------- HF dataset fetch -----------------------------
def _http_get(url: str, accept: str = "application/json", timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_rows_metadata(samples: int, config: str, split: str) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": DATASET_REPO, "config": config, "split": split, "offset": 0, "length": samples}
    )
    url = f"{DATASETS_SERVER}/rows?{query}"
    eprint(f"Fetching HF dataset metadata: {url}")
    return json.loads(_http_get(url).decode("utf-8"))


def fetch_audio_bytes(row: dict) -> Optional[bytes]:
    audio_list = row.get("row", {}).get("audio") or row.get("audio") or []
    if not audio_list:
        return None
    src = audio_list[0].get("src")
    if not src:
        return None
    try:
        return _http_get(src, accept="*/*")
    except Exception as e:  # noqa: BLE001
        eprint(f"audio fetch failed: {e}")
        return None


def download_samples(samples: int, config: str, split: str, concurrency: int) -> List[bytes]:
    meta = fetch_rows_metadata(samples, config, split)
    rows = meta.get("rows", [])
    total = min(len(rows), samples)
    out: List[Optional[bytes]] = [None] * total
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(fetch_audio_bytes, rows[i]): i for i in range(total)}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
    good = [b for b in out if b]
    eprint(f"Downloaded {len(good)}/{total} audio samples")
    return good


def wav_duration_seconds(data: bytes) -> Optional[float]:
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001
        return None


# ----------------------------- multipart POST -------------------------------
def _multipart_body(fields: dict, filename: str, file_bytes: bytes) -> Tuple[bytes, str]:
    boundary = f"----ttbench{uuid.uuid4().hex}"
    lines: List[bytes] = []
    for k, v in fields.items():
        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="{k}"'.encode())
        lines.append(b"")
        lines.append(str(v).encode())
    lines.append(f"--{boundary}".encode())
    lines.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode())
    lines.append(b"Content-Type: audio/wav")
    lines.append(b"")
    lines.append(file_bytes)
    lines.append(f"--{boundary}--".encode())
    lines.append(b"")
    return b"\r\n".join(lines), boundary


def resolve_audio_seconds(resp: dict, fallback: Optional[float]) -> Optional[float]:
    # verbose_json / media: top-level "duration"
    d = resp.get("duration")
    if d is not None:
        try:
            return float(d)
        except (TypeError, ValueError):
            pass
    # json: usage.seconds (OpenAI/vLLM standard)
    usage = resp.get("usage")
    if isinstance(usage, dict) and usage.get("seconds") is not None:
        try:
            return float(usage["seconds"])
        except (TypeError, ValueError):
            pass
    return fallback


def transcribe_once(
    host: str,
    file_bytes: bytes,
    model: str,
    language: Optional[str],
    response_format: str,
    authorization: str,
    timeout: int,
) -> Tuple[bool, float, Optional[float], str]:
    fields = {"model": model, "response_format": response_format}
    if language:
        fields["language"] = language
    body, boundary = _multipart_body(fields, "audio.wav", file_bytes)
    url = f"{host.rstrip('/')}/v1/audio/transcriptions"
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Authorization": f"Bearer {authorization}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
            ok = resp.status == 200
    except Exception as e:  # noqa: BLE001
        return False, time.perf_counter() - start, None, f"ERROR: {e}"
    elapsed = time.perf_counter() - start
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ok, elapsed, None, payload[:200]
    audio_s = resolve_audio_seconds(data, None)
    text = data.get("text", "")
    return ok, elapsed, audio_s, text


def run(args: argparse.Namespace) -> int:
    ensure_data_dir()
    samples = download_samples(args.samples, args.config, args.split, args.download_concurrency)
    if not samples:
        eprint("No audio samples downloaded; aborting.")
        return 1

    results: List[Tuple[bool, float, Optional[float], Optional[float]]] = []

    def one(idx: int) -> Tuple[bool, float, Optional[float], Optional[float]]:
        wav = samples[idx % len(samples)]
        wav_dur = wav_duration_seconds(wav)
        ok, elapsed, audio_s, text = transcribe_once(
            args.host, wav, args.model, args.language, args.response_format,
            args.authorization, args.timeout,
        )
        dur = audio_s if audio_s is not None else wav_dur
        if args.verbose:
            eprint(f"[req {idx}] ok={ok} elapsed={elapsed:.3f}s audio={dur}s text={text[:60]!r}")
        return ok, elapsed, dur, wav_dur

    n = args.num_requests
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = [ex.submit(one, i) for i in range(n)]
        for fut in as_completed(futs):
            results.append(fut.result())
    wall = time.perf_counter() - t0

    ok_results = [r for r in results if r[0]]
    n_ok = len(ok_results)
    n_fail = len(results) - n_ok
    if n_ok == 0:
        eprint("All requests failed.")
        return 1

    elapseds = [r[1] for r in ok_results]
    rtfs = [r[1] / r[2] for r in ok_results if r[2]]
    rtrs = [r[2] / r[1] for r in ok_results if r[2]]
    total_audio = sum(r[2] for r in ok_results if r[2])

    def avg(x: List[float]) -> float:
        return sum(x) / len(x) if x else 0.0

    def p(x: List[float], q: float) -> float:
        if not x:
            return 0.0
        s = sorted(x)
        return s[min(len(s) - 1, int(q * len(s)))]

    report = {
        "host": args.host,
        "model": args.model,
        "response_format": args.response_format,
        "num_requests": n,
        "num_ok": n_ok,
        "num_fail": n_fail,
        "concurrency": args.concurrency,
        "wall_time_s": round(wall, 3),
        "avg_latency_s": round(avg(elapseds), 3),
        "p50_latency_s": round(p(elapseds, 0.5), 3),
        "p99_latency_s": round(p(elapseds, 0.99), 3),
        "avg_ttft_s": round(avg(elapseds), 3),
        "avg_rtf": round(avg(rtfs), 4),
        "avg_rtr": round(avg(rtrs), 4),
        "total_audio_s": round(total_audio, 2),
        "throughput_audio_s_per_s": round(total_audio / wall, 3) if wall else 0.0,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        eprint(f"Wrote report to {args.output}")
    return 0


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="vLLM OpenAI-compatible ASR (/v1/audio/transcriptions) benchmark client."
    )
    ap.add_argument("--host", default="http://127.0.0.1:8100", help="Server base URL")
    ap.add_argument("--model", default="Qwen3-ASR-1.7B-JA", help="Served model name")
    ap.add_argument("--language", default=None, help="Optional language hint (e.g. ja, en)")
    ap.add_argument(
        "--response-format", default="json", choices=["json", "verbose_json", "text"],
        help="OpenAI response_format. json -> usage.seconds; verbose_json -> duration.",
    )
    ap.add_argument("--samples", type=int, default=8, help="Distinct LibriSpeech samples to download")
    ap.add_argument("--num-requests", type=int, default=16, help="Total requests to send")
    ap.add_argument("--concurrency", type=int, default=1, help="Concurrent in-flight requests")
    ap.add_argument("--download-concurrency", type=int, default=4)
    ap.add_argument("--config", default="clean")
    ap.add_argument("--split", default="test")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--authorization", default="your-secret-key")
    ap.add_argument("--output", default=None, help="Optional path to write JSON report")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
