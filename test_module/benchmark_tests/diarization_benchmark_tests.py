# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Benchmark runner for speaker-diarization models (pyannote community-1).

Diarization fits none of the other runners: it is not a prompt/token workload
(so the LLM path is meaningless), it consumes a recording rather than text (so
the TTS path does not apply), and it returns speaker turns rather than a
transcript (so the ASR runner would score the wrong thing).

The figure that matters is the real-time ratio -- audio seconds handled per
wall-clock second -- because the request carries a recording of a known length.
That is measured here over the pyannoteAI-shaped API the service exposes: stage
the recording in the request and poll the job -- ``POST /v1/diarize`` then
``GET /v1/jobs/{jobId}``. The job API is the only shape the official API has,
so it is the one worth measuring. The audio goes inline (a non-standard
extension of our server; pyannoteAI takes a url only) rather than through the
``media://`` staging flow, which would need an object store standing next to
the server; a benchmark that cannot run without one measures the deployment
rather than the model.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import aiohttp

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from report_module.schema import Block

from .._test_common import (
    MetricSpec,
    block_id,
    run_tiered_check,
)
from ..context import MediaContext, require_health
from ..test_status import DiarizationTestStatus

logger = logging.getLogger(__name__)

# Requests are serialized server-side (the pyannote pipeline is not
# thread-safe), so a cold first call on a long recording needs a wide timeout.
REQUEST_TIMEOUT_S = 600

# Enough samples to mean something without turning a benchmark into an hour:
# each call runs the whole pipeline over a 30 s recording.
DEFAULT_NUM_CALLS = 3


# Cheap enough not to distort a timing measurement, frequent enough that the
# poll interval is noise next to a pipeline run measured in seconds.
JOB_POLL_INTERVAL_S = 0.25


async def _await_job(session, ctx: MediaContext, job_id: str, headers: dict):
    """Poll a diarization job until it leaves the running states.

    Returns the job ``output`` on success, or None on failure -- including a
    job that never finished inside the request timeout, which is a failed
    sample and not a reason to hang the run.
    """
    deadline = time.monotonic() + REQUEST_TIMEOUT_S
    while time.monotonic() < deadline:
        async with session.get(
            f"{ctx.base_url}/v1/jobs/{job_id}", headers=headers
        ) as response:
            if response.status != 200:
                logger.error(
                    f"Job poll failed with status {response.status}: "
                    f"{await response.text()}"
                )
                return None
            job = await response.json()
        status = job.get("status")
        if status == "succeeded":
            return job.get("output") or {}
        if status in ("failed", "canceled"):
            logger.error(f"Diarization job {job_id} ended as {status}: {job}")
            return None
        await asyncio.sleep(JOB_POLL_INTERVAL_S)
    logger.error(f"Diarization job {job_id} did not finish in {REQUEST_TIMEOUT_S}s")
    return None


def sample_audio_path() -> str:
    """pyannote's bundled 30 s two-speaker sample.

    Built from the package path rather than by importing
    ``pyannote.audio.sample``, which decodes the file eagerly at import time
    and so needs a working torchcodec.
    """
    import pyannote.audio

    return str(Path(pyannote.audio.__file__).parent / "sample" / "sample.wav")


def audio_duration_seconds(path: str) -> Optional[float]:
    """Duration of a WAV file in seconds, or None if it cannot be read."""
    try:
        with wave.open(path, "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception as e:  # noqa: BLE001 - duration is optional metadata
        logger.warning(f"Could not read audio duration from {path}: {e}")
        return None


async def diarize_once(
    ctx: MediaContext, audio_path: str, audio_duration: Optional[float]
) -> DiarizationTestStatus:
    """Submit the audio as a diarization job and time the round trip."""
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
    }
    with open(audio_path, "rb") as handle:
        audio_bytes = handle.read()
    # Inline base64 rather than the media:// staging flow. Staging needs an
    # object store standing next to the server, and a benchmark that cannot run
    # without one measures the deployment rather than the model. The bytes
    # reach the pipeline identically either way; what is timed is the
    # diarization, not the transport.
    #
    # Note this is a non-standard extension of tt-media-server: the pyannoteAI
    # cloud API takes a url only. Fine here because this benchmark targets our
    # own server, but it is not a pattern to copy into client code.
    audio_url = base64.b64encode(audio_bytes).decode("ascii")

    start_time = time.monotonic()
    try:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            request_start = time.monotonic()
            async with session.post(
                f"{ctx.base_url}/v1/diarize",
                json={"url": audio_url, "exclusive": True},
                headers={**headers, "Content-Type": "application/json"},
            ) as response:
                if response.status != 201:
                    logger.error(
                        f"Diarize job creation failed with status {response.status}: "
                        f"{await response.text()}"
                    )
                    return DiarizationTestStatus(status=False, elapsed=0.0)
                job_id = (await response.json())["jobId"]

            # Poll the job. ttft is measured at the first response that
            # carries output, which is when the server actually produced it.
            body = await _await_job(session, ctx, job_id, headers)
            if body is None:
                return DiarizationTestStatus(status=False, elapsed=0.0)
            ttft_ms = (time.monotonic() - request_start) * 1000
    except Exception as e:  # noqa: BLE001 - a failed call is a failed sample
        logger.error(f"Diarization request failed: {type(e).__name__}: {e}")
        return DiarizationTestStatus(status=False, elapsed=0.0)

    elapsed = time.monotonic() - start_time

    turns = body.get("diarization") or []
    num_speakers = len({turn.get("speaker") for turn in turns})
    rtr = audio_duration / elapsed if audio_duration and elapsed > 0 else None

    rtr_str = f"{rtr:.2f}" if rtr is not None else "N/A"
    logger.info(
        f"✅ Done in {elapsed:.2f}s | ttft={ttft_ms:.1f}ms | RTR={rtr_str} | "
        f"speakers={num_speakers} turns={len(turns)}"
    )
    return DiarizationTestStatus(
        status=bool(turns),
        elapsed=elapsed,
        ttft_ms=ttft_ms,
        rtr=rtr,
        num_speakers=num_speakers,
        num_turns=len(turns),
        turns=turns,
    )


def _diarization_avg(
    status_list: list[DiarizationTestStatus], attr: str
) -> Optional[float]:
    valid = [getattr(s, attr) for s in status_list if getattr(s, attr) is not None]
    return sum(valid) / len(valid) if valid else None


def _throughput_rps(
    status_list: list[DiarizationTestStatus], wall_seconds: float
) -> Optional[float]:
    if not status_list or wall_seconds <= 0:
        return None
    return len(status_list) / wall_seconds


def run_diarization_benchmark(ctx: MediaContext) -> Block:
    """Run benchmarks for a speaker-diarization model."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} "
        f"on device: {ctx.device.name}"
    )
    require_health(ctx)

    audio_path = sample_audio_path()
    audio_duration = audio_duration_seconds(audio_path)

    try:
        bench_start = time.monotonic()
        status_list = [
            asyncio.run(diarize_once(ctx, audio_path, audio_duration))
            for _ in range(DEFAULT_NUM_CALLS)
        ]
        wall_seconds = time.monotonic() - bench_start
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    # A failed call yields no ttft/rtr, and averaging skips missing values, so a
    # run where most calls errored would otherwise report the healthy average of
    # the few that worked -- a benchmark that gets better the more it fails.
    failed = [status for status in status_list if not status.status]
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(status_list)} diarization calls failed; "
            "refusing to report an average over the survivors"
        )

    logger.info("Generating benchmark report...")
    ttft_ms_value = _diarization_avg(status_list, "ttft_ms")
    rtr_value = _diarization_avg(status_list, "rtr")
    throughput_rps = _throughput_rps(status_list, wall_seconds)

    target_checks, target_check = run_tiered_check(
        ctx,
        [
            MetricSpec(
                "TTFT",
                ttft_ms_value,
                "ttft_ms",
                lower_is_better=True,
                field_name="ttft",
            ),
            MetricSpec(
                "RTR", rtr_value, "rtr", lower_is_better=False, field_name="rtr"
            ),
        ],
    )

    return Block(
        kind="benchmarks",
        task_type="diarization",
        title="Speaker Diarization Benchmark",
        id=block_id(ctx) or None,
        targets={"num_prompts": len(status_list)},
        data={
            "Benchmarks": {
                "num_requests": len(status_list),
                "ttft": ttft_ms_value / 1000 if ttft_ms_value is not None else None,
                "rtr": rtr_value,
                "throughput_rps": throughput_rps,
                "num_speakers": _diarization_avg(status_list, "num_speakers"),
                "target_check": target_check,
                "target_checks": target_checks,
            },
        },
    )
