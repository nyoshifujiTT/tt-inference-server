# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Benchmark client for speaker-diarization models (pyannote community-1).

Diarization does not fit any of the existing media strategies: it is not a
prompt/token workload (so the LLM path is meaningless), it consumes a recording
rather than text (so the TTS path does not apply), and it returns speaker turns
rather than a transcript (so an ASR client would score the wrong thing).

The figure that matters is the real-time ratio -- audio seconds handled per
wall-clock second -- because the request carries a recording of a known length.
That is what this client measures, using the pyannoteAI-shaped API the service
exposes: stage the audio through ``POST /v1/media/input`` + ``PUT``, then
``POST /v1/audio/diarize``.

Accuracy is scored as a diarization error rate against the human annotation
that ships with the sample recording. The scoring itself is imported from
tt-metal's ``models.demos.audio.pyannote_diarization.accuracy``, the same
module its on-device tests use, so the number reported here and the number the
tt-metal suite asserts on are produced by identical code rather than by two
implementations that could drift apart.
"""

import asyncio
import json
import logging
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import aiohttp

from .base_strategy_interface import BaseMediaStrategy, PerfCheck
from .test_status import DiarizationTestStatus

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.utils import get_num_calls
from workflows.workflow_types import ReportCheckTypes

logger = logging.getLogger(__name__)

# Requests are serialized server-side (the pyannote pipeline is not
# thread-safe), so a cold first call plus a long recording needs a wide timeout.
REQUEST_TIMEOUT_S = 600

DIARIZATION_EVAL_TASK_NAME = "pyannote_sample_der"


def _accuracy():
    """The scoring helpers from tt-metal, imported lazily.

    The tt-metal checkout is already on PYTHONPATH wherever the ttnn port runs
    (``tt_port/tt_nn_accelerator`` imports from it the same way), but the
    benchmark half of this client must keep working without it, so the import
    is deferred to the eval path.
    """
    from models.demos.audio.pyannote_diarization import accuracy

    return accuracy


def _sample_audio_path() -> str:
    """Path to pyannote's bundled 30 s two-speaker sample.

    Kept here rather than taken from the tt-metal helper so the benchmark path
    needs no tt-metal checkout. Importing ``pyannote.audio.sample`` is avoided
    on purpose: it decodes the file eagerly at import time, which needs a
    working torchcodec.
    """
    import pyannote.audio

    return str(Path(pyannote.audio.__file__).parent / "sample" / "sample.wav")


def _audio_duration_seconds(path: str) -> Optional[float]:
    """Duration of a WAV file in seconds, or None if it cannot be read."""
    try:
        with wave.open(path, "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception as e:  # noqa: BLE001 - duration is optional metadata
        logger.warning(f"Could not read audio duration from {path}: {e}")
        return None


class DiarizationClientStrategy(BaseMediaStrategy):
    """Strategy for speaker-diarization models (pyannote community-1, etc.)."""

    def run_eval(self) -> None:
        """Score the served model with the diarization error rate.

        DER is the standard diarization metric: the fraction of speaking time
        attributed to the wrong speaker, plus missed speech and false alarm.
        The reference is the hand annotation pyannote ships beside its sample
        recording, so this measures the deployed pipeline against ground truth
        rather than against another run of itself.

        Lower is better, so the report inverts nothing: ``score`` is the DER and
        the accuracy check passes when it is at or below the configured target.
        """
        accuracy = _accuracy()

        logger.info(
            f"Running evals for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        self.require_health()

        audio_path = accuracy.sample_audio_path()
        audio_duration = _audio_duration_seconds(audio_path)
        reference = accuracy.load_rttm(accuracy.sample_reference_path())

        loop_start = time.monotonic()
        status = asyncio.run(self._diarize_once(audio_path, audio_duration))
        wall_clock_seconds = time.monotonic() - loop_start
        if not status.status:
            raise RuntimeError("diarization request failed; cannot score a DER")

        scored = accuracy.score_against_reference(
            accuracy.turns_to_annotation(status.turns), reference
        )
        der = scored["der"]
        reference_speakers = scored["reference_num_speakers"]
        speaker_count_matches = scored["speaker_count_matches"]

        logger.info(
            f"DER={der:.5f} | speakers={scored['num_speakers']} "
            f"(reference {reference_speakers}) | RTR={status.rtr}"
        )

        eval_data = [
            {
                "model": self.model_spec.model_name,
                "device": self.device.name.lower(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "task_type": "diarization",
                "task_name": DIARIZATION_EVAL_TASK_NAME,
                "score": der,
                "published_score": accuracy.PUBLISHED_DER,
                "published_score_ref": accuracy.PUBLISHED_DER_REF,
                "num_speakers": scored["num_speakers"],
                "reference_num_speakers": reference_speakers,
                "speaker_count_matches": speaker_count_matches,
                "rtr": status.rtr,
                "latency": status.latency,
                "throughput_rps": self._calculate_throughput_rps(
                    1, wall_clock_seconds
                ),
                "performance_check": self._calculate_performance_check(
                    status.latency, status.rtr
                ),
                "accuracy_check": self._calculate_accuracy_check(
                    der, speaker_count_matches
                ),
            }
        ]

        eval_filename = (
            Path(self.output_path)
            / f"eval_{self.model_spec.model_id}"
            / self.model_spec.hf_model_repo.replace("/", "__")
            / f"results_{time.time()}.json"
        )
        eval_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_filename, "w") as f:
            json.dump(eval_data, f, indent=4)
        logger.info(f"Evaluation data written to: {eval_filename}")

    def _calculate_accuracy_check(
        self, der: float, speaker_count_matches: bool
    ) -> ReportCheckTypes:
        """PASS when the DER is within target and the speaker count is right.

        A DER can look acceptable while the pipeline splits or merges speakers,
        so the speaker count is checked alongside it rather than folded in.
        The threshold is tt-metal's, so the served model and the on-device test
        are held to the same bar.
        """
        if not speaker_count_matches:
            return ReportCheckTypes.FAIL
        return (
            ReportCheckTypes.PASS
            if der <= _accuracy().ACCURACY_DER_MAX
            else ReportCheckTypes.FAIL
        )

    def run_benchmark(self) -> list[DiarizationTestStatus]:
        """Run benchmarks for the diarization model."""
        logger.info(
            f"Running benchmarks for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            self.require_health()
            num_calls = get_num_calls(self)
            loop_start = time.monotonic()
            status_list = self._run_diarization_benchmark(num_calls)
            wall_clock_seconds = time.monotonic() - loop_start
            self._generate_report(status_list, wall_clock_seconds)
            return status_list
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise

    def _run_diarization_benchmark(
        self, num_calls: int
    ) -> list[DiarizationTestStatus]:
        """Diarize the sample recording ``num_calls`` times."""
        logger.info(f"Running diarization benchmark with {num_calls} calls.")
        audio_path = _sample_audio_path()
        audio_duration = _audio_duration_seconds(audio_path)

        status_list = []
        for i in range(num_calls):
            logger.info(f"Diarizing {i + 1}/{num_calls}...")
            status_list.append(
                asyncio.run(self._diarize_once(audio_path, audio_duration))
            )
        return status_list

    async def _diarize_once(
        self, audio_path: str, audio_duration: Optional[float]
    ) -> DiarizationTestStatus:
        """Stage the audio, diarize it, and time the round trip."""
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
        }
        object_key = f"benchmark/{time.time()}.wav"

        with open(audio_path, "rb") as handle:
            audio_bytes = handle.read()

        start_time = time.monotonic()
        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # 1. declare the object, 2. upload the bytes to the returned url
                async with session.post(
                    f"{self.base_url}/v1/media/input",
                    json={"url": f"media://{object_key}"},
                    headers={**headers, "Content-Type": "application/json"},
                ) as response:
                    if response.status not in (200, 201):
                        logger.error(
                            f"Staging declaration failed with status {response.status}: "
                            f"{await response.text()}"
                        )
                        return DiarizationTestStatus(status=False, elapsed=0.0)
                    put_url = (await response.json())["url"]

                async with session.put(
                    put_url, data=audio_bytes, headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(
                            f"Audio upload failed with status {response.status}: "
                            f"{await response.text()}"
                        )
                        return DiarizationTestStatus(status=False, elapsed=0.0)

                # 3. diarize
                request_start = time.monotonic()
                async with session.post(
                    f"{self.base_url}/v1/audio/diarize",
                    json={"url": f"media://{object_key}", "exclusive": True},
                    headers={**headers, "Content-Type": "application/json"},
                ) as response:
                    if response.status != 200:
                        logger.error(
                            f"Diarize request failed with status {response.status}: "
                            f"{await response.text()}"
                        )
                        return DiarizationTestStatus(status=False, elapsed=0.0)
                    latency = time.monotonic() - request_start
                    body = await response.json()
        except Exception as e:  # noqa: BLE001 - a failed call is a failed sample
            logger.error(f"Diarization request failed: {type(e).__name__}: {e}")
            return DiarizationTestStatus(status=False, elapsed=0.0)

        elapsed = time.monotonic() - start_time

        turns = body.get("diarization") or []
        num_speakers = len({turn.get("speaker") for turn in turns})
        rtr = audio_duration / elapsed if audio_duration and elapsed > 0 else None

        rtr_str = f"{rtr:.2f}" if rtr is not None else "N/A"
        logger.info(
            f"✅ Done in {elapsed:.2f}s | latency={latency:.4f}s | RTR={rtr_str} | "
            f"speakers={num_speakers} turns={len(turns)}"
        )
        return DiarizationTestStatus(
            status=bool(turns),
            elapsed=elapsed,
            latency=latency,
            rtr=rtr,
            num_speakers=num_speakers,
            num_turns=len(turns),
            turns=turns,
        )

    def _generate_report(
        self,
        status_list: list[DiarizationTestStatus],
        wall_clock_seconds: Optional[float] = None,
    ) -> None:
        """Write the benchmark report for the diarization model."""
        logger.info("Generating benchmark report...")
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time.time()}.json"
        )
        result_filename.parent.mkdir(parents=True, exist_ok=True)

        latency_value = _mean_of(status_list, "latency")
        rtr_value = _mean_of(status_list, "rtr")
        tail = self._calculate_tail_latencies(
            [status.latency for status in status_list]
        )
        throughput_rps = self._calculate_throughput_rps(
            len(status_list), wall_clock_seconds
        )

        report_data = {
            "benchmarks": {
                "num_requests": len(status_list),
                "latency": latency_value,
                "rtr": rtr_value,
                "throughput_rps": throughput_rps,
                **tail,
            },
            "model": self.model_spec.model_name,
            "device": self.device.name.lower(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "diarization",
            "performance_check": self._calculate_performance_check(
                latency_value, rtr_value
            ),
        }

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report generated: {result_filename}")

    def _calculate_performance_check(
        self,
        latency_value: Optional[float] = None,
        rtr_value: Optional[float] = None,
    ) -> ReportCheckTypes:
        """Compare latency and RTR against the configured targets.

        The targets file stores latency in ms; convert here so the shared helper
        compares same-unit values (same boundary the TTS client uses).
        """
        targets = self.get_performance_targets()
        logger.info(f"Performance targets: {targets}")
        latency_target_s = (
            targets.ttft_ms / 1000.0 if targets.ttft_ms is not None else None
        )
        return self.calculate_performance_check(
            checks=[
                PerfCheck(
                    "latency", latency_value, latency_target_s, lower_is_better=True
                ),
                PerfCheck("RTR", rtr_value, targets.rtr, lower_is_better=False),
            ],
            tolerance=targets.tolerance,
        )


def _mean_of(status_list: list[DiarizationTestStatus], attr: str) -> float:
    """Mean of ``attr`` over the samples that reported it (0 when none did)."""
    values = [
        value
        for value in (getattr(status, attr) for status in status_list)
        if value is not None
    ]
    return sum(values) / len(values) if values else 0
