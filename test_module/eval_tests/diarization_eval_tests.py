# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Accuracy eval for speaker-diarization models, scored as a DER.

The diarization error rate is the standard metric for this task: the fraction
of speaking time attributed to the wrong speaker, plus missed speech and false
alarm.

Scores a real corpus when ``DIARIZATION_CORPUS_DIR`` points at one, so the
result can be held against the DER this model is published as scoring. Falls
back to the 30 s sample pyannote ships otherwise: that still measures the
deployed pipeline against a human annotation rather than against another run of
itself, but one clean two-speaker clip says nothing about overlap, speaker
count or noise, and its DER is not comparable to any published figure.

The scoring itself comes from tt-metal's
``models.demos.audio.pyannote_diarization.accuracy``, the same module its
on-device tests use, so the number reported for the served model and the number
the tt-metal suite asserts on are produced by identical code. That module and
pyannote live in the EVALS_DIARIZATION venv, reached through
``test_module._diarization_scoring``; this process only calls the server.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from report_module.schema import Block

from .. import _diarization_scoring as scoring
from .._test_common import ReportCheckTypes, block_id
from ..benchmark_tests.diarization_benchmark_tests import (
    audio_duration_seconds,
    diarize_once,
)
from ..context import MediaContext, require_health

logger = logging.getLogger(__name__)


def _eval_block(
    ctx: MediaContext,
    *,
    task_name: str,
    score: Optional[float],
    published_score: Optional[float],
    published_score_ref: str,
    accuracy_check: ReportCheckTypes,
    extra: Optional[dict] = None,
) -> Block:
    data = {
        "task_name": task_name,
        "score": score,
        "published_score": published_score,
        "published_score_ref": published_score_ref,
        "accuracy_check": accuracy_check,
    }
    if extra:
        data.update(extra)
    return Block(
        kind="evals",
        task_type="diarization",
        title="Speaker Diarization Eval",
        id=block_id(ctx) or None,
        targets={
            "task_name": task_name,
            "published_score": published_score,
            "published_score_ref": published_score_ref,
        },
        data=data,
    )


def _diarize(ctx: MediaContext, wav_path: str):
    status = asyncio.run(
        diarize_once(ctx, wav_path, audio_duration_seconds(wav_path))
    )
    if not status.status:
        raise RuntimeError(f"diarization request failed for {wav_path}")
    return status


def _eval_over_sample(ctx: MediaContext, plan: dict) -> Block:
    """Score the bundled sample against the annotation shipped beside it."""
    audio_path = plan["sample_audio"]
    status = _diarize(ctx, audio_path)
    scored = scoring.score({audio_path: status.turns})
    logger.info(
        f"DER={scored['der']:.5f} | speakers={scored['num_speakers']} "
        f"(reference {scored['reference_num_speakers']}) | RTR={status.rtr}"
    )

    # A DER can look acceptable while the pipeline splits or merges speakers,
    # so the speaker count is checked alongside it rather than folded in.
    if not scored["speaker_count_matches"]:
        check = ReportCheckTypes.FAIL
    elif scored["der"] <= plan["accuracy_der_max"]:
        check = ReportCheckTypes.PASS
    else:
        check = ReportCheckTypes.FAIL

    return _eval_block(
        ctx,
        task_name="pyannote_sample_der",
        score=scored["der"],
        published_score=plan["published_der"],
        published_score_ref=plan["published_der_ref"],
        accuracy_check=check,
        extra={
            "num_speakers": scored["num_speakers"],
            "reference_num_speakers": scored["reference_num_speakers"],
            "speaker_count_matches": scored["speaker_count_matches"],
            "rtr": status.rtr,
        },
    )


def _eval_over_corpus(
    ctx: MediaContext, plan: dict, corpus_name: str, limit: Optional[int]
) -> Block:
    """Score a whole corpus so the DER is comparable to the published one.

    Every recording goes through the served endpoint, one at a time, and the
    metric is accumulated across them rather than averaged per file -- the same
    way the published figures are computed, so a long recording weighs more
    than a short one.

    The split has to match the published one to be a like-for-like check.
    Scoring a development split against a test-split figure lands well under it
    -- dev is the easier half -- so a split with no published number of its own
    reports NA rather than a pass it did not earn.
    """
    turns = {wav: _diarize(ctx, wav).turns for wav in plan["corpus_audio"]}
    scored = scoring.score(turns, corpus_root=plan["corpus_root"], limit=limit)

    published = plan["published_corpus_der"]
    if published is None:
        check = ReportCheckTypes.NA
        logger.warning(
            f"no published DER for split {corpus_name!r}; reporting the "
            "measured value without a pass/fail verdict"
        )
    else:
        ceiling = published + plan["corpus_der_tolerance"]
        check = (
            ReportCheckTypes.PASS
            if scored["der"] <= ceiling
            else ReportCheckTypes.FAIL
        )

    logger.info(
        f"{corpus_name} DER={scored['der']:.5f} over "
        f"{scored['num_recordings']} recordings (published {published})"
    )

    return _eval_block(
        ctx,
        task_name=f"{corpus_name}_der",
        score=scored["der"],
        published_score=published,
        published_score_ref=plan["published_der_ref"],
        accuracy_check=check,
        extra={
            "num_recordings": scored["num_recordings"],
            "per_recording_der": scored["per_recording"],
        },
    )


def run_diarization_eval(ctx: MediaContext) -> Block:
    """Score the served diarization model with a diarization error rate."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} "
        f"on device: {ctx.device.name}"
    )
    require_health(ctx)

    corpus_name = os.environ.get("DIARIZATION_CORPUS_NAME", "voxconverse-test")
    limit = os.environ.get("DIARIZATION_CORPUS_LIMIT")
    limit = int(limit) if limit else None
    plan = scoring.plan(corpus_name, limit)

    started = time.monotonic()
    if plan["corpus_root"]:
        block = _eval_over_corpus(ctx, plan, corpus_name, limit)
    else:
        logger.info(
            "DIARIZATION_CORPUS_DIR is unset; scoring the bundled 30 s sample. "
            "Set it to a corpus to compare against the published DER."
        )
        block = _eval_over_sample(ctx, plan)
    logger.info(f"Diarization eval finished in {time.monotonic() - started:.1f}s")
    return block
