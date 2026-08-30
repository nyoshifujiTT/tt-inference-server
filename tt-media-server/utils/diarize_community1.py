#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""community-1 diarization worker, run inside the audio venv.

``pyannote.audio`` is installed in ``audio_venv`` and not in the venv the
server itself runs in, because pulling it into the main venv drags torch from
2.7 to 2.13 and breaks the vLLM install that shares it. The server therefore
cannot call pyannote in-process; it speaks to this script the same way
``utils/audio_manager.py`` speaks to ``utils/diarize.py`` -- a long-lived
child, one JSON request per line on stdin, one JSON response per line on
stdout, ready signal first.

The device offload lives on this side of the boundary too: the audio venv can
import ``ttnn`` and sees the tt-metal checkout that carries the ttnn port, so
the accelerator hook is built here rather than in the parent.

Protocol::

    ready:    {"status": "ready"}
    request:  {"id": ..., "audio_path": "/tmp/x.npy", "sample_rate": 16000,
               "num_speakers": null, "min_speakers": null, "max_speakers": null,
               "exclusive": true}
    response: {"id": ..., "status": "success"|"error",
               "result": {...} | null, "error": null | "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _build_nn_accelerator(logger_write):
    """Return the ttnn offload hook, or None to stay on CPU.

    Mirrors the parent's policy: offload when the catalog resolved a device,
    fall back to CPU (rather than failing the request) when the device cannot
    be opened, since a slower answer beats no answer.
    """
    device_id = os.environ.get("DIARIZATION_DEVICE_ID", "").strip()
    if not device_id.isdigit():
        return None, None

    try:
        import ttnn

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )
        from tt_port.tt_nn_accelerator import make_tt_accelerator

        device = ttnn.open_device(
            device_id=int(device_id),
            l1_small_size=int(os.environ.get("DIARIZATION_L1_SMALL", "32768")),
        )
        logger_write(f"diarize_community1: offloading onto device {device_id}\n")
        return make_tt_accelerator(device), device
    except Exception as exc:  # noqa: BLE001 - any device failure means CPU
        logger_write(f"diarize_community1: device unavailable ({exc}); using CPU\n")
        return None, None


def serve(model_path: str, stdin=None, stdout=None, stderr=None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    def log(message: str) -> None:
        stderr.write(message)
        stderr.flush()

    from utils.diarization_backend import DiarizationBackend

    accelerator, _device = _build_nn_accelerator(log)
    backend = DiarizationBackend(
        model_path=model_path, device="cpu", nn_accelerator=accelerator
    )

    stdout.write(json.dumps({"status": "ready"}) + "\n")
    stdout.flush()

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            stdout.write(
                json.dumps({"id": None, "status": "error", "error": str(exc)}) + "\n"
            )
            stdout.flush()
            continue

        request_id = request.get("id")
        try:
            import numpy as np
            import torch

            audio = np.load(request["audio_path"])
            waveform = torch.from_numpy(audio).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            result = backend.diarize(
                {
                    "waveform": waveform,
                    "sample_rate": int(request.get("sample_rate", 16000)),
                },
                num_speakers=request.get("num_speakers"),
                min_speakers=request.get("min_speakers"),
                max_speakers=request.get("max_speakers"),
                exclusive=bool(request.get("exclusive", True)),
            )
            response = {"id": request_id, "status": "success", "result": result}
        except Exception as exc:  # noqa: BLE001 - report, never kill the worker
            response = {"id": request_id, "status": "error", "error": str(exc)}

        stdout.write(json.dumps(response) + "\n")
        stdout.flush()

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args(argv)
    return serve(args.model_path)


if __name__ == "__main__":
    raise SystemExit(main())
