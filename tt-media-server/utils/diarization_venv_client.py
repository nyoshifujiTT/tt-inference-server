# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Talk to the community-1 diarization worker running in the audio venv.

Same shape as ``AudioVenvWorker`` in ``utils/audio_manager.py``: a long-lived
child process, a ready handshake, then one JSON request per line. It exists
because ``pyannote.audio`` is only installed in the audio venv -- see
``utils/diarize_community1.py`` for why -- so the service cannot simply import
the backend.

``is_available()`` lets the caller fall back to an in-process backend, which is
what happens outside the container (development checkouts have pyannote in the
one venv they use).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

AUDIO_VENV_PYTHON = os.getenv(
    "AUDIO_VENV_PYTHON", "/home/container_app_user/tt-metal/audio_venv/bin/python"
)

WORKER_SCRIPT = Path(__file__).parent / "diarize_community1.py"

READY_TIMEOUT_SECONDS = 120


class DiarizationVenvClient:
    """Run community-1 diarization in the audio venv, one request at a time."""

    def __init__(
        self,
        model_path: str,
        logger,
        python_executable: str = AUDIO_VENV_PYTHON,
        script_path: Path = WORKER_SCRIPT,
        popen_factory=None,
        env: Optional[Dict[str, str]] = None,
    ):
        self._model_path = model_path
        self._logger = logger
        self._python = python_executable
        self._script = script_path
        self._popen_factory = popen_factory or subprocess.Popen
        self._env = env
        self._proc = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        """True when the audio venv interpreter and the worker script exist."""
        return os.path.exists(self._python) and self._script.exists()

    def _ensure_started(self):
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        env = dict(os.environ if self._env is None else self._env)
        # The worker imports utils.diarization_backend from the server tree.
        server_root = str(self._script.parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(
            [p for p in (server_root, env.get("PYTHONPATH", "")) if p]
        )

        self._proc = self._popen_factory(
            [
                self._python,
                str(self._script),
                "--serve",
                "--model-path",
                self._model_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            env=env,
        )

        ready = self._proc.stdout.readline()
        if not ready or json.loads(ready).get("status") != "ready":
            raise RuntimeError("diarization worker did not signal ready")
        self._logger.info("Diarization worker ready in the audio venv")
        return self._proc

    def diarize(
        self,
        audio,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        exclusive: bool = True,
    ) -> Dict[str, Optional[List[Dict]]]:
        """Mirror ``DiarizationBackend.diarize`` across the process boundary."""
        import numpy as np

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        array = waveform.numpy() if hasattr(waveform, "numpy") else np.asarray(waveform)

        with self._lock:
            proc = self._ensure_started()
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
                path = handle.name
            try:
                np.save(path, array)
                request = {
                    "id": str(uuid.uuid4()),
                    "audio_path": path,
                    "sample_rate": sample_rate,
                    "num_speakers": num_speakers,
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                    "exclusive": exclusive,
                }
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        if not line:
            raise RuntimeError("diarization worker exited while handling a request")
        response = json.loads(line)
        if response.get("status") != "success":
            raise RuntimeError(response.get("error") or "diarization worker failed")
        return response["result"]

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            proc.kill()
