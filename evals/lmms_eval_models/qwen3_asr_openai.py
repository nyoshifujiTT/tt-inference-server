# SPDX-License-Identifier: Apache-2.0
"""
lmms-eval model for ASR models served on an OpenAI-compatible HTTP server
(vLLM /v1/audio/transcriptions), e.g. Qwen3-ASR on the TT vLLM backend.

Unlike `whisper_tt` (which posts a base64 JSON body to the tt-media-server
`/audio/transcriptions` route), this model speaks the OpenAI standard:
multipart/form-data upload to `{base_url}/audio/transcriptions` where base_url
already includes `/v1`. It is engine-agnostic and reads the transcript from the
standard response (`text`).
"""
import asyncio
import io
import os
import time
import wave
from typing import List, Tuple

import aiohttp
import numpy as np
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import eval_logger

SAMPLING_RATE = 16000


def _downsample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


@register_model("qwen3_asr_openai")
class Qwen3ASROpenAI(lmms):
    """ASR over an OpenAI-compatible /v1/audio/transcriptions HTTP endpoint."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-ASR-1.7B",
        base_url: str = None,
        model: str = None,
        language: str = None,
        response_format: str = "json",
        batch_size: int = 1,
        timeout: int = 600,
        num_concurrent: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning(f"Ignoring unexpected kwargs: {kwargs}")
        # base_url already includes /v1 (run_evals sets OPENAI_API_BASE for audio)
        self.base_url = base_url or os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        self.model = model or pretrained
        self.language = language
        self.response_format = response_format
        self.timeout = timeout
        self.num_concurrent = int(num_concurrent)
        self.api_key = os.getenv("OPENAI_API_KEY", "your-secret-key")
        self._batch_size = int(batch_size)
        self._rank = 0
        self._world_size = 1
        # tokenizer only used by lm-eval collator; a simple whitespace splitter suffices
        eval_logger.info(f"Qwen3ASROpenAI base_url={self.base_url} model={self.model}")

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _wav_bytes(self, audio: np.ndarray, sr: int) -> bytes:
        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm16.tobytes())
        return buf.getvalue()

    async def _transcribe(self, session, audio: np.ndarray, sr: int, idx: int) -> str:
        wav = self._wav_bytes(audio, sr)
        form = aiohttp.FormData()
        form.add_field("file", wav, filename=f"audio_{idx}.wav", content_type="audio/wav")
        form.add_field("model", self.model)
        form.add_field("response_format", self.response_format)
        if self.language:
            form.add_field("language", self.language)
        url = f"{self.base_url}/audio/transcriptions"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            async with session.post(
                url, data=form, headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    eval_logger.warning(f"[{idx}] HTTP {resp.status}")
                    return ""
                result = await resp.json()
                if isinstance(result, dict):
                    return result.get("text") or result.get("transcription") or ""
                return str(result)
        except Exception as e:  # noqa: BLE001
            eval_logger.warning(f"[{idx}] request failed: {e}")
            return ""

    def loglikelihood(self, requests):
        raise NotImplementedError("loglikelihood not supported for ASR")

    def flatten(self, x):
        return [j for i in x for j in i]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        def _collate(x):
            return -len(x[0]), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        all_audios = []
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched = [doc_to_visual[0](self.task_dict[task][split][i]) for i in doc_id]
            for audio in self.flatten(batched):
                all_audios.append(
                    _downsample(audio["array"], audio["sampling_rate"], SAMPLING_RATE)
                )

        sem = asyncio.Semaphore(self.num_concurrent)

        async def run():
            async with aiohttp.ClientSession() as session:
                async def one(i, a):
                    async with sem:
                        return await self._transcribe(session, a, SAMPLING_RATE, i)
                return await asyncio.gather(*[one(i, a) for i, a in enumerate(all_audios)])

        t0 = time.time()
        answers = asyncio.run(run())
        eval_logger.info(f"Transcribed {len(all_audios)} clips in {time.time()-t0:.1f}s")

        pbar = tqdm(total=len(answers), disable=(self.rank != 0), desc="Model Responding")
        res = [a if a is not None else "" for a in answers]
        for _ in res:
            pbar.update(1)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("multi-round generation is not supported for ASR")
