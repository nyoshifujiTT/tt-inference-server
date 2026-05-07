# SPDX-License-Identifier: Apache-2.0
"""Runtime monkey patches loaded at Python startup via sitecustomize.

Used for bring-up compatibility in docker runs where we cannot rebuild vLLM image yet.
"""

import logging
import os

logger = logging.getLogger("sitecustomize")


def _patch_bge_tt_platform_arch_prefix() -> None:
    model = (os.getenv("MODEL") or "").lower()
    if model not in {"bge-m3", "baai/bge-m3"}:
        return

    try:
        from vllm.platforms.tt import TTPlatform
    except Exception as e:  # pragma: no cover - best effort patch
        logger.debug("sitecustomize: TTPlatform import skipped: %s", e)
        return

    original = TTPlatform.check_and_update_config

    def _normalize_arches(vllm_config):
        arch_names = vllm_config.model_config.hf_config.architectures or []
        for i, arch in enumerate(list(arch_names)):
            if arch == "TTXLMRobertaModel":
                arch_names[i] = "XLMRobertaModel"
            elif arch == "TTRobertaModel":
                arch_names[i] = "RobertaModel"
        return arch_names

    def _patched(vllm_config):
        try:
            original(vllm_config)
        except ValueError as e:
            msg = str(e)
            if "TTXLMRobertaModel" not in msg and "TTRobertaModel" not in msg:
                raise
            arch_names = _normalize_arches(vllm_config)
            logger.warning(
                "sitecustomize: intercepted TT-prefixed arch resolution error, "
                "normalized for bge-m3 -> %s", arch_names
            )
            return

        arch_names = _normalize_arches(vllm_config)
        logger.warning(
            "sitecustomize: normalized TT arch names for bge-m3 -> %s", arch_names
        )

    TTPlatform.check_and_update_config = classmethod(
        lambda cls, vllm_config: _patched(vllm_config)
    )
    logger.warning("sitecustomize: installed bge-m3 TTPlatform prefix patch")


_patch_bge_tt_platform_arch_prefix()
