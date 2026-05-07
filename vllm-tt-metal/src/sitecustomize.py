# SPDX-License-Identifier: Apache-2.0
"""Runtime monkey patches loaded at Python startup via sitecustomize.

Used for bring-up compatibility in docker runs where we cannot rebuild vLLM image yet.
"""

import logging
import os
import sys

logger = logging.getLogger("sitecustomize")


def _is_bge_context() -> bool:
    model = (os.getenv("MODEL") or "").lower()
    hf_model = (os.getenv("HF_MODEL") or "").lower()
    argv = " ".join(sys.argv).lower()
    haystack = f"{model} {hf_model} {argv}"
    return "bge-m3" in haystack or "baai/bge-m3" in haystack



def _patch_bge_tt_platform_arch_prefix() -> None:
    if not _is_bge_context():
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
        arch_names = _normalize_arches(vllm_config)
        logger.warning(
            "sitecustomize: normalized TT arch names for bge-m3 -> %s", arch_names
        )
        return original(vllm_config)

    TTPlatform.check_and_update_config = classmethod(
        lambda cls, vllm_config: _patched(vllm_config)
    )
    logger.warning("sitecustomize: installed bge-m3 TTPlatform prefix patch")


_patch_bge_tt_platform_arch_prefix()


def _patch_bge_roberta_embedding_shim() -> None:
    if not _is_bge_context():
        return

    try:
        from vllm.model_executor.models.roberta import RobertaEmbeddingModel
    except Exception as e:  # pragma: no cover - best effort patch
        logger.debug("sitecustomize: RobertaEmbeddingModel import skipped: %s", e)
        return

    if hasattr(RobertaEmbeddingModel, "initialize_vllm_model"):
        logger.warning("sitecustomize: RobertaEmbeddingModel already has initialize_vllm_model")
        return

    @classmethod
    def _initialize_vllm_model(cls, *args, **kwargs):
        from models.demos.wormhole.bge_m3.demo.generator_vllm import BgeM3ForEmbedding

        logger.warning(
            "sitecustomize: redirecting RobertaEmbeddingModel.initialize_vllm_model to BgeM3ForEmbedding"
        )
        return BgeM3ForEmbedding.initialize_vllm_model(*args, **kwargs)

    RobertaEmbeddingModel.initialize_vllm_model = _initialize_vllm_model
    logger.warning("sitecustomize: installed RobertaEmbeddingModel shim for bge-m3")


_patch_bge_roberta_embedding_shim()



def _patch_tt_compat_sampling_guard() -> None:
    if not _is_bge_context():
        return

    try:
        from vllm.platforms.tt import TTPlatform
    except Exception as e:  # pragma: no cover - best effort patch
        logger.debug("sitecustomize: compat_sampling guard skipped: %s", e)
        return

    original = TTPlatform.compat_sampling_required

    @staticmethod
    def _patched(sampling_params):
        if sampling_params is None:
            return False
        return original(sampling_params)

    TTPlatform.compat_sampling_required = _patched
    logger.warning("sitecustomize: installed compat_sampling_required None guard for bge-m3")


_patch_tt_compat_sampling_guard()



def _patch_tt_model_runner_sampling_none_guard() -> None:
    if not _is_bge_context():
        return

    try:
        from vllm.worker.tt_model_runner import TTModelRunner
        from vllm.sampling_params import SamplingParams
    except Exception as e:  # pragma: no cover - best effort patch
        logger.debug("sitecustomize: TTModelRunner sampling guard skipped: %s", e)
        return

    original = TTModelRunner.prepare_model_input

    def _patched(self, seq_group_metadata_list, *args, **kwargs):
        for seq_group_metadata in seq_group_metadata_list:
            if getattr(seq_group_metadata, "sampling_params", None) is None:
                seq_group_metadata.sampling_params = SamplingParams()

            # Embedding requests may carry None block tables; normalize to empty lists.
            bt = getattr(seq_group_metadata, "block_tables", None)
            if isinstance(bt, dict):
                for k, v in list(bt.items()):
                    if v is None:
                        bt[k] = []

            if getattr(seq_group_metadata, "cross_block_table", None) is None:
                seq_group_metadata.cross_block_table = []

        return original(self, seq_group_metadata_list, *args, **kwargs)

    TTModelRunner.prepare_model_input = _patched
    logger.warning("sitecustomize: installed TTModelRunner sampling_params None guard for bge-m3")


_patch_tt_model_runner_sampling_none_guard()



def _patch_tt_worker_kv_cache_guard() -> None:
    if not _is_bge_context():
        return

    try:
        from vllm.worker.tt_worker import TTWorker
    except Exception as e:  # pragma: no cover - best effort patch
        logger.debug("sitecustomize: TTWorker kv_cache guard skipped: %s", e)
        return

    def _safe_kv_cache(self):
        return getattr(self, "tt_cache", None)

    TTWorker.kv_cache = property(_safe_kv_cache)
    logger.warning("sitecustomize: installed TTWorker.kv_cache guard for bge-m3")


_patch_tt_worker_kv_cache_guard()



def _patch_ttplatform_sample_on_device_mode_attr() -> None:
    if not _is_bge_context():
        return

    try:
        from vllm.platforms.tt import TTPlatform
    except Exception as e:  # pragma: no cover
        logger.debug("sitecustomize: TTPlatform attr guard skipped: %s", e)
        return

    if not hasattr(TTPlatform, "sample_on_device_mode"):
        TTPlatform.sample_on_device_mode = None
        logger.warning("sitecustomize: added TTPlatform.sample_on_device_mode=None for bge-m3")


_patch_ttplatform_sample_on_device_mode_attr()


def _patch_bge_model_registry_registration() -> None:
    if not _is_bge_context():
        return

    try:
        from vllm import ModelRegistry
    except Exception as e:  # pragma: no cover
        logger.debug("sitecustomize: ModelRegistry import skipped: %s", e)
        return

    for arch_name in ["TTXLMRobertaModel", "XLMRobertaModel", "RobertaModel"]:
        try:
            ModelRegistry.register_model(
                arch_name,
                "models.demos.wormhole.bge_m3.demo.generator_vllm:BgeM3ForEmbedding",
            )
            logger.warning(
                "sitecustomize: registered BGE-M3 model architecture in subprocess: %s",
                arch_name,
            )
        except Exception as e:
            logger.warning(
                "sitecustomize: failed to register BGE-M3 architecture %s: %s",
                arch_name,
                e,
            )




def _patch_vllm_tt_loader_vllm_config_passthrough() -> None:
    if not _is_bge_context():
        return

    try:
        from vllm.model_executor.model_loader.tt_loader import TTModelLoader
        from vllm.model_executor.model_loader.utils import get_model_architecture
        from vllm.config.vllm import set_current_vllm_config
    except Exception as e:  # pragma: no cover
        logger.debug("sitecustomize: TTModelLoader import skipped: %s", e)
        return

    original = TTModelLoader.load_model

    def _patched(self, vllm_config, model_config):
        model_class, _ = get_model_architecture(model_config)
        if not hasattr(model_class, "initialize_vllm_model"):
            return original(self, vllm_config, model_config)

        device_config = vllm_config.device_config
        scheduler_config = vllm_config.scheduler_config
        data_parallel = vllm_config.parallel_config.data_parallel_size
        max_batch_size = scheduler_config.max_num_seqs * data_parallel

        optimizations = None
        if getattr(model_config, "override_tt_config", None):
            optimizations = model_config.override_tt_config.get("optimizations", None)

        init_kwargs = {
            "max_seq_len": model_config.max_model_len,
            "tt_data_parallel": data_parallel,
            "optimizations": optimizations,
        }

        # Ensure adapter/pooler init can resolve current vLLM config.
        with set_current_vllm_config(vllm_config, prefix="model"):
            try:
                return model_class.initialize_vllm_model(
                    model_config.hf_config,
                    device_config.device,
                    max_batch_size,
                    vllm_config=vllm_config,
                    **init_kwargs,
                )
            except TypeError as e:
                # Backward-compatible fallback only for models that don't
                # accept vllm_config kwarg.
                if "vllm_config" not in str(e):
                    raise
                logger.warning(
                    "sitecustomize: %s.initialize_vllm_model rejected vllm_config; retrying without it",
                    model_class.__name__,
                )
                return model_class.initialize_vllm_model(
                    model_config.hf_config,
                    device_config.device,
                    max_batch_size,
                    **init_kwargs,
                )

    TTModelLoader.load_model = _patched
    logger.warning("sitecustomize: installed TTModelLoader vllm_config passthrough patch for bge-m3")


_patch_vllm_tt_loader_vllm_config_passthrough()

_patch_bge_model_registry_registration()


def _force_bge_runner_type_pooling() -> None:
    if not _is_bge_context():
        return

    try:
        from vllm.worker.tt_worker import TTWorker
    except Exception as e:  # pragma: no cover
        logger.debug("sitecustomize: TTWorker import skipped for runner_type patch: %s", e)
        return

    original_load_model = TTWorker.load_model

    def _patched_load_model(self):
        try:
            self.model_config.runner_type = "pooling"
            logger.warning("sitecustomize: forced model_config.runner_type=pooling for bge-m3")
        except Exception as e:
            logger.warning("sitecustomize: failed to force runner_type=pooling: %s", e)
        return original_load_model(self)

    TTWorker.load_model = _patched_load_model
    logger.warning("sitecustomize: installed TTWorker runner_type pooling patch for bge-m3")


_force_bge_runner_type_pooling()


def _patch_v1_tt_model_runner_kv_for_embedding() -> None:
    if not _is_bge_context():
        return

    try:
        from vllm.v1.worker.tt_model_runner import TTModelRunner
    except Exception as e:  # pragma: no cover
        logger.debug("sitecustomize: v1 TTModelRunner import skipped for kv patch: %s", e)
        return

    original_init_kv = TTModelRunner.initialize_kv_cache

    def _patched_init_kv(self, kv_cache_config):
        m = getattr(self, "model", None)
        # Embedding wrappers do not implement KV cache path.
        if m is not None and hasattr(m, "get_embedding_dim") and not hasattr(m, "allocate_kv_cache"):
            self.kv_caches = []
            self.input_batch = None
            self.max_num_blocks_per_req = 0
            logger.warning(
                "sitecustomize: skipping TTModelRunner.initialize_kv_cache for embedding model %s",
                type(m).__name__,
            )
            return
        return original_init_kv(self, kv_cache_config)

    TTModelRunner.initialize_kv_cache = _patched_init_kv
    logger.warning("sitecustomize: installed v1 TTModelRunner KV bypass patch for bge-m3")


_patch_v1_tt_model_runner_kv_for_embedding()


def _patch_v1_tt_model_runner_warmup_for_embedding() -> None:
    if not _is_bge_context():
        return

    try:
        from vllm.v1.worker.tt_model_runner import TTModelRunner
    except Exception as e:  # pragma: no cover
        logger.debug("sitecustomize: v1 TTModelRunner import skipped for warmup patch: %s", e)
        return

    original_warmup = TTModelRunner.warmup_model

    def _patched_warmup(self):
        m = getattr(self, "model", None)
        if m is not None and hasattr(m, "get_embedding_dim") and not hasattr(m, "warmup_model_prefill"):
            logger.warning(
                "sitecustomize: skipping TTModelRunner.warmup_model for embedding model %s",
                type(m).__name__,
            )
            return
        return original_warmup(self)

    TTModelRunner.warmup_model = _patched_warmup
    logger.warning("sitecustomize: installed v1 TTModelRunner warmup bypass patch for bge-m3")


_patch_v1_tt_model_runner_warmup_for_embedding()


def _patch_v1_tt_model_runner_pooling_tasks_for_embedding() -> None:
    if not _is_bge_context():
        return

    try:
        from vllm.v1.worker.tt_model_runner import TTModelRunner
    except Exception as e:  # pragma: no cover
        logger.debug("sitecustomize: v1 TTModelRunner import skipped for pooling task patch: %s", e)
        return

    original = TTModelRunner.get_supported_pooling_tasks

    def _patched(self):
        # For bge-m3 bring-up, always expose embed task from TTModelRunner.
        # Relying on model attributes is brittle across worker init timing.
        logger.warning(
            "sitecustomize: forcing TTModelRunner supported pooling tasks to ['embed'] in bge context"
        )
        return ["embed"]

    TTModelRunner.get_supported_pooling_tasks = _patched
    logger.warning("sitecustomize: installed v1 TTModelRunner pooling-task patch for bge-m3")


_patch_v1_tt_model_runner_pooling_tasks_for_embedding()
