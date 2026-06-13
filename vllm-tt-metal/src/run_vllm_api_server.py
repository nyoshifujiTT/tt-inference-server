# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import argparse
import importlib
import inspect
import json
import logging
import math
import multiprocessing
import os
import runpy
import shlex
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from vllm import ModelRegistry

try:
    import transformers
except ImportError:
    transformers = None


def apply_transformers_compat() -> None:
    if transformers is None:
        return

    if not hasattr(transformers, "AutoModelForVision2Seq") and hasattr(
        transformers, "AutoModelForImageTextToText"
    ):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText

    def _all_special_tokens_extended(self):
        return list(getattr(self, "all_special_tokens", []))

    for cls_name in ("Qwen2Tokenizer", "Qwen2TokenizerFast"):
        cls = getattr(transformers, cls_name, None)
        if cls is not None and not hasattr(cls, "all_special_tokens_extended"):
            cls.all_special_tokens_extended = property(_all_special_tokens_extended)

    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        PreTrainedTokenizerBase = None

    if PreTrainedTokenizerBase is not None and not hasattr(
        PreTrainedTokenizerBase, "all_special_tokens_extended"
    ):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            _all_special_tokens_extended
        )


apply_transformers_compat()


def apply_huggingface_local_path_compat() -> None:
    import huggingface_hub

    if (
        getattr(huggingface_hub.snapshot_download, "__name__", "")
        == "_snapshot_download_local_path_compat"
    ):
        return

    original_snapshot_download = huggingface_hub.snapshot_download

    def _snapshot_download_local_path_compat(repo_id, *args, **kwargs):
        repo_path = Path(repo_id)
        if repo_path.exists():
            return str(repo_path)
        return original_snapshot_download(repo_id, *args, **kwargs)

    huggingface_hub.snapshot_download = _snapshot_download_local_path_compat
    globals()["snapshot_download"] = _snapshot_download_local_path_compat


apply_huggingface_local_path_compat()


def apply_vllm_qwen35_compat() -> None:
    try:
        from vllm.config.model import ModelConfig
    except Exception:
        return

    if (
        getattr(ModelConfig.get_num_layers_by_block_type, "__name__", "")
        == "_get_num_layers_by_block_type_qwen35_compat"
    ):
        return

    def _get_num_layers_by_block_type_qwen35_compat(
        self, parallel_config, block_type="attention"
    ):
        attn_block_type = block_type == "attention"
        is_transformer = (
            not self.is_hybrid and not self.has_noops and not self.is_attention_free
        )
        start, end = self.get_layers_start_end_indices(parallel_config)

        if is_transformer:
            return end - start if attn_block_type else 0
        if self.is_attention_free:
            return 0 if attn_block_type else end - start
        if self.has_noops:
            block_configs = self.hf_config.block_configs
            return sum(not bc.attention.no_op for bc in block_configs[start:end])

        layers_block_type_value = getattr(self.hf_text_config, "layers_block_type", None)
        if layers_block_type_value is not None:
            if hasattr(self.hf_text_config, "model_type") and (
                self.hf_text_config.model_type == "zamba2"
            ):
                if attn_block_type:
                    return sum(
                        t == "hybrid" for t in layers_block_type_value[start:end]
                    )
                return self.get_num_layers(parallel_config)
            return sum(t == block_type for t in layers_block_type_value[start:end])

        attn_type_list = getattr(self.hf_config, "attn_type_list", None)
        if attn_type_list:
            return sum(t == 1 for t in attn_type_list[start:end])

        layer_types_value = getattr(self.hf_config, "layer_types", None)
        if layer_types_value is None:
            layer_types_value = getattr(self.hf_text_config, "layer_types", None)
        if layer_types_value is not None:
            if block_type == "attention":
                return sum(t == "full_attention" for t in layer_types_value[start:end])
            if block_type == "linear_attention":
                return sum(t == "linear_attention" for t in layer_types_value[start:end])
            return sum(t == block_type for t in layer_types_value[start:end])

        raise ValueError(
            "The model is a hybrid model without layers_block_type, attn_type_list, or layer_types metadata."
        )

    ModelConfig.get_num_layers_by_block_type = _get_num_layers_by_block_type_qwen35_compat


apply_vllm_qwen35_compat()


def apply_qwen35_warmup_compat() -> None:
    try:
        import ttnn
        import torch
        from models.common.warmup.warmup_utils import WarmupForwardMixin
        from models.demos.blackhole.qwen3_5_9b.tt.qwen35_vllm import TTQwen35ForCausalLM
        from vllm.v1.worker.tt_model_runner import TTModelRunner
    except Exception:
        return

    if getattr(TTQwen35ForCausalLM, "_codex_qwen35_warmup_compat", False):
        return

    original_prefill_forward = TTQwen35ForCausalLM.prefill_forward
    original_decode_forward = TTQwen35ForCausalLM.decode_forward
    original_execute_with_model_input = TTModelRunner.execute_with_model_input
    original_update_states = TTModelRunner._update_states

    def _iter_qwen35_deltanet_layers(model):
        for layer in model.layers:
            if not layer.is_full_attention:
                yield layer.attention

    def _shape_tuple(shape):
        return tuple(int(dim) for dim in shape)

    def _get_qwen35_deltanet_qkv_dims(dn):
        q_dim = getattr(getattr(dn, "cfg", None), "q_dim", None)
        k_dim = getattr(getattr(dn, "cfg", None), "k_dim", None)
        v_dim = getattr(getattr(dn, "cfg", None), "v_dim", None)
        if q_dim is None or k_dim is None or v_dim is None:
            q_dim = getattr(getattr(dn, "args", None), "linear_q_dim")
            k_dim = getattr(getattr(dn, "args", None), "linear_k_dim")
            v_dim = getattr(getattr(dn, "args", None), "linear_v_dim")
        return int(q_dim), int(k_dim), int(v_dim)

    def _extract_qwen35_model(wrapper):
        return wrapper.model[0] if isinstance(wrapper.model, (list, tuple)) else wrapper.model

    def _extract_qwen35_wrapper_from_runner(runner):
        wrapper = runner.model[0] if isinstance(runner.model, (list, tuple)) else runner.model
        return wrapper if isinstance(wrapper, TTQwen35ForCausalLM) else None

    def _get_qwen35_request_state_store(wrapper):
        store = getattr(wrapper, "_codex_qwen35_request_state_store", None)
        if store is None:
            store = {}
            wrapper._codex_qwen35_request_state_store = store
        return store

    def _drop_qwen35_request_states(wrapper, req_ids):
        if not req_ids:
            return
        store = _get_qwen35_request_state_store(wrapper)
        for req_id in req_ids:
            store.pop(req_id, None)

    def _save_qwen35_request_states(wrapper, model, req_ids, external_states=None):
        req_ids = list(req_ids or [])
        if not req_ids:
            return

        store = _get_qwen35_request_state_store(wrapper)
        deltanet_layers = list(_iter_qwen35_deltanet_layers(model))
        state_batches = []
        for dn_idx, dn in enumerate(deltanet_layers):
            if external_states is not None:
                src_rec, src_conv = external_states[dn_idx]
            else:
                src_rec = dn.recurrent_state
                src_conv = dn.fused_conv_state
            if src_rec is None:
                raise RuntimeError("Qwen3.5 DeltaNet recurrent state missing while saving request state")
            rec_batch = ttnn.to_torch(src_rec)
            conv_batch = ttnn.to_torch(src_conv) if src_conv is not None else None
            state_batches.append((rec_batch, conv_batch))

        for row_idx, req_id in enumerate(req_ids):
            per_layer = []
            for rec_batch, conv_batch in state_batches:
                per_layer.append(
                    (
                        rec_batch[row_idx : row_idx + 1].clone().contiguous(),
                        None
                        if conv_batch is None
                        else conv_batch[row_idx : row_idx + 1].clone().contiguous(),
                    )
                )
            store[req_id] = per_layer

    def _load_qwen35_request_states(wrapper, model, req_ids, external_states):
        req_ids = list(req_ids or [])
        if not req_ids or external_states is None:
            return

        store = _get_qwen35_request_state_store(wrapper)
        for dn_idx, (dn, (ext_rec, ext_conv)) in enumerate(
            zip(_iter_qwen35_deltanet_layers(model), external_states)
        ):
            rec_torch = torch.zeros(_shape_tuple(ext_rec.shape), dtype=torch.bfloat16)
            conv_torch = (
                torch.zeros(_shape_tuple(ext_conv.shape), dtype=torch.bfloat16)
                if ext_conv is not None
                else None
            )

            for row_idx, req_id in enumerate(req_ids):
                per_layer = store.get(req_id)
                if not per_layer:
                    continue
                saved_rec, saved_conv = per_layer[dn_idx]
                rec_torch[row_idx : row_idx + 1] = saved_rec
                if conv_torch is not None and saved_conv is not None:
                    conv_torch[row_idx : row_idx + 1] = saved_conv

            rec_tt = ttnn.from_torch(
                rec_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=model.device,
            )
            ttnn.copy(rec_tt, ext_rec)
            ttnn.deallocate(rec_tt)

            if conv_torch is not None:
                conv_tt = ttnn.from_torch(
                    conv_torch,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=model.device,
                )
                ttnn.copy(conv_tt, ext_conv)
                ttnn.deallocate(conv_tt)

            dn.recurrent_state = ext_rec
            dn.fused_conv_state = ext_conv
            dn.conv_state_q = None
            dn.conv_state_k = None
            dn.conv_state_v = None
            dn.split_conv_state = None
            dn.use_inplace_state = True

    def _reset_qwen35_runtime_states(model):
        for dn in _iter_qwen35_deltanet_layers(model):
            dn.recurrent_state = None
            dn.conv_state_q = None
            dn.conv_state_k = None
            dn.conv_state_v = None
            dn.fused_conv_state = None
            dn.split_conv_state = None
            dn.use_inplace_state = False

    def _run_qwen35_serial_serving_prefill(self, model, tokens, page_table, kv_cache, prompt_lens, req_ids, **kwargs):
        if not isinstance(page_table, torch.Tensor):
            page_table = ttnn.to_torch(page_table)

        logits_rows = []
        rope_rows = []
        for row_idx, req_id in enumerate(req_ids):
            prompt_len = int(prompt_lens[row_idx])
            _reset_qwen35_runtime_states(model)
            output = original_prefill_forward(
                self,
                tokens=tokens[row_idx : row_idx + 1, :prompt_len].contiguous(),
                page_table=page_table[row_idx : row_idx + 1].contiguous(),
                kv_cache=kv_cache,
                prompt_lens=[prompt_len],
                **kwargs,
            )
            if isinstance(output, tuple) and len(output) == 2:
                user_logits, user_rope = output
            else:
                user_logits = output
                user_rope = torch.zeros(1, dtype=torch.long)
            _save_qwen35_request_states(self, model, [req_id])
            logits_rows.append(user_logits)
            rope_rows.append(user_rope)

        return torch.cat(logits_rows, dim=0), torch.cat(rope_rows, dim=0)

    def _broadcast_qwen35_deltanet_state_to_batch(model, target_batch_size, external_states=None):
        dn_idx = 0
        for dn in _iter_qwen35_deltanet_layers(model):
            rec = dn.recurrent_state
            if rec is None:
                raise RuntimeError("Qwen3.5 DeltaNet recurrent state is missing after prefill warmup")
            rec_torch = ttnn.to_torch(rec)
            if rec_torch.shape[0] == 1 and target_batch_size > 1:
                rec_torch = rec_torch.repeat(target_batch_size, 1, 1, 1)
            elif rec_torch.shape[0] != target_batch_size:
                raise RuntimeError(
                    f"Cannot broadcast Qwen3.5 recurrent state from batch {rec_torch.shape[0]} to {target_batch_size}"
                )
            rec_tt = ttnn.from_torch(
                rec_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=model.device,
            )

            conv_tt = None
            if dn.fused_conv_state is not None:
                conv_torch = ttnn.to_torch(dn.fused_conv_state)
                if conv_torch.shape[0] == 1 and target_batch_size > 1:
                    conv_torch = conv_torch.repeat(target_batch_size, 1, 1)
                elif conv_torch.shape[0] != target_batch_size:
                    raise RuntimeError(
                        f"Cannot broadcast Qwen3.5 fused conv state from batch {conv_torch.shape[0]} to {target_batch_size}"
                    )
                conv_tt = ttnn.from_torch(
                    conv_torch,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=model.device,
                )

            if external_states is not None:
                ext_rec, ext_conv = external_states[dn_idx]
                ttnn.copy(rec_tt, ext_rec)
                ttnn.deallocate(rec_tt)
                dn.recurrent_state = ext_rec
                if conv_tt is not None:
                    ttnn.copy(conv_tt, ext_conv)
                    ttnn.deallocate(conv_tt)
                    dn.fused_conv_state = ext_conv
                dn.split_conv_state = None
            else:
                dn.recurrent_state = rec_tt
                if conv_tt is not None:
                    dn.fused_conv_state = conv_tt
                dn.split_conv_state = None
            dn.use_inplace_state = True
            dn_idx += 1

    def _qwen35_layout_will_change(runner, scheduler_output):
        current_req_ids = list(getattr(runner.input_batch, "req_ids", []))
        current_req_id_to_index = getattr(runner.input_batch, "req_id_to_index", {})
        if scheduler_output.finished_req_ids:
            return True

        scheduled_req_ids = set(scheduler_output.num_scheduled_tokens.keys())
        if set(current_req_ids) != scheduled_req_ids:
            return True

        for new_req_data in scheduler_output.scheduled_new_reqs:
            if new_req_data.req_id not in current_req_id_to_index:
                return True

        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            if req_id not in current_req_id_to_index:
                return True

        return False

    def allocate_kv_cache_with_max_batch(self, kv_cache_shape, dtype, num_layers):
        _ = num_layers

        model = _extract_qwen35_model(self)
        batch_size = int(getattr(getattr(model, "args", None), "max_batch_size", 1) or 1)
        logger.info(
            "Allocating Qwen3.5 KV/GDN state with batch_size=%s for warmup/serving",
            batch_size,
        )

        if batch_size <= 1 or getattr(model, "num_devices", 1) > 1:
            return model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=batch_size)

        assert getattr(model, "_deltanet_external_states", None) is None, (
            "allocate_kv_caches already called; deallocate first"
        )

        kv_caches = []
        for idx in model._attention_layer_indices:
            _ = idx
            k_cache = ttnn.zeros(
                kv_cache_shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=model.device,
            )
            v_cache = ttnn.zeros(
                kv_cache_shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=model.device,
            )
            kv_caches.append([k_cache, v_cache])
        model.set_paged_kv_caches(kv_caches)

        model._deltanet_external_states = []
        for dn in _iter_qwen35_deltanet_layers(model):
            rec = ttnn.from_torch(
                torch.zeros(
                    batch_size,
                    dn.num_v_heads,
                    dn.head_k_dim,
                    dn.head_v_dim,
                    dtype=torch.bfloat16,
                ),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=model.device,
            )
            q_dim, k_dim, v_dim = _get_qwen35_deltanet_qkv_dims(dn)
            conv = ttnn.from_torch(
                torch.zeros(
                    batch_size,
                    dn.conv_kernel_size - 1,
                    q_dim + k_dim + v_dim,
                    dtype=torch.bfloat16,
                ),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=model.device,
            )
            dn.recurrent_state = rec
            dn.fused_conv_state = conv
            dn.conv_state_q = None
            dn.conv_state_k = None
            dn.conv_state_v = None
            dn.split_conv_state = None
            dn.use_inplace_state = True
            model._deltanet_external_states.append((rec, conv))

        self._codex_qwen35_batch_layout_changed = False
        return kv_caches

    def prefill_forward_with_rope_deltas(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        max_prompt_tokens = int(page_table.shape[1]) * 64 if page_table is not None else None
        if max_prompt_tokens is not None and tokens.shape[1] > max_prompt_tokens:
            raise RuntimeError(
                "Qwen3.5 prefill received %s tokens with only %s tokens of page-table coverage; fixed-width page-table / KV-block override is missing"
                % (tokens.shape[1], max_prompt_tokens)
            )

        model = _extract_qwen35_model(self)
        req_ids = list(getattr(self, "_codex_active_req_ids", []) or [])
        broadcast_batch = int(getattr(self, "_codex_prefill_state_broadcast_batch", 0) or 0)
        saved_external_states = getattr(model, "_deltanet_external_states", None)
        using_request_state_compat = bool(req_ids) and broadcast_batch == 0 and saved_external_states is not None

        if using_request_state_compat and not getattr(self, "_codex_logged_request_state_prefill", False):
            logger.warning(
                "Qwen3.5 request-scoped DeltaNet prefill compat active: req_count=%s",
                len(req_ids),
            )
            self._codex_logged_request_state_prefill = True

        if using_request_state_compat:
            model._deltanet_external_states = None
            _reset_qwen35_runtime_states(model)
        elif broadcast_batch > 1 and tokens.shape[0] == 1 and saved_external_states is not None:
            model._deltanet_external_states = None
            _reset_qwen35_runtime_states(model)
        else:
            saved_external_states = None

        try:
            if using_request_state_compat and int(getattr(tokens, "shape", [1])[0]) > 1:
                if not getattr(self, "_codex_logged_request_state_serial_prefill", False):
                    logger.warning(
                        "Qwen3.5 request-scoped DeltaNet serial prefill fallback active: req_count=%s",
                        len(req_ids),
                    )
                    self._codex_logged_request_state_serial_prefill = True
                output = _run_qwen35_serial_serving_prefill(
                    self,
                    model,
                    tokens=tokens,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    prompt_lens=prompt_lens,
                    req_ids=req_ids,
                    **kwargs,
                )
            else:
                output = original_prefill_forward(
                    self,
                    tokens=tokens,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    prompt_lens=prompt_lens,
                    **kwargs,
                )
        finally:
            if using_request_state_compat:
                model._deltanet_external_states = saved_external_states
            elif saved_external_states is not None:
                _broadcast_qwen35_deltanet_state_to_batch(
                    model,
                    broadcast_batch,
                    external_states=saved_external_states,
                )
                model._deltanet_external_states = saved_external_states

        if using_request_state_compat:
            output_tensor = output[0] if isinstance(output, tuple) and len(output) == 2 else output
            batch_size = int(getattr(output_tensor, "shape", [len(req_ids)])[0])
            active_req_ids = req_ids[:batch_size]
            if batch_size <= 1:
                _save_qwen35_request_states(self, model, active_req_ids)
            _load_qwen35_request_states(self, model, active_req_ids, saved_external_states)
            self._codex_qwen35_batch_layout_changed = False

        if isinstance(output, tuple) and len(output) == 2:
            return output
        batch_size = getattr(output, "shape", [1])[0] if output is not None else 1
        return output, torch.zeros(batch_size, dtype=torch.long)

    def decode_forward_with_request_state_compat(
        self, tokens, start_pos, page_table, kv_cache, enable_trace=False, read_from_device=True, **kwargs
    ):
        model = _extract_qwen35_model(self)
        req_ids = list(getattr(self, "_codex_active_req_ids", []) or [])
        external_states = getattr(model, "_deltanet_external_states", None)
        if (
            req_ids
            and external_states is not None
            and getattr(self, "_codex_qwen35_batch_layout_changed", False)
        ):
            if not getattr(self, "_codex_logged_request_state_decode", False):
                logger.warning(
                    "Qwen3.5 request-scoped DeltaNet decode reload active: req_count=%s",
                    len(req_ids),
                )
                self._codex_logged_request_state_decode = True
            _load_qwen35_request_states(self, model, req_ids, external_states)
            self._codex_qwen35_batch_layout_changed = False

        return original_decode_forward(
            self,
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            read_from_device=read_from_device,
            **kwargs,
        )

    def warmup_model_prefill(
        self, kv_cache, enable_trace, can_sample_on_device, non_greedy_decoding_on_device
    ):
        _ = can_sample_on_device
        _ = non_greedy_decoding_on_device
        if getattr(self, "_codex_prefill_warmed_up", False):
            return
        self._codex_prefill_warmed_up = True
        if kv_cache:
            valid_num_blocks = int(kv_cache[0][0].shape[0])
            num_blocks = max(1, math.ceil(valid_num_blocks / 32) * 32)
        else:
            valid_num_blocks = 1
            num_blocks = 32

        model = _extract_qwen35_model(self)
        batch_size = int(getattr(getattr(model, "args", None), "max_batch_size", 1) or 1)
        warmup_tokens = 128
        blocks_per_user = max(1, math.ceil(warmup_tokens / 64))

        tokens = torch.zeros(1, warmup_tokens, dtype=torch.int32)
        prompt_lens = [warmup_tokens]
        page_table = torch.zeros(1, num_blocks, dtype=torch.int32)
        if blocks_per_user > valid_num_blocks:
            raise RuntimeError(
                f"Qwen3.5 prefill warmup needs {blocks_per_user} KV blocks but only {valid_num_blocks} are available"
            )
        page_table[0, :blocks_per_user] = torch.arange(0, blocks_per_user, dtype=torch.int32)

        logger.info(
            "Running Qwen3.5 prefill warmup for %s tokens, then broadcasting DeltaNet state to batch_size=%s",
            warmup_tokens,
            batch_size,
        )
        self._codex_prefill_state_broadcast_batch = batch_size
        try:
            self.prefill_forward(
                tokens=tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                enable_trace=enable_trace,
                sampling_params=None,
            )
        finally:
            self._codex_prefill_state_broadcast_batch = 0

    def execute_with_model_input_with_qwen35_request_ids(self, model_input):
        wrapper = _extract_qwen35_wrapper_from_runner(self)
        if wrapper is None:
            return original_execute_with_model_input(self, model_input)

        wrapper._codex_active_req_ids = list(getattr(self.input_batch, "req_ids", []))
        try:
            return original_execute_with_model_input(self, model_input)
        finally:
            wrapper._codex_active_req_ids = None

    def update_states_with_qwen35_request_snapshot(self, scheduler_output):
        wrapper = _extract_qwen35_wrapper_from_runner(self)
        if wrapper is None:
            return original_update_states(self, scheduler_output)

        model = _extract_qwen35_model(wrapper)
        prev_req_ids = list(getattr(self.input_batch, "req_ids", []))
        layout_will_change = _qwen35_layout_will_change(self, scheduler_output)
        if layout_will_change and prev_req_ids and getattr(model, "_deltanet_external_states", None) is not None:
            _save_qwen35_request_states(
                wrapper,
                model,
                prev_req_ids,
                external_states=model._deltanet_external_states,
            )

        original_update_states(self, scheduler_output)
        _drop_qwen35_request_states(wrapper, scheduler_output.finished_req_ids)
        wrapper._codex_qwen35_batch_layout_changed = layout_will_change

    TTQwen35ForCausalLM.allocate_kv_cache = allocate_kv_cache_with_max_batch
    TTQwen35ForCausalLM.prefill_forward = prefill_forward_with_rope_deltas
    TTQwen35ForCausalLM.decode_forward = decode_forward_with_request_state_compat
    TTQwen35ForCausalLM._create_sampling_params = WarmupForwardMixin._create_sampling_params
    TTQwen35ForCausalLM._create_decode_warmup_inputs = (
        WarmupForwardMixin._create_decode_warmup_inputs
    )
    TTQwen35ForCausalLM.warmup_model_decode = WarmupForwardMixin.warmup_model_decode
    TTQwen35ForCausalLM.warmup_model_prefill = warmup_model_prefill
    TTQwen35ForCausalLM._codex_qwen35_warmup_compat = True
    TTModelRunner.execute_with_model_input = execute_with_model_input_with_qwen35_request_ids
    TTModelRunner._update_states = update_states_with_qwen35_request_snapshot


apply_qwen35_warmup_compat()


def apply_qwen35_deltanet_decode_debug_compat() -> None:
    try:
        import models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_deltanet as qwen35_gated_deltanet
    except Exception:
        return

    if getattr(qwen35_gated_deltanet, "_codex_deltanet_decode_debug_compat", False):
        return

    original_forward = qwen35_gated_deltanet.Qwen35GatedDeltaNet.forward

    def _forward_with_decode_debug(self, x, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else None)
        if mode == "recurrent" and not getattr(self, "_codex_logged_recurrent_shape", False):
            logging.getLogger(__name__).warning(
                "Qwen3.5 DeltaNet recurrent call: x.shape=%s recurrent_state.shape=%s fused_conv_state.shape=%s split_conv_state_len=%s use_inplace_state=%s",
                getattr(x, "shape", None),
                getattr(getattr(self, "recurrent_state", None), "shape", None),
                getattr(getattr(self, "fused_conv_state", None), "shape", None),
                len(getattr(self, "split_conv_state", []) or []),
                getattr(self, "use_inplace_state", None),
            )
            self._codex_logged_recurrent_shape = True
        return original_forward(self, x, *args, **kwargs)

    qwen35_gated_deltanet.Qwen35GatedDeltaNet.forward = _forward_with_decode_debug
    qwen35_gated_deltanet._codex_deltanet_decode_debug_compat = True


apply_qwen35_deltanet_decode_debug_compat()


def apply_qwen35_decode_memory_compat() -> None:
    try:
        import ttnn
        import models.demos.blackhole.qwen3_5_9b.tt.qwen35_gated_attention as qwen35_gated_attention
    except Exception:
        return

    if getattr(qwen35_gated_attention, "_codex_decode_memory_compat", False):
        return

    original_forward = qwen35_gated_attention.gated_attention_forward_ttnn

    def _codex_num_cores_to_corerangeset(target_num_cores, grid_size, row_wise=True):
        num_cores_x = int(grid_size.x)
        num_cores_y = int(grid_size.y)
        total_available_cores = num_cores_x * num_cores_y
        target_num_cores = int(target_num_cores)
        if target_num_cores <= 0 or target_num_cores > total_available_cores:
            raise RuntimeError(
                f"Requested {target_num_cores} decode shards for grid {grid_size}, but only {total_available_cores} compute cores are available"
            )

        ranges = []
        remaining = target_num_cores
        x = 0
        y = 0
        if row_wise:
            while remaining > 0:
                take = min(num_cores_x - x, remaining)
                ranges.append(
                    ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x + take - 1, y))
                )
                remaining -= take
                x = 0
                y += 1
        else:
            while remaining > 0:
                take = min(num_cores_y - y, remaining)
                ranges.append(
                    ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y + take - 1))
                )
                remaining -= take
                x += 1
                y = 0

        return ttnn.CoreRangeSet(set(ranges))

    patched_source = inspect.getsource(original_forward)
    replacements = {
        "        _shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(B - 1, 0))})": "        _shard_grid = _codex_num_cores_to_corerangeset(B, device.compute_with_storage_grid_size(), row_wise=True)",
        "            _shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(N - 1, 0))})": "            _shard_grid = _codex_num_cores_to_corerangeset(N, device.compute_with_storage_grid_size(), row_wise=True)",
        "        q_decode = ttnn.transpose(query_states, 1, 2)  # [B, H_q, 1, D] -> [B, 1, H_q, D] = [1, B, H_q, D] for B=1": (
            "        if B > 1:\n"
            "            q_decode = ttnn.reshape(query_states, [1, B, num_attention_heads, head_dim])\n"
            "        else:\n"
            "            q_decode = ttnn.transpose(query_states, 1, 2)"
        ),
        "        attn_output = ttnn.transpose(attn_output, 1, 2)  # back to [B, H_q, 1, D]": (
            "        if B > 1:\n"
            "            attn_output = ttnn.reshape(attn_output, [B, num_attention_heads, 1, head_dim])\n"
            "        else:\n"
            "            attn_output = ttnn.transpose(attn_output, 1, 2)"
        ),
    }
    replaced_any = False
    for before, after in replacements.items():
        if before in patched_source:
            patched_source = patched_source.replace(before, after)
            replaced_any = True

    patched_forward = original_forward
    if replaced_any:
        original_globals = original_forward.__globals__
        original_globals["_codex_num_cores_to_corerangeset"] = _codex_num_cores_to_corerangeset
        exec(
            compile(
                patched_source,
                inspect.getsourcefile(original_forward) or "<codex_qwen35_gated_attention>",
                "exec",
            ),
            original_globals,
        )
        patched_forward = original_globals["gated_attention_forward_ttnn"]
    else:
        logging.getLogger(__name__).warning(
            "Could not find the expected Qwen3.5 gated-attention decode shard-grid pattern to patch"
        )

    def _gated_attention_forward_ttnn_compat(*args, **kwargs):
        hidden_states = kwargs.get("hidden_states")
        page_table = kwargs.get("page_table")
        if page_table is not None and hidden_states is not None:
            seq_len = getattr(hidden_states, "shape", [None, None])[1]
            if seq_len == 1:
                kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        return patched_forward(*args, **kwargs)

    qwen35_gated_attention.gated_attention_forward_ttnn = _gated_attention_forward_ttnn_compat
    qwen35_gated_attention._codex_decode_memory_compat = True


apply_qwen35_decode_memory_compat()


def apply_qwen35_kv_block_compat() -> None:
    try:
        from vllm.v1.worker import tt_worker as v1_tt_worker
    except Exception:
        return

    if getattr(v1_tt_worker, "_codex_qwen35_kv_block_compat", False):
        return

    original_ttworker_init = v1_tt_worker.TTWorker.__init__
    original_get_num_available_blocks_tt = v1_tt_worker.get_num_available_blocks_tt

    def _is_qwen35_9b_model(vllm_config) -> bool:
        model_name = getattr(vllm_config.model_config, "model", "") or ""
        return "Qwen3.5-9B" in model_name

    def _patched_ttworker_init(self, vllm_config, *args, **kwargs):
        if _is_qwen35_9b_model(vllm_config):
            logical_block_size = 64
            existing_block_size = int(vllm_config.cache_config.block_size)
            if existing_block_size != logical_block_size:
                logging.getLogger(__name__).warning(
                    "Overriding Qwen3.5-9B cache block_size from %s to logical page-table block size %s",
                    existing_block_size,
                    logical_block_size,
                )
                vllm_config.cache_config.block_size = logical_block_size
        return original_ttworker_init(self, vllm_config, *args, **kwargs)

    def _patched_get_num_available_blocks_tt(vllm_config):
        if _is_qwen35_9b_model(vllm_config):
            block_size = 64
            max_batch = int(vllm_config.scheduler_config.max_num_seqs)
            max_tokens_all_users = int(
                getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 0)
                or getattr(vllm_config.model_config, "max_model_len", 0)
                or 131072
            )
            num_blocks = math.ceil(
                (max_tokens_all_users + block_size * max_batch) / block_size
            )
            logging.getLogger(__name__).warning(
                "Forcing Qwen3.5-9B TT KV geometry to block_size=%s max_tokens_all_users=%s num_blocks=%s",
                block_size,
                max_tokens_all_users,
                num_blocks,
            )
            return num_blocks
        return original_get_num_available_blocks_tt(vllm_config)

    v1_tt_worker.TTWorker.__init__ = _patched_ttworker_init
    v1_tt_worker.get_num_available_blocks_tt = _patched_get_num_available_blocks_tt
    v1_tt_worker._codex_qwen35_kv_block_compat = True


apply_qwen35_kv_block_compat()

from utils.cache_monitor import get_container_cache_dir
from utils.device_utils import get_mesh_device_name
from utils.logging_utils import set_vllm_logging_config
from utils.prompt_client import run_background_trace_capture
from utils.vllm_run_utils import (
    create_model_symlink,
    get_encoded_api_key,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_VLLM_SERVER_PORT = "8000"


def parse_args():
    """Parse wrapper CLI args and return remaining vLLM passthrough args."""
    parser = argparse.ArgumentParser(description="TT vLLM API Server")
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model repo (e.g., meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--tt-device",
        type=str,
        required=True,
        help="Device type (e.g., n300, t3k, galaxy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device type (e.g., n300, t3k, galaxy)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["vllm", "media", "forge"],
        help="Inference engine override (vllm/media/forge).",
    )
    parser.add_argument(
        "--impl",
        type=str,
        help="Implementation name override (e.g. tt-transformers).",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable vLLM API key authorization (skips JWT_SECRET requirement)",
    )
    parser.add_argument(
        "--disable-trace-capture",
        action="store_true",
        help="Disable automatic trace capture requests on server startup",
    )
    parser.add_argument(
        "--service-port",
        type=int,
        default=None,
        help="Service port for vLLM server and trace capture client",
    )
    # Parse known args to allow vLLM args to pass through
    args, remaining_args = parser.parse_known_args()

    return args, remaining_args


def normalize_device_type(device_arg: str) -> str:
    """Convert user-provided device string to canonical device type name.

    Args:
        device_arg: User-provided device type (e.g., "n300", "galaxy", "T3K")

    Returns:
        Canonical device type name (e.g., "N300", "GALAXY", "T3K")
    """
    return device_arg.upper()


def normalize_engine_type(engine_arg: Optional[str]) -> Optional[str]:
    if not engine_arg:
        return None
    engine_map = {
        "vllm": "vLLM",
        "media": "media",
        "forge": "forge",
    }
    return engine_map[engine_arg.lower()]


def unwrap_model_specs_catalog(model_specs: dict) -> dict:
    """Return the nested model specs catalog from wrapped or legacy JSON."""
    if "model_specs" in model_specs and isinstance(model_specs["model_specs"], dict):
        return model_specs["model_specs"]
    return model_specs


def load_model_spec(
    model_arg: Optional[str],
    device_arg: Optional[str],
    engine_arg: Optional[str] = None,
    impl_arg: Optional[str] = None,
) -> dict:
    """Load and resolve a single model spec.

    Resolution order:
    1. Runtime mode: RUNTIME_MODEL_SPEC_JSON_PATH points to a pre-resolved spec
       (produced by run.py --docker-server)
    2. Catalog mode: MODEL_SPECS_JSON_PATH + --model/--tt-device/--device (+ optional
       --engine/--impl) are used to resolve one spec from the built-in catalog.

    Returns:
        dict: The resolved single model spec.

    Raises:
        RuntimeError: If runtime path is not available and required CLI args are missing.
    """
    runtime_path = os.getenv("RUNTIME_MODEL_SPEC_JSON_PATH")
    if runtime_path:
        runtime_path = Path(runtime_path)
        if runtime_path.exists():
            logger.info(
                "Using pre-resolved runtime model spec from "
                f"RUNTIME_MODEL_SPEC_JSON_PATH={runtime_path}"
            )
            logger.info(f"Loading runtime model spec from: {runtime_path}")
            with open(runtime_path, "r") as f:
                data = json.load(f)
            return data.get("runtime_model_spec", data)
        logger.warning(
            f"RUNTIME_MODEL_SPEC_JSON_PATH={runtime_path} does not exist, "
            "falling back to default model spec catalog."
        )

    if not model_arg or not device_arg:
        raise RuntimeError(
            "Either set RUNTIME_MODEL_SPEC_JSON_PATH env var "
            "(for 'python run.py --docker-server' workflow), or provide --model and "
            "--tt-device/--device for direct docker run. "
            "Example: docker run <image> --model meta-llama/Llama-3.1-8B --tt-device n300"
        )

    # Catalog mode (model_spec.json built into image)
    specs_path = os.getenv(
        "MODEL_SPECS_JSON_PATH",
        "/home/container_app_user/model_specs/model_spec.json",
    )
    logger.info(f"Loading all model specs from MODEL_SPECS_JSON_PATH: {specs_path}")
    with open(specs_path, "r") as f:
        model_specs = unwrap_model_specs_catalog(json.load(f))

    device_type = normalize_device_type(device_arg)
    model_spec = find_default_impl(
        model_specs,
        model_arg,
        device_type,
        engine_arg=engine_arg,
        impl_arg=impl_arg,
    )
    logger.info(
        f"Using default interface: found model spec for --model={model_arg}, "
        f"--device={device_type}, --engine={engine_arg}, --impl={impl_arg}"
    )
    return model_spec


def _resolve_hf_repo(model_specs: dict, model_arg: str) -> str:
    """Resolve model_arg to an hf_model_repo key in model_specs.

    Tries exact match first, then falls back to matching the short model name
    (last path segment) against all hf_model_repo keys.

    Args:
        model_specs: Nested model specs dict keyed by hf_model_repo at top level
        model_arg: The --model argument (HuggingFace repo or model name)

    Returns:
        The matching hf_model_repo key

    Raises:
        ValueError: If no matching hf_model_repo is found
    """
    if model_arg in model_specs:
        return model_arg

    short_name = model_arg.split("/")[-1]
    for hf_repo in model_specs:
        if hf_repo.split("/")[-1] == short_name:
            return hf_repo

    raise ValueError(
        f"No model spec found for model={model_arg}. "
        f"Available models: {list(model_specs.keys())[:10]}..."
    )


def find_default_impl(
    model_specs: dict,
    model_arg: str,
    device_type: str,
    engine_arg: Optional[str] = None,
    impl_arg: Optional[str] = None,
) -> dict:
    """Find the default implementation spec for a given model and device.

    Navigates the nested model spec structure to find the spec with
    default_impl=True for the given hf_model_repo and device_type.

    Args:
        model_specs: Nested dict: hf_model_repo > device_type > engine > impl_id > spec
        model_arg: The --model argument (HuggingFace repo or model name)
        device_type: Canonical device type name (e.g., "N300", "GALAXY")

    Returns:
        dict: The matching model spec with default_impl=True

    Raises:
        ValueError: If no matching spec is found
    """
    hf_repo = _resolve_hf_repo(model_specs, model_arg)
    device_specs = model_specs[hf_repo].get(device_type)
    if not device_specs:
        available_devices = list(model_specs[hf_repo].keys())
        raise ValueError(
            f"No model spec found for model={model_arg}, device={device_type}. "
            f"Available devices for {hf_repo}: {available_devices}"
        )

    if engine_arg:
        device_specs = {engine_arg: device_specs.get(engine_arg, {})}

    for engine_specs in device_specs.values():
        for spec in engine_specs.values():
            spec_impl_name = spec.get("impl", {}).get("impl_name")
            if impl_arg and spec_impl_name != impl_arg:
                continue
            if spec.get("device_model_spec", {}).get("default_impl"):
                return spec

    for engine_specs in device_specs.values():
        for spec in engine_specs.values():
            spec_impl_name = spec.get("impl", {}).get("impl_name")
            if impl_arg and spec_impl_name != impl_arg:
                continue
            return spec

    raise ValueError(
        f"No default_impl found for model={model_arg}, device={device_type}, "
        f"engine={engine_arg}, impl={impl_arg}. "
        f"Check that at least one impl has default_impl=True."
    )


def ensure_weights_available(model_spec: dict) -> Path:
    """Ensure model weights are available, downloading if necessary.

    If MODEL_WEIGHTS_DIR is already set (e.g. from --host-weights-dir bind mount),
    uses that directory directly and skips downloading.

    Args:
        model_spec: The model specification dictionary

    Returns:
        Path: Path to the model weights directory
    """
    # If MODEL_WEIGHTS_DIR is already set, use it directly and skip downloading
    model_weights_dir = os.getenv("MODEL_WEIGHTS_DIR")
    if model_weights_dir:
        weights_path = Path(model_weights_dir)
        if not weights_path.exists():
            raise RuntimeError(
                f"MODEL_WEIGHTS_DIR={model_weights_dir} does not exist. "
                "Ensure the host directory is correctly bind-mounted."
            )
        if not any(weights_path.iterdir()):
            raise RuntimeError(
                f"MODEL_WEIGHTS_DIR={model_weights_dir} is empty. "
                "Ensure the host directory contains model weight files."
            )
        logger.info(f"Using pre-mounted weights from MODEL_WEIGHTS_DIR: {weights_path}")
        return weights_path

    # Default: download weights into cache_root
    cache_root = Path(os.getenv("CACHE_ROOT", "/home/container_app_user/cache_root"))
    model_name = model_spec["model_name"]
    weights_path = cache_root / "weights" / model_name

    if not weights_path.exists() or not any(weights_path.iterdir()):
        hf_repo = model_spec.get("hf_weights_repo") or model_spec["hf_model_repo"]
        logger.info(f"Downloading weights from {hf_repo} to {weights_path}")
        weights_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=hf_repo, local_dir=weights_path)
    else:
        logger.info(f"Weights already exist at {weights_path}")

    os.environ["MODEL_WEIGHTS_DIR"] = str(weights_path)
    return weights_path


def set_cache_paths(model_spec: dict, device_type: str):
    """Set TT_CACHE_PATH and MESH_DEVICE for model-specific cache directory.

    Args:
        model_spec: The model specification dictionary
        device_type: Canonical device type name (e.g., "N300", "GALAXY")
    """
    mesh_device = get_mesh_device_name(device=device_type)
    tt_cache_path = get_container_cache_dir(model_spec, device=device_type)
    if tt_cache_path is None:
        raise RuntimeError("Could not resolve TT cache path from model spec.")

    # Set MESH_DEVICE env var for other components that need it
    os.environ["MESH_DEVICE"] = mesh_device
    logger.info(f"Set MESH_DEVICE to {mesh_device}")

    tt_cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["TT_CACHE_PATH"] = str(tt_cache_path)
    logger.info(f"Set TT_CACHE_PATH to {tt_cache_path}")


def register_tt_models(impl_id=None):
    """Configure vLLM ModelRegistry according to ModelSpec.impl.impl_id.

    Args:
        impl_id: Implementation ID from ModelSpec JSON (e.g., "tt_transformers",
                 "llama3_70b_galaxy", "qwen3_32b_galaxy"). If None, defaults to
                 "tt_transformers".
    """
    impl_id = impl_id or "tt_transformers"

    # Llama path selection based on impl_id
    if impl_id == "llama3_70b_galaxy":
        os.environ["TT_LLAMA_TEXT_VER"] = "llama3_70b_galaxy"
    else:  # default: tt_transformers
        os.environ["TT_LLAMA_TEXT_VER"] = "tt_transformers"

    # Qwen3 env var setting based on impl_id
    if impl_id == "qwen3_32b_galaxy":
        os.environ["TT_QWEN3_TEXT_VER"] = "qwen3_32b_galaxy"
    else:
        os.environ["TT_QWEN3_TEXT_VER"] = "tt_transformers"

    # Arcee AFM-4.5B - Text
    ModelRegistry.register_model(
        "TTArceeForCausalLM",
        "models.tt_transformers.tt.generator_vllm:TTArceeForCausalLM",
    )

    # Qwen3.5-9B (hybrid DeltaNet + Full Attention) - Blackhole P150.
    # The HF arch is the *VL* name (Qwen3_5ForConditionalGeneration) even though
    # this checkpoint is text-only. Point that arch at vLLM's native *text* class
    # so ModelConfig validation sees a text-generation, non-multimodal model
    # (passes --runner generate, skips the MM pipeline; it is never instantiated
    # -- the TT loader builds the model). The TT-prefixed alias routes execution
    # to the tt-metal class.
    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration",
        "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM",
    )
    ModelRegistry.register_model(
        "TTQwen3_5ForConditionalGeneration",
        resolve_qwen35_blackhole_entrypoint(),
    )


def resolve_qwen35_blackhole_entrypoint() -> str:
    module_name = "models.demos.blackhole.qwen3_5_9b.tt.qwen35_vllm"
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return f"{module_name}:Qwen35ForCausalLM"

    for class_name in ("Qwen35ForCausalLM", "TTQwen35ForCausalLM"):
        if hasattr(module, class_name):
            logger.info(
                "Using Qwen3.5 Blackhole vLLM class %s.%s", module_name, class_name
            )
            return f"{module_name}:{class_name}"

    raise AttributeError(
        f"Neither Qwen35ForCausalLM nor TTQwen35ForCausalLM exists in {module_name}"
    )


def model_setup(model_spec_json):
    # step 1: validate env vars passed in
    cache_root = Path(os.getenv("CACHE_ROOT"))
    assert cache_root.exists(), f"CACHE_ROOT: {cache_root} does not exist"
    symlinks_dir = cache_root / "model_file_symlinks_map"
    symlinks_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"MODEL_WEIGHTS_DIR: {os.getenv('MODEL_WEIGHTS_DIR')}")
    assert os.getenv("MODEL_WEIGHTS_DIR") is not None, "MODEL_WEIGHTS_DIR must be set"
    weights_dir = Path(os.getenv("MODEL_WEIGHTS_DIR"))
    assert weights_dir.exists(), f"MODEL_WEIGHTS_DIR: {weights_dir} does not exist"

    logging.info(f"TT_CACHE_PATH: {os.getenv('TT_CACHE_PATH')}")
    assert os.getenv("TT_CACHE_PATH") is not None, "TT_CACHE_PATH must be set"

    # step 2: set default runtime env vars
    # set up logging
    config_path, log_path = set_vllm_logging_config(level="DEBUG")
    logger.info(f"setting vllm logging config at: {config_path}")
    logger.info(f"setting vllm logging file at: {log_path}")

    # set HF_MODEL environment variable for loading
    logging.info(f"HF model setup for {model_spec_json['hf_model_repo']}")
    model_dir_name = model_spec_json["hf_model_repo"].split("/")[-1]

    source_tensor_cache_dir = weights_dir / "tensor_cache_bfp8"
    if source_tensor_cache_dir.exists():
        logger.info(
            "Creating writable local model view with symlinked files and a writable tensor_cache_bfp8 dir"
        )
        file_symlinks_map = {
            item.name: item.name for item in weights_dir.iterdir() if item.is_file()
        }
        hf_dir = create_model_symlink(
            symlinks_dir,
            model_dir_name,
            weights_dir,
            file_symlinks_map=file_symlinks_map,
        )
        writable_tensor_cache_dir = hf_dir / "tensor_cache_bfp8"
        writable_tensor_cache_dir.mkdir(parents=True, exist_ok=True)
        for source_path in sorted(source_tensor_cache_dir.rglob("*")):
            relative_path = source_path.relative_to(source_tensor_cache_dir)
            dest_path = writable_tensor_cache_dir / relative_path
            if source_path.is_dir():
                dest_path.mkdir(parents=True, exist_ok=True)
            elif not dest_path.exists():
                dest_path.symlink_to(source_path)
    else:
        hf_dir = create_model_symlink(symlinks_dir, model_dir_name, weights_dir)

    dynamic_env_vars = {
        "VLLM_LOGGING_CONFIG_PATH": str(config_path),
        "HF_MODEL": hf_dir,
    }

    # Set dynamic environment variables
    logger.info("setting dynamic runtime environment variables:")
    for key, value in dynamic_env_vars.items():
        if value is not None:
            logger.info(f"setting env var: {key}={value}")
            os.environ[key] = str(value)
        elif key in os.environ:
            logger.warning(
                f"removing env var: {key} from os.environ, previous value={os.environ[key]}"
            )
            del os.environ[key]


def handle_secrets(no_auth=False):
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logger.info("HF_TOKEN is set")
    else:
        logger.warning(
            "HF_TOKEN is not set - this may cause issues accessing private models or models requiring authorization"
        )

    if no_auth:
        # Remove VLLM_API_KEY if present to disable authorization
        if "VLLM_API_KEY" in os.environ:
            del os.environ["VLLM_API_KEY"]
        logger.info(
            "--no-auth is set: requests to vLLM API will not require authorization. "
            "HTTP Authorization header will not be checked."
        )
        return

    # Check for VLLM_API_KEY first, then fall back to JWT_SECRET
    vllm_api_key = os.getenv("VLLM_API_KEY")
    if vllm_api_key:
        logger.info("VLLM_API_KEY is already set, using existing value")
        return

    # VLLM_API_KEY is not set, check if JWT_SECRET is available
    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        logger.warning(
            "Neither VLLM_API_KEY nor JWT_SECRET are set: HTTP requests to vLLM API will not require authorization"
        )
        return

    encoded_api_key = get_encoded_api_key(jwt_secret)
    if encoded_api_key is not None:
        os.environ["VLLM_API_KEY"] = encoded_api_key
        logger.info(
            "JWT_SECRET is set: HTTP requests to vLLM API require bearer token in 'Authorization' header. See docs for how to get bearer token."
        )


def runtime_settings(model_spec_json, no_auth=False):
    logger.info(f"using model: {model_spec_json['model_id']}")
    handle_secrets(no_auth=no_auth)

    # In multihost deployments, model weights are on shared storage and accessed
    # via model-specific environment variables (e.g., DEEPSEEK_V3_HF_MODEL).
    # Skip model_setup() which requires MODEL_WEIGHTS_DIR and creates symlinks.
    # TODO(tt-metal): Update DeepSeek model impl to use standard HF_MODEL env var
    # so we can reuse existing model setup and standard weight/cache mounting.
    if os.getenv("MULTIHOST_ROLE"):
        logger.info(
            "Multihost mode detected, skipping model_setup() - "
            "weights accessed via model-specific env vars on shared storage"
        )
        return

    # TODO: check HF repo access with HF_TOKEN supplied
    model_setup(model_spec_json)


def set_metal_timeout_env_vars():
    """Set tt-metal operation timeout env vars for automatic hang detection.

    When enabled (default), configures TT_METAL_OPERATION_TIMEOUT_SECONDS and
    TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE so that tt-triage runs
    automatically when an op dispatch hangs.

    Disabled when DISABLE_METAL_OP_TIMEOUT=1 is set (via run.py --disable-metal-timeout).
    """
    if os.getenv("DISABLE_METAL_OP_TIMEOUT") == "1":
        logger.info("Metal op timeout disabled via DISABLE_METAL_OP_TIMEOUT=1")
        return

    tt_metal_home = os.getenv("TT_METAL_HOME", "/home/container_app_user/tt-metal")
    python_env_dir = os.getenv("PYTHON_ENV_DIR", f"{tt_metal_home}/python_env")
    log_dir = os.getenv("TT_METAL_LOGS_PATH", "/home/container_app_user/logs")

    triage_new = Path(tt_metal_home) / "tools" / "triage" / "triage.py"
    triage_old = Path(tt_metal_home) / "scripts" / "debugging_scripts" / "triage.py"
    triage_script = str(triage_new if triage_new.exists() else triage_old)

    timeout_cmd = (
        f"{python_env_dir}/bin/python {triage_script} "
        f"--disable-progress > {log_dir}/tt-triage-$(date +%Y%m%d-%H%M%S).log 2>&1"
    )

    os.environ["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = "5.0"
    os.environ["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"] = timeout_cmd
    logger.info("Set TT_METAL_OPERATION_TIMEOUT_SECONDS=5.0")
    logger.info(f"Set TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE={timeout_cmd}")


def set_runtime_env_vars(model_spec_json):
    """Set runtime environment variables from model spec.

    Handles env_vars in two possible locations:
    1. Top level: model_spec_json["env_vars"] (from ModelSpec.__post_init__ merge)
    2. Nested: model_spec_json["device_model_spec"]["env_vars"] (raw JSON)

    Both locations are checked and merged, with top-level taking precedence.
    """
    env_vars = {}

    # Check nested location first (device_model_spec.env_vars)
    device_model_spec = model_spec_json.get("device_model_spec", {})
    if isinstance(device_model_spec, dict):
        nested_env_vars = device_model_spec.get("env_vars", {})
        if nested_env_vars:
            env_vars.update(nested_env_vars)

    # Check top-level location (takes precedence)
    top_level_env_vars = model_spec_json.get("env_vars", {})
    if top_level_env_vars:
        env_vars.update(top_level_env_vars)

    if not env_vars:
        logger.info("No env_vars found in model spec")
        return

    for key, value in env_vars.items():
        if not isinstance(key, str):
            key = str(key)
            logger.warning(
                f"env var key:={key} is not a string, converting to string: {key}"
            )
        if not isinstance(value, str):
            logger.warning(
                f"env var value:={value} is not a string, converting to string: {value}"
            )
            value = str(value)

        original_value = os.getenv(key)
        if original_value is not None:
            logger.warning(
                f"env var {key} is already set to {original_value}, overriding with {value}"
            )
        logger.info(f"setting env var: {key}={value}")
        os.environ[key] = value


def start_trace_capture(
    model_spec_json, service_port: int, disable_trace_capture: bool = False
):
    # Models with builtin warmup handle their own trace capture internally
    if model_spec_json.get("has_builtin_warmup", False):
        logger.info(
            "Model has builtin warmup (has_builtin_warmup=True), so no separate background PromptClient trace capture will be launched. Any trace capture must happen inside the model warmup path."
        )
        return

    if disable_trace_capture:
        logger.info("Trace capture is disabled via --disable-trace-capture")
        return

    supported_modalities = model_spec_json.get("supported_modalities", ["text"])

    # Get max_context from device_model_spec for trace calculation
    max_context = model_spec_json.get("device_model_spec", {}).get("max_context")
    if max_context is None:
        # Fallback to vllm_args if not in device_model_spec
        max_model_len_str = (
            model_spec_json.get("device_model_spec", {})
            .get("vllm_args", {})
            .get("max_model_len")
        )
        if max_model_len_str:
            max_context = int(max_model_len_str)

    logger.info("Starting background trace capture process...")
    trace_process = multiprocessing.Process(
        target=run_background_trace_capture,
        args=(
            model_spec_json["hf_model_repo"],
            service_port,
            supported_modalities,
            max_context,
        ),
        daemon=True,
        name="trace_capture",
    )
    trace_process.start()
    logger.info(
        f"Background trace capture process started (PID: {trace_process.pid}, "
        f"max_context: {max_context})"
    )


def _normalize_vllm_arg_name(arg_name: str) -> str:
    return arg_name.lstrip("-").split("=", 1)[0].replace("-", "_")


def _append_vllm_arg(argv: list[str], arg_name: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(arg_name)
        return
    argv.extend([arg_name, str(value)])


def _extract_cli_arg_value(argv: list[str], arg_name: str) -> Optional[str]:
    for index, token in enumerate(argv):
        if token == arg_name:
            if index + 1 < len(argv):
                return argv[index + 1]
            return None
        if token.startswith(f"{arg_name}="):
            return token.split("=", 1)[1]
    return None


def resolve_service_port() -> int:
    port_value = _extract_cli_arg_value(sys.argv[1:], "--port")
    if port_value is not None:
        return int(port_value)
    return int(DEFAULT_VLLM_SERVER_PORT)


def format_vllm_serve_command(argv) -> str:
    """Render the normalized argv as a multi-line bash command."""
    command_lines = ["vllm serve"]
    index = 1
    while index < len(argv):
        token = argv[index]
        rendered_tokens = [shlex.quote(token)]
        has_separate_value = (
            token.startswith("--")
            and "=" not in token
            and index + 1 < len(argv)
            and not argv[index + 1].startswith("--")
        )
        if has_separate_value:
            rendered_tokens.append(shlex.quote(argv[index + 1]))
            index += 1

        command_lines.append(" ".join(rendered_tokens))
        index += 1

    return " \\\n  ".join(command_lines)


def set_vllm_sys_argv(args, remaining_sys_argv, default_vllm_args):
    # runpy uses sys.argv, rebuild it with the merged vLLM args.
    vllm_argv = [sys.argv[0]]
    remaining_default_vllm_args = dict(default_vllm_args)
    default_arg_name_by_normalized_name = {
        _normalize_vllm_arg_name(arg_name): arg_name
        for arg_name in remaining_default_vllm_args
    }
    input_vllm_argv = list(remaining_sys_argv)
    if args.service_port is not None:
        already_set_port = _extract_cli_arg_value(input_vllm_argv, "--port")
        if already_set_port is not None:
            logger.warning(
                f"vLLM server --port={already_set_port} already set direcly, ignoring --service-port={args.service_port}"
            )
        else:
            # Remap wrapper --service-port to vLLM's --port.
            input_vllm_argv.extend(["--port", str(args.service_port)])

    index = 0
    while index < len(input_vllm_argv):
        token = input_vllm_argv[index]
        if not token.startswith("--"):
            vllm_argv.append(token)
            index += 1
            continue

        cli_arg_name, separator, inline_value = token.partition("=")
        overridden_default_arg_name = default_arg_name_by_normalized_name.pop(
            _normalize_vllm_arg_name(cli_arg_name), None
        )
        if overridden_default_arg_name is not None:
            remaining_default_vllm_args.pop(overridden_default_arg_name, None)

        if separator:
            vllm_argv.append(f"{cli_arg_name}={inline_value}")
            index += 1
            continue

        vllm_argv.append(cli_arg_name)
        next_token_is_value = index + 1 < len(input_vllm_argv) and not input_vllm_argv[
            index + 1
        ].startswith("--")
        if next_token_is_value:
            value = input_vllm_argv[index + 1]
            vllm_argv.append(value)
            index += 2
            continue

        index += 1

    for key, value in remaining_default_vllm_args.items():
        cli_arg_name = f"--{key}"
        _append_vllm_arg(vllm_argv, cli_arg_name, value)

    # finally set sys.argv to the vllm server args
    sys.argv = vllm_argv
    logger.info(f"vLLM command:\n{format_vllm_serve_command(sys.argv)}")


def main():
    # Step 1: Parse --model argument (if provided)
    args, remaining_sys_argv = parse_args()
    args.device = args.tt_device or args.device
    args.engine = normalize_engine_type(args.engine)

    # Step 2: Load model spec
    model_spec = load_model_spec(
        model_arg=args.model,
        device_arg=args.device,
        engine_arg=args.engine,
        impl_arg=args.impl,
    )
    device_type = model_spec.get("device_type")
    if device_type:
        device_type = normalize_device_type(device_type)
    elif args.device:
        device_type = normalize_device_type(args.device)

    if device_type and not os.getenv("TT_CACHE_PATH"):
        set_cache_paths(model_spec, device_type)
    # NOTE: In multihost deployments, model weights are expected to reside on shared
    # storage (e.g., NFS) and are read directly by each worker via model-specific
    # environment variables (e.g., DEEPSEEK_V3_HF_MODEL). Users are responsible for
    # downloading weights to a location on shared storage beforehand. Therefore,
    # automatic weight download is skipped when MULTIHOST_ROLE is set.
    if not os.getenv("MODEL_WEIGHTS_DIR") and not os.getenv("MULTIHOST_ROLE"):
        ensure_weights_available(model_spec)

    logger.info(f"Using model spec: {model_spec['model_id']}")

    # Step 3: Register TT models (after lookup, with correct impl_id)
    impl_id = model_spec.get("impl", {}).get("impl_id")
    register_tt_models(impl_id)

    # Step 4: Set runtime environment variables and vLLM server args
    set_metal_timeout_env_vars()
    set_runtime_env_vars(model_spec)
    runtime_settings(model_spec, no_auth=args.no_auth)
    default_vllm_args = dict(model_spec["device_model_spec"]["vllm_args"])
    hf_model_path = os.getenv("HF_MODEL")
    if hf_model_path:
        default_vllm_args["model"] = hf_model_path
        default_vllm_args["tokenizer"] = hf_model_path
        default_vllm_args["served-model-name"] = model_spec["hf_model_repo"]
        logger.info(
            "Overriding vLLM model/tokenizer path to local HF_MODEL symlink: %s",
            hf_model_path,
        )
        logger.info(
            "Preserving served model name for API requests: %s",
            model_spec["hf_model_repo"],
        )
    set_vllm_sys_argv(args, remaining_sys_argv, default_vllm_args)

    # Step 5: Start trace capture if needed
    start_trace_capture(
        model_spec,
        service_port=resolve_service_port(),
        disable_trace_capture=args.disable_trace_capture,
    )

    # Step 6: Launch vLLM server
    # runpy uses the same process and environment so the registered models are available
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
