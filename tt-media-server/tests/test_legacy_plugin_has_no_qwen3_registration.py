# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The in-repo legacy plugin must not bind Qwen3-Embedding any more.

Serving Qwen3-Embedding goes through tenstorrent/vllm-tt-plugin, whose pooling
runner delegates to ``model.pooler`` over the flat per-token hidden states. The
class that meets that contract is ``Qwen3EmbeddingForTTvLLM``.

This plugin used to register the arch names against the model wrapper, whose
forward returns the finished embedding instead -- it consumed that output
directly as the pooled result, a convention vLLM does not have. A leftover
registration would let a stale engine bind the model to the wrong caller
convention, and the failure mode is wrong numbers rather than an error.
"""

import pathlib


def _plugin_source():
    path = pathlib.Path(__file__).resolve().parents[2] / "tt-vllm-plugin" / "tt_vllm_plugin" / "__init__.py"
    return path.read_text()


def test_no_qwen3_arch_is_registered_here():
    source = _plugin_source()

    for arch in ("TTQwen3Model", "TTQwen3ForCausalLM"):
        assert f'"{arch}",' not in source, (
            f"{arch} is registered in the legacy in-repo plugin; Qwen3-Embedding "
            "is served through vllm-tt-plugin's pooling runner instead"
        )


def test_the_model_wrapper_is_not_named_as_a_vllm_entry_point():
    source = _plugin_source()

    assert "qwen3_embedding_8b.demo.generator_vllm" not in source, (
        "the model wrapper returns the finished embedding, not the pre-pooling "
        "layout a vLLM pooling runner indexes; it must not be a registered arch"
    )


def test_the_other_registrations_are_untouched():
    source = _plugin_source()

    # Removing the Qwen3 entries must not have disturbed its neighbours.
    assert '"TTLlamaForCausalLM"' in source
    assert '"TTBertModel"' in source
