# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Catalog contract tests for the Qwen3-Embedding dev specs (p150x1).

Qwen3-Embedding ships an HF ``config.json`` whose ``architectures`` is
``["Qwen3ForCausalLM"]`` (plus a sentence-transformers ``1_Pooling`` module).
vLLM's ``--runner auto`` therefore resolves that architecture to the *generate*
runner, which would serve text completions instead of embeddings. The catalog
must pin ``runner: pooling`` so the model is served as an embedding model.
These tests lock that invariant into the dev catalog so a future edit cannot
silently drop it.
"""
from pathlib import Path

import pytest

from workflows.utils import get_repo_root_path
from workflows.model_spec import load_templates_from_yaml

EMBEDDING_YAML = get_repo_root_path() / "workflows" / "model_specs" / "dev" / "embedding.yaml"

QWEN3_EMBEDDING_P150 = ("Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-0.6B")


def _p150_specs_by_weight():
    """Return {weight: ModelSpec} for the P150 entry of each Qwen3-Embedding template."""
    out = {}
    for template in load_templates_from_yaml(EMBEDDING_YAML):
        for spec in template.expand_to_specs():
            if (
                spec.hf_model_repo in QWEN3_EMBEDDING_P150
                and spec.device_model_spec.device.name == "P150"
            ):
                out[spec.hf_model_repo] = spec
    return out


@pytest.mark.parametrize("weight", QWEN3_EMBEDDING_P150)
def test_qwen3_embedding_p150_runner_is_pooling(weight):
    specs = _p150_specs_by_weight()
    assert weight in specs, f"{weight} P150 spec missing from dev/embedding.yaml"
    vllm_args = specs[weight].device_model_spec.vllm_args
    assert vllm_args.get("runner") == "pooling", (
        f"{weight} P150 must pin runner=pooling (HF arch is Qwen3ForCausalLM, "
        f"which vLLM --runner auto would otherwise serve as a generate model); "
        f"got runner={vllm_args.get('runner')!r}"
    )
