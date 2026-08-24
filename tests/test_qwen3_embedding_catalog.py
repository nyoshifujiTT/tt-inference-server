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
from workflows.workflow_types import InferenceEngine

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


# ---------------------------------------------------------------------------
# trace_region_size: batched (B>=4) embedding prefill on a single p150 builds a
# mesh trace larger than tt-metal's 50_000_000 B default, which otherwise fails
# with a mesh_trace TT_FATAL (EngineDead). Only the 0.6B P150 entry is covered
# here: it is the size we brought up and swept B=1..32 on p150x1. The 8B batched
# requirement was not measured on this hardware, so we intentionally do not
# assert (or set) a value for it (same "don't ship what we haven't measured"
# policy as the device list).
# ---------------------------------------------------------------------------
import json

# Measured on p150x1: B=4 batched prefill needs 53,780,480 B of trace buffers.
B4_TRACE_BYTES = 53_780_480


def test_qwen3_embedding_06b_p150_trace_region_size_admits_batched_prefill():
    specs = _p150_specs_by_weight()
    weight = "Qwen/Qwen3-Embedding-0.6B"
    assert weight in specs, f"{weight} P150 spec missing from dev/embedding.yaml"
    dms = specs[weight].device_model_spec
    trs = dms.override_tt_config.get("trace_region_size")
    assert trs is not None, (
        f"{weight} P150 must set override_tt_config.trace_region_size; batched "
        f"(B>=4) embedding prefill needs >50MB of trace buffers and the tt "
        f"default (50_000_000) triggers a mesh_trace TT_FATAL EngineDead."
    )
    assert int(trs) >= B4_TRACE_BYTES, (
        f"{weight} P150 trace_region_size={trs} is below the measured B=4 "
        f"batched-prefill trace requirement ({B4_TRACE_BYTES} B)."
    )
    # The value must surface in the serialized additional_config.tt that vLLM reads.
    add_cfg = json.loads(dms.vllm_args["additional_config"])
    assert int(add_cfg["tt"]["trace_region_size"]) == int(trs)


@pytest.mark.parametrize("weight", QWEN3_EMBEDDING_P150)
def test_qwen3_embedding_p150_inference_engine_is_vllm(weight):
    """The P150 entries are served by vLLM, so they must declare it.

    These specs run the model through vLLM's pooling runner (hence
    ``runner: pooling`` above), but they used to be declared as
    ``inference_engine: MEDIA``. That mismatch is not cosmetic:
    run_docker_server only appends ``--model`` / ``--tt-device`` to the container
    command for VLLM specs, so a MEDIA-declared spec starts the vLLM entrypoint
    with neither, and it exits with "the following arguments are required:
    --tt-device". MEDIA also changes the published-port mapping and suppresses
    the vLLM default env vars. Pin the declaration so the transport matches the
    runner.
    """
    specs = _p150_specs_by_weight()
    assert weight in specs, f"{weight} P150 spec missing from dev/embedding.yaml"
    assert specs[weight].inference_engine == InferenceEngine.VLLM.value, (
        f"{weight} P150 is served through the vLLM pooling runner, so its "
        f"inference_engine must be VLLM (got {specs[weight].inference_engine!r}); "
        "otherwise the container is started without --model/--tt-device."
    )
