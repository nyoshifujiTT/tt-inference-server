# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import shutil

import pytest

from scripts.release.promote_dev_spec_to_prod import promote
from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.utils import get_repo_root_path
from workflows.workflow_types import ModelStatusTypes

MTP_CLASS = "models.demos.blackhole.qwen36.tt.qwen36_vllm:Qwen36MTPContractForCausalLM"
# Qwen36MTPContractForCausalLM verifies max_num_seqs x (K+1) rows in one 32-row tile.
VERIFY_TILE_ROWS = 32
SUPPORTED_K = (3, 7, 11)
# Qwen3.5-27B has the same structure and MTP head as Qwen3.6-27B.
MODELS = ["Qwen/Qwen3.6-27B", "Qwen/Qwen3.5-27B"]


def _specs(model):
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
    )
    return [
        spec
        for template in templates
        if template.weights == [model]
        for spec in template.expand_to_specs()
    ]


def _resolve(model, impl=None):
    return resolve_model_spec(
        _specs(model),
        model=model,
        device="p150x4",
        impl=impl,
        catalog_name="Shield",
    )


@pytest.mark.parametrize("model", MODELS)
def test_mtp_profile_is_opt_in(model):
    default = _resolve(model)
    mtp = _resolve(model, impl="qwen36-blackhole-mtp")

    assert default.impl.impl_name == "qwen36-blackhole"
    assert mtp.impl.impl_name == "qwen36-blackhole-mtp"
    assert mtp.model_id != default.model_id
    assert mtp.device_model_spec.default_impl is False
    assert mtp.status == ModelStatusTypes.EXPERIMENTAL
    assert mtp.supported_modalities == ["text"]


@pytest.mark.parametrize("model", MODELS)
def test_mtp_profile_selects_the_contract_class_and_custom_class_drafts(model):
    mtp = _resolve(model, impl="qwen36-blackhole-mtp")
    env = mtp.device_model_spec.env_vars
    args = mtp.device_model_spec.vllm_args

    # The plugin resolves the TT-prefixed architecture of the checkpoint.
    assert env["TT_MODEL_CLASS_OVERRIDES"] == (
        f"TTQwen3_5ForConditionalGeneration={MTP_CLASS}"
    )
    assert env["TT_QWEN35_TEXT_VER"] == "qwen36_blackhole"
    assert env["HF_MODEL"] == model
    assert env["MESH_DEVICE"] == "P150x4"

    speculative = json.loads(args["speculative_config"])
    assert speculative == {
        "method": "custom_class",
        "model": "vllm_tt_plugin.model_owned_drafter",
        "num_speculative_tokens": int(env["QWEN36_SPEC_K"]),
    }
    # The MTP class samples on the host.
    assert "sample_on_device_mode" not in mtp.device_model_spec.override_tt_config


@pytest.mark.parametrize("model", MODELS)
def test_mtp_profile_fits_the_verify_tile_and_page_tables(model):
    mtp = _resolve(model, impl="qwen36-blackhole-mtp")
    args = mtp.device_model_spec.vllm_args
    k = int(mtp.device_model_spec.env_vars["QWEN36_SPEC_K"])

    assert k in SUPPORTED_K
    assert int(args["max_num_seqs"]) == mtp.device_model_spec.max_concurrency == 8
    assert int(args["max_num_seqs"]) * (k + 1) <= VERIFY_TILE_ROWS
    assert args["block_size"] == "64"


def test_promoting_the_plain_profile_carries_no_mtp_text(tmp_path):
    specs_dir = get_repo_root_path() / "workflows" / "model_specs"
    prod_dir = tmp_path / "prod"
    shutil.copytree(specs_dir / "prod", prod_dir)
    ci_path = tmp_path / "models-ci-config.json"
    ci_path.write_text(
        json.dumps(
            {
                "models": {
                    model: {
                        "inference_engine": "vLLM",
                        "ci": {"release": {"devices": ["P150X4"]}},
                    }
                    for model in MODELS
                }
            }
        )
    )

    report = promote(
        ci_path,
        specs_dir / "dev",
        prod_dir,
        tt_metal_commit="new-metal",
        version="9.9.9",
        vllm_commit="new-vllm",
    )

    assert {identity[3] for identity in report["requested_identities"]} == {
        "qwen36_blackhole"
    }
    promoted = (prod_dir / "llm.yaml").read_text()
    for mtp_text in (
        "qwen36_blackhole_mtp",
        "qwen36-blackhole-mtp",
        "Qwen36MTPContractForCausalLM",
    ):
        assert mtp_text not in promoted
