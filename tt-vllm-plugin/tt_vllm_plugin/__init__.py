# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def register():
    # At first we used ttnn.get_device_ids() to truly understand if the TT platform is supported.
    # This caused the offline inference to hang and never complete, so for now we just assume that we always have TT support.
    return "tt_vllm_plugin.platform.TTPlatform"


def register_models():
    """Register custom models with ModelRegistry for online inference.

    This function is called automatically by vLLM when the plugin is loaded,
    ensuring models are registered before the API server or engine starts.
    """
    from vllm import ModelRegistry

    # Register TT Llama model
    ModelRegistry.register_model(
        "TTLlamaForCausalLM",
        "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM",
    )

    # Register BGE embedding model (TTBertModel)
    # This allows vLLM to find the TT-specific BGE implementation
    try:
        ModelRegistry.register_model(
            "TTBertModel",
            "models.demos.wormhole.bge_large_en.demo.generator_vllm:BGEForEmbedding",
        )
        print("Registered BGE embedding model")
    except Exception as e:
        # If registration fails (e.g., module not found), log warning but continue
        # This allows the plugin to work even if BGE model isn't available
        import logging

        logging.warning(
            f"Failed to register TTBertModel (BGE): {e}. "
            "BGE model may not be available. Ensure tt-metal is in Python path."
        )

    # Qwen3-Embedding is deliberately not registered here. Serving it goes
    # through tenstorrent/vllm-tt-plugin, whose pooling runner delegates to
    # model.pooler over the flat per-token hidden states, and the class that
    # meets that contract is
    # models.demos.qwen3_embedding.tt.generator_vllm:Qwen3EmbeddingForTTvLLM.
    #
    # The entries removed here named the model wrapper instead, whose forward
    # returns the finished embedding rather than the pre-pooling stage. This
    # plugin's runner consumed that output directly as the pooled result, a
    # contract vLLM itself does not have, so keeping the registration alive
    # would let a stale engine bind Qwen3-Embedding to the wrong caller
    # convention.

    # Add additional model registrations here as needed
    # ModelRegistry.register_model("AnotherModel", "path.to:ModelClass")
