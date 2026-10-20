# bge-reranker-v2-m3 EXTRA_MODELS_DIR bundle

This bundle registers the Tenstorrent bge-reranker-v2-m3 cross-encoder with the
standalone `vllm-tt-plugin` (upstream vLLM 0.24) via `EXTRA_MODELS_DIR`, without
editing plugin source or using runtime monkey-patching (`sitecustomize`).

At startup the plugin scans `EXTRA_MODELS_DIR`, reads each
`<name>/vllm_metadata.json`, and registers `TT<arch> -> main_class`. Here that
maps `TTXLMRobertaForSequenceClassification` to the reranker adapter class.

`main_class` resolves against the tt-metal tree (mounted via `--tt-metal-home`
and on `PYTHONPATH`), so only the metadata lives here; no model code is copied.

This is the canonical (new-path) registration mechanism. The legacy (fork) path
registered the same arch at runtime through
`vllm-tt-metal/src/sitecustomize.py`; the plugin path replaces that.
