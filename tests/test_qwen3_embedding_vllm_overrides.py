# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Qwen3-Embedding's stake in the --vllm-override-args rendering.

``tests/test_setup_host.py`` already covers the renderer generically.  What is
specific to this model is *which* overrides we depend on and how a bad
rendering shows up here:

* ``served-model-name`` takes ``nargs='+'``.  Before the list-expansion fix a
  list was emitted as one ``json.dumps`` token, so vLLM registered a model
  literally named ``["Qwen/Qwen3-Embedding-0.6B", ...]`` and every
  ``/v1/embeddings`` request naming the real model 404'd.  That is an
  end-to-end serving break for an embedding deployment, not a cosmetic issue.
* ``override-pooler-config`` is a *dict*, and it is how the OpenAI-API
  ``normalize`` behaviour is pinned at server start.  It must stay a single
  JSON token, because the flag is ``nargs=None``; expanding it would corrupt
  the pooling contract our device pooler honours.

Both directions are asserted so a future "simplification" of the renderer
cannot quietly break one while fixing the other.
"""
from workflows.run_docker_server import _vllm_override_cli_args

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def test_served_model_name_list_becomes_separate_argv_tokens():
    """Each alias must be its own token, or the served name is a JSON blob."""
    rendered = _vllm_override_cli_args(
        '{"served-model-name": ["%s", "qwen3-embedding"]}' % MODEL
    )
    assert rendered == [
        "--served-model-name",
        MODEL,
        "qwen3-embedding",
    ]
    assert not any(
        token.startswith("[") for token in rendered
    ), f"a list value was collapsed into a single JSON token: {rendered!r}"


def test_pooler_config_stays_one_json_token():
    """override-pooler-config is nargs=None: expanding it would break pooling."""
    rendered = _vllm_override_cli_args('{"override-pooler-config": {"normalize": false}}')
    assert rendered == ["--override-pooler-config", '{"normalize": false}']
    assert len(rendered) == 2, (
        "a dict override was expanded into multiple tokens; --override-pooler-config "
        f"takes a single JSON value: {rendered!r}"
    )


def test_embedding_overrides_render_together():
    """The combination we actually deploy with must round-trip."""
    rendered = _vllm_override_cli_args(
        '{"served-model-name": ["%s"], "override-pooler-config": {"normalize": true},'
        ' "max_model_len": 8192}' % MODEL
    )
    assert rendered[:2] == ["--served-model-name", MODEL]
    assert "--override-pooler-config" in rendered
    assert rendered[rendered.index("--override-pooler-config") + 1] == '{"normalize": true}'
    assert rendered[rendered.index("--max-model-len") + 1] == "8192"
