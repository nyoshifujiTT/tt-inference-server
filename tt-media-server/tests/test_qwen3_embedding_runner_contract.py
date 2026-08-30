# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The Qwen3-Embedding runner must ask the model for the finished embedding.

This runner owns no Pooler: it hands whatever the model returns straight to
``_process_result``, which slices ``result[:num_requests]`` as rows and serves
those as the embedding vectors. So it needs the model's finished, L2-normalized
``[batch, hidden]`` output.

The model wrapper also exposes the stage right before pooling -- the flat
``[total_tokens, hidden]`` layout vLLM's pooling runner indexes -- and both come
out of the same ``forward``. Calling ``forward`` here therefore leaves which
stage we get to a default rather than to this runner's requirement, and a flat
tensor read as rows is silently wrong rather than an error: the token axis lands
where the batch axis should be and every served vector is garbage.

Pin the named call so that cannot come back.
"""

import importlib.util
import pathlib
import types

import pytest


class _FakeTensor:
    """Minimal stand-in for a 2-D tensor.

    conftest replaces ``torch`` with a MagicMock so the suite can run without
    it, which means real tensors are unavailable and mock arithmetic answers
    every assertion with another mock. Carry the two properties these checks
    are about -- the row count and each row's L2 norm -- explicitly instead.
    """

    def __init__(self, rows, cols, value):
        self.shape = (rows, cols)
        self._rows = rows
        self._cols = cols
        self._value = value

    def row_norms(self):
        return [(self._cols * self._value**2) ** 0.5 for _ in range(self._rows)]


def _real_embedding_runner():
    """Load the real module, bypassing conftest's mock runner classes.

    conftest replaces ``tt_model_runners.embedding_runner`` with stand-ins so the
    suite can run without ttnn. Those stand-ins have no methods, so importing the
    name normally here would test nothing. Load the file itself instead, with the
    ttnn/torch-device imports it needs stubbed out.
    """
    path = pathlib.Path(__file__).resolve().parents[1] / "tt_model_runners" / "embedding_runner.py"
    spec = importlib.util.spec_from_file_location("_real_embedding_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubModel:
    """Stands in for the wrapper, recording which entry point was used."""

    def __init__(self):
        self.encode_calls = []
        self.forward_calls = []

    def encode(self, input_ids, attention_mask=None):
        self.encode_calls.append((input_ids, attention_mask))
        # Finished embeddings: one unit-norm row per request.
        return _FakeTensor(input_ids.shape[0], 8, 1 / (8**0.5))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.forward_calls.append((input_ids, attention_mask, kwargs))
        # The pre-pooling layout, which this runner must never receive: one row
        # per token rather than per request.
        rows, cols = input_ids.shape
        return _FakeTensor(rows * cols, 8, 1.0)


@pytest.fixture
def runner():
    cls = _real_embedding_runner().Qwen3Embedding8BRunner

    instance = cls.__new__(cls)
    instance.model = _StubModel()
    instance.max_model_len = 8192
    instance.model_name = "Qwen/Qwen3-Embedding-8B"
    instance.ttnn_device = types.SimpleNamespace()
    return instance


def _tokenized(num_requests, seq_len):
    return {
        "input_ids": _FakeTensor(num_requests, seq_len, 0.0),
        "attention_mask": _FakeTensor(num_requests, seq_len, 1.0),
    }


def test_embed_uses_the_finished_embedding_entry_point(runner):
    out = runner._embed(_tokenized(4, 16))

    assert runner.model.encode_calls, "the runner must ask for the finished embedding by name"
    assert not runner.model.forward_calls, (
        "forward() is the shared implementation behind both output stages; calling it "
        "leaves the layout to a default rather than to this runner's requirement"
    )
    # One row per request, not one row per token.
    assert out.shape[0] == 4


def test_embed_passes_the_attention_mask_through(runner):
    tokenized = _tokenized(2, 8)

    runner._embed(tokenized)

    (_, mask), = runner.model.encode_calls
    assert mask is tokenized["attention_mask"]


def test_served_vectors_are_unit_norm(runner):
    out = runner._embed(_tokenized(3, 8))

    assert all(abs(norm - 1.0) < 1e-5 for norm in out.row_norms()), (
        "Qwen3-Embedding's modules.json ends in a Normalize stage; the runner serves "
        "whatever the model returns, so it must return normalized vectors"
    )
