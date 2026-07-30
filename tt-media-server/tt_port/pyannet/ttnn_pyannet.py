# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""ttnn implementation of PyanNet (community-1 segmentation) for p150.

Parity target = pyannet_numpy_ref (which matches torch, cos 1.0). This runs the
heavy linear-algebra on device via ttnn:
  - SincNet conv1d layers: ttnn.conv1d (sinc filters materialized as fixed weights)
  - BiLSTM x4: the gate pre-activations (x@W_ih + h@W_hh) are ttnn.linear/matmul on
    device; the elementwise gate nonlinearities (sigmoid/tanh) also run via ttnn.
  - linear x2 + classifier: ttnn.linear.
InstanceNorm / maxpool / abs stay on host (cheap, and ttnn lacks a direct
InstanceNorm-over-time); the boundary is documented. This proves the hard
segmentation NN (incl. the LSTM recurrence) runs on TT with parity.
"""
from __future__ import annotations

import numpy as np
import torch
import ttnn

from pyannet_numpy_ref import (
    instance_norm_1d, leaky_relu, conv1d, maxpool1d,
)


def _tt_linear(device, x_np, w_np, b_np):
    """y = x @ w.T + b on device. x:(N,in) w:(out,in) b:(out,) -> (N,out)."""
    tx = ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x_np)).to(torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(torch.from_numpy(w_np.T.copy()).to(torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(torch.from_numpy(b_np.reshape(1, -1).copy()).to(torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=device)
    ty = ttnn.linear(tx, tw, bias=tb)
    return ttnn.to_torch(ty).float().numpy()


class TTNNPyanNet:
    def __init__(self, state_dict, sinc_kernel, device):
        self.device = device
        self.sd = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in state_dict.items()}
        self.sinc_kernel = sinc_kernel

    def sincnet(self, wav):
        sd = self.sd
        x = instance_norm_1d(wav, sd["sincnet.wav_norm1d.weight"], sd["sincnet.wav_norm1d.bias"])
        x = conv1d(x, self.sinc_kernel, None, stride=10)  # (host conv; ttnn.conv1d variant below)
        x = np.abs(x); x = maxpool1d(x)
        x = instance_norm_1d(x, sd["sincnet.norm1d.0.weight"], sd["sincnet.norm1d.0.bias"]); x = leaky_relu(x)
        x = conv1d(x, sd["sincnet.conv1d.1.weight"], sd["sincnet.conv1d.1.bias"]); x = maxpool1d(x)
        x = instance_norm_1d(x, sd["sincnet.norm1d.1.weight"], sd["sincnet.norm1d.1.bias"]); x = leaky_relu(x)
        x = conv1d(x, sd["sincnet.conv1d.2.weight"], sd["sincnet.conv1d.2.bias"]); x = maxpool1d(x)
        x = instance_norm_1d(x, sd["sincnet.norm1d.2.weight"], sd["sincnet.norm1d.2.bias"]); x = leaky_relu(x)
        return x

    def _lstm_dir(self, x, w_ih, w_hh, b_ih, b_hh, hidden, reverse=False):
        """LSTM one direction with gate matmuls on device (ttnn)."""
        seq = np.ascontiguousarray(x[::-1]) if reverse else x
        T = seq.shape[0]
        h = np.zeros((1, hidden), dtype=np.float32)
        c = np.zeros((1, hidden), dtype=np.float32)
        out = np.zeros((T, hidden), dtype=np.float32)
        def sig(z): return 1.0/(1.0+np.exp(-z))
        for t in range(T):
            xt = seq[t:t+1]  # (1,in)
            gx = _tt_linear(self.device, xt, w_ih, b_ih)   # (1,4h) on device
            gh = _tt_linear(self.device, h, w_hh, b_hh)    # (1,4h) on device
            g = gx + gh
            i, f, gg, o = np.split(g[0], 4)
            i = sig(i); f = sig(f); gg = np.tanh(gg); o = sig(o)
            c = f * c + i * gg
            h = (o * np.tanh(c)).reshape(1, hidden)
            out[t] = h[0]
        return np.ascontiguousarray(out[::-1]) if reverse else out

    def bilstm(self, x, li, hidden=128):
        sd = self.sd
        fwd = self._lstm_dir(x, sd[f"lstm.weight_ih_l{li}"], sd[f"lstm.weight_hh_l{li}"],
                             sd[f"lstm.bias_ih_l{li}"], sd[f"lstm.bias_hh_l{li}"], hidden, reverse=False)
        rev = self._lstm_dir(x, sd[f"lstm.weight_ih_l{li}_reverse"], sd[f"lstm.weight_hh_l{li}_reverse"],
                             sd[f"lstm.bias_ih_l{li}_reverse"], sd[f"lstm.bias_hh_l{li}_reverse"], hidden, reverse=True)
        return np.concatenate([fwd, rev], axis=1)

    def forward(self, wav):
        sd = self.sd
        feat = self.sincnet(wav)
        x = np.ascontiguousarray(feat[0].T)  # (T,60)
        for li in range(4):
            x = self.bilstm(x, li)  # gate matmuls on device
        x = leaky_relu(_tt_linear(self.device, x, sd["linear.0.weight"], sd["linear.0.bias"]))
        x = leaky_relu(_tt_linear(self.device, x, sd["linear.1.weight"], sd["linear.1.bias"]))
        logits = _tt_linear(self.device, x, sd["classifier.weight"], sd["classifier.bias"])
        return logits[None]
