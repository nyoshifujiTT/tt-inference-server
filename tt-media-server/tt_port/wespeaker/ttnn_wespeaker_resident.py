# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Device-resident WeSpeaker ResNet34 embedding for p150 (fast path).

Keeps the activation ON DEVICE across the whole ResNet34: input uploaded once,
every conv/relu/residual-add runs on device consuming/producing ttnn tensors,
and only the final conv feature map is downloaded once. Statistics pooling
(TSTP, optionally *weighted* per speaker) and the seg_1 linear are left to the
caller (`backbone`) so the exact pyannote pooling incl. `weights` is preserved
bit-for-bit on host; the earlier per-layer host<->device transfers that
dominated TTNNWeSpeaker runtime are removed.

Parity target = torch WeSpeaker embedding (cos > 0.99). Weights bn-folded.
"""
from __future__ import annotations

import numpy as np
import torch
import ttnn

from wespeaker_numpy_ref import WeSpeakerNumpyRef


class TTNNWeSpeakerResident:
    BLOCKS = [3, 4, 6, 3]
    MIN_DEVICE_W = 8  # run tiny-time chunks on host to avoid conv2d auto-shard assert

    def __init__(self, state_dict, device):
        self.device = device
        self._ref = WeSpeakerNumpyRef(state_dict)
        self.folded = self._ref.folded
        self.seg_w = self._ref.seg_w
        self.seg_b = self._ref.seg_b
        self._wcache = {}
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        self.conv_cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)

    def _w(self, key, arr):
        t = self._wcache.get(key)
        if t is None:
            t = ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16))
            self._wcache[key] = t
        return t

    def _conv_dev(self, tx, wb, key, B, H, W, stride, pad):
        w, b = wb
        Cout, Cin, kh, kw = w.shape
        tw = self._w(key + ".w", w)
        tb = self._w(key + ".b", b.reshape(1, 1, 1, Cout))
        out, (Hout, Wout) = ttnn.conv2d(
            input_tensor=tx, weight_tensor=tw, bias_tensor=tb, device=self.device,
            in_channels=Cin, out_channels=Cout, batch_size=B,
            input_height=H, input_width=W, kernel_size=(kh, kw),
            stride=(stride, stride), padding=(pad, pad),
            conv_config=self.conv_cfg, compute_config=self.compute_cfg,
            return_output_dim=True)
        return out, Hout, Wout, Cout

    def backbone(self, feats_nchw):
        """feats_nchw: torch (B,1,80,T) -> conv feature map torch (B,C,H,W).

        Runs the full ResNet34 conv stack device-resident (no per-layer host
        round-trips) and downloads the final feature map once. Statistics
        pooling and the seg_1 linear are left to the caller so the exact
        pyannote pooling (including per-speaker `weights`) is preserved on host.
        """
        B, _, H, W = feats_nchw.shape
        # Degenerate short chunks (tiny time width) make ttnn conv2d auto-shard
        # estimate a reader-index page that trips a TT_FATAL and forces an
        # internal fallback. Those chunks are a negligible fraction of runtime,
        # so run them on host (numpy, bn-folded) for an identical result with no
        # device assert. Threshold is one tile of time (32) after the first two
        # stride-2 stages have to still leave >=1 column.
        if W < self.MIN_DEVICE_W:
            import numpy as _np
            xt = self._ref.backbone_numpy(feats_nchw.detach().cpu().numpy())
            return torch.from_numpy(_np.ascontiguousarray(xt)).float()
        x_nhwc = feats_nchw.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, 1)
        x = ttnn.from_torch(x_nhwc.to(torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        x, H, W, C = self._conv_dev(x, self.folded["conv1"], "conv1", B, H, W, 1, 1)
        x = ttnn.relu(x)
        for li, nb in enumerate(self.BLOCKS, start=1):
            for bi in range(nb):
                stride = 2 if (bi == 0 and li > 1) else 1
                p = f"resnet.layer{li}.{bi}"
                out, H1, W1, C1 = self._conv_dev(x, self.folded[f"{p}.c1"], f"{p}.c1", B, H, W, stride, 1)
                out = ttnn.relu(out)
                out, H2, W2, C2 = self._conv_dev(out, self.folded[f"{p}.c2"], f"{p}.c2", B, H1, W1, 1, 1)
                if f"{p}.ds" in self.folded:
                    idd, Hd, Wd, Cd = self._conv_dev(x, self.folded[f"{p}.ds"], f"{p}.ds", B, H, W, stride, 0)
                else:
                    idd = x
                x = ttnn.relu(ttnn.add(out, idd))
                H, W, C = H2, W2, C2
        # download once: NHWC flattened (1,1,B*H*W,C) -> (B,C,H=freq,W=time)
        xt = ttnn.to_torch(x).float().reshape(B, H, W, C).permute(0, 3, 1, 2)
        return xt

    def forward(self, feats_nchw):
        """feats_nchw: torch (B,1,80,T) -> (B,256). Unweighted TSTP (parity path)."""
        xt = self.backbone(feats_nchw)  # (B,C,H,W)
        B, C, H, W = xt.shape
        x2 = xt.reshape(B, C * H, W)
        mean = x2.mean(dim=2)
        std = x2.std(dim=2)
        pooled = torch.cat([mean, std], dim=1)  # (B, 5120)
        emb = pooled @ torch.from_numpy(self.seg_w).float().T + torch.from_numpy(self.seg_b).float()
        return emb
