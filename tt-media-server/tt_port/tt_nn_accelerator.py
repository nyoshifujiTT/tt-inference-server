# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Tenstorrent p150 NN accelerator hook for community-1 diarization.

`make_tt_accelerator(device)` returns a callable `(pipeline) -> None` suitable for
`DiarizationBackend(nn_accelerator=...)`. It patches the pipeline's two neural
nets to run on the p150 via ttnn:
  - segmentation PyanNet (SincNet + BiLSTM x4)  -> TTNNPyanNet
  - embedding WeSpeaker ResNet34 backbone       -> TTNNWeSpeaker

Verified end-to-end (tt_full_diarization.py): real community-1 diarization with
both NNs on p150 yields the CPU-consistent 2-speaker result. ttnn/torch are
imported lazily so this module is importable without them (for wiring/tests).
"""
from __future__ import annotations

import os
import sys

_WESPEAKER = os.path.join(os.path.dirname(__file__), "wespeaker")
_PYANNET = os.path.join(os.path.dirname(__file__), "pyannet")


def make_tt_accelerator(device):
    """Return a (pipeline)->None hook that offloads both NNs onto `device` (ttnn)."""
    import torch
    import torch.nn.functional as F

    for pth in (_WESPEAKER, _PYANNET):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    from ttnn_wespeaker import TTNNWeSpeaker
    from ttnn_pyannet import TTNNPyanNet

    def _apply(pipeline) -> None:
        # ---- segmentation on TT ----
        seg_model = pipeline._segmentation.model
        sinc = seg_model.sincnet.conv1d[0].filterbank.filters().detach().numpy()
        tt_seg = TTNNPyanNet(seg_model.state_dict(), sinc, device)

        def seg_forward(waveforms):
            outs = []
            for i in range(waveforms.shape[0]):
                logits = tt_seg.forward(waveforms[i:i + 1].detach().numpy())
                outs.append(torch.from_numpy(logits[0]).float())
            return F.log_softmax(torch.stack(outs, 0), dim=-1)

        seg_model.forward = seg_forward

        # ---- embedding backbone on TT ----
        wespeaker = pipeline._embedding.model_
        resnet = wespeaker.resnet
        tt_emb = TTNNWeSpeaker(wespeaker.state_dict(), device)
        tt_emb.use_device_elementwise = True

        def _bb_one(feats1):
            x = tt_emb._relu_dev(tt_emb._conv(feats1, tt_emb.folded["conv1"], 1))
            for li, nb in enumerate(tt_emb.BLOCKS, start=1):
                for bi in range(nb):
                    st = 2 if (bi == 0 and li > 1) else 1
                    x = tt_emb._block(x, f"resnet.layer{li}.{bi}", st)
            return x

        def resnet_forward(fbank, weights=None):
            feats = fbank.permute(0, 2, 1).unsqueeze(1).float()
            outs = [_bb_one(feats[i:i + 1]) for i in range(feats.shape[0])]
            minT = min(o.shape[-1] for o in outs)
            x = torch.cat([o[..., :minT] for o in outs], dim=0)
            stats = resnet.pool(x, weights=weights)
            return torch.tensor(0.0), resnet.seg_1(stats)

        resnet.forward = resnet_forward

    return _apply
