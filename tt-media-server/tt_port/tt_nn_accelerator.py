# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Tenstorrent p150 NN accelerator hook for community-1 diarization.

`make_tt_accelerator(device)` returns a callable `(pipeline) -> None` suitable for
`DiarizationBackend(nn_accelerator=...)`. It offloads the embedding backbone
(WeSpeaker ResNet34) onto the p150 via a *device-resident* ttnn implementation
(`TTNNWeSpeakerResident`): the activation stays on device across the whole
ResNet34 (input uploaded once, every conv/relu/residual-add consumes/produces
ttnn tensors, only the final conv feature map is downloaded). This removes the
per-layer host<->device transfer that dominated the earlier `TTNNWeSpeaker`
path and makes the embedding stage ~2x faster than 32-thread CPU (measured:
63 chunks 4.9s on p150 vs 9.2s CPU) while preserving parity.

Crucially the temporal statistics pooling (TSTP) and the seg_1 linear stay in
the original pyannote modules on host, so the *weighted* per-speaker pooling
that real diarization relies on (`resnet.pool(out, weights=...)`) is preserved
bit-for-bit. Only the ResNet34 conv stack (the expensive part) runs on device.

Segmentation (PyanNet SincNet+BiLSTM) stays on CPU by default: it is cheap
(~0.36s for a 30s clip) and the ttnn segmentation path is slower per window, so
offloading it would regress total latency. Set env DIARIZATION_TT_SEGMENTATION=1
to also run segmentation on the p150 (full on-device, slower but useful for
device-coverage validation).

The ttnn implementations themselves live in tt-metal under
``models.demos.audio.pyannote_diarization`` (the community-standard home for a
model's ttnn code, next to the whisper / qwen3_asr demos); this module is only
the thin tt-inference-server-side adapter that wires them into pyannote. ttnn /
torch / the metal package are imported lazily so this module stays importable
(and unit-testable) without a device or a tt-metal checkout on PYTHONPATH.
"""

from __future__ import annotations

import os


def make_tt_accelerator(device):
    """Return a (pipeline)->None hook that offloads the NN(s) onto `device` (ttnn)."""
    import torch
    import torch.nn.functional as F

    from models.demos.audio.pyannote_diarization.tt.ttnn_wespeaker_resident import (
        TTNNWeSpeakerResident,
    )

    tt_seg_enabled = os.environ.get("DIARIZATION_TT_SEGMENTATION", "0") == "1"

    def _apply(pipeline) -> None:
        # ---- embedding backbone (ResNet34 conv stack) on TT, device-resident ----
        wespeaker = pipeline._embedding.model_
        resnet = wespeaker.resnet
        tt_emb = TTNNWeSpeakerResident(wespeaker.state_dict(), device)

        def resnet_forward(fbank, weights=None):
            # fbank: (B, T, 80) log-mel -> (B, 1, 80, T)
            feats = fbank.permute(0, 2, 1).unsqueeze(1).float()
            outs = [tt_emb.backbone(feats[i : i + 1]) for i in range(feats.shape[0])]
            # (B, C, H=freq, W=time); align time dim across chunks (conv rounding)
            minW = min(o.shape[-1] for o in outs)
            x = torch.cat([o[..., :minW] for o in outs], dim=0)
            # original pyannote pooling honours per-speaker `weights`
            stats = resnet.pool(x, weights=weights)
            return torch.tensor(0.0), resnet.seg_1(stats)

        resnet.forward = resnet_forward

        # ---- segmentation on TT (optional, off by default) ----
        if tt_seg_enabled:
            from models.demos.audio.pyannote_diarization.tt.ttnn_pyannet import (
                TTNNPyanNet,
            )

            seg_model = pipeline._segmentation.model
            sinc = seg_model.sincnet.conv1d[0].filterbank.filters().detach().numpy()
            tt_seg = TTNNPyanNet(seg_model.state_dict(), sinc, device)

            def seg_forward(waveforms):
                # Batch every sliding window through one device-resident BiLSTM
                # pass (weights/state stay on device, no per-timestep transfer).
                wav_batch = waveforms.detach().cpu().numpy()
                if wav_batch.ndim == 2:
                    wav_batch = wav_batch[:, None, :]
                logits = tt_seg.forward_batch(wav_batch)  # (B,T,7)
                return F.log_softmax(torch.from_numpy(logits).float(), dim=-1)

            seg_model.forward = seg_forward

    return _apply
