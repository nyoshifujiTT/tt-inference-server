# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""On-device parity for the narrow-width (pad+crop) conv path of the resident
WeSpeaker backbone.

The last chunks of a recording produce very small conv input time-widths (W as
low as 1). The default ttnn.conv2d auto-shard estimate trips a reader-index CB
assert for those (a known ttnn bug, tt-metal #35207 / #43193), so
``TTNNWeSpeakerResident._conv_dev`` zero-pads the time axis up to ``SAFE_W`` on
device, runs the conv, and crops the output back. This test asserts that path is
numerically faithful to the numpy reference backbone for degenerate widths and
that the whole backbone still runs on the p150.

Skipped automatically when ttnn / a Tenstorrent device / the fixtures are
unavailable, so it is safe in the media-server suite.
"""
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("ttnn")

EMB_NPZ = "/home/ubuntu/diar-work/emb_all.npz"
if not os.path.exists(EMB_NPZ):
    pytest.skip("resident parity fixtures not present", allow_module_level=True)

import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from wespeaker_numpy_ref import WeSpeakerNumpyRef  # noqa: E402
from ttnn_wespeaker_resident import TTNNWeSpeakerResident  # noqa: E402


@pytest.fixture(scope="module")
def device():
    try:
        dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no Tenstorrent device available: {exc}")
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def state_dict():
    dd = np.load(EMB_NPZ)
    return {k[4:]: torch.from_numpy(dd[k]) for k in dd.files if k.startswith("sd::")}


@pytest.mark.parametrize("W", [1, 2, 4, 8, 12])
def test_resident_narrow_matches_numpy_backbone(device, state_dict, W):
    sdn = {k: v.numpy() for k, v in state_dict.items()}
    ref = WeSpeakerNumpyRef(sdn)
    tt = TTNNWeSpeakerResident(state_dict, device)

    feat = np.random.RandomState(W).randn(1, 1, 80, W).astype(np.float32)
    ref_map = ref.backbone_numpy(feat)                       # (1,C,H,Wref)
    dev_map = tt.backbone(torch.from_numpy(feat).float()).numpy()

    # the pad+crop path must reproduce the exact unpadded output width
    assert dev_map.shape == ref_map.shape, (dev_map.shape, ref_map.shape)

    a = dev_map.flatten()
    b = ref_map.flatten()
    cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos > 0.99, f"narrow-width backbone parity too low for W={W}: cos={cos}"
