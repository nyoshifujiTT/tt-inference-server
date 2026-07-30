"""On-device (p150) parity: TTNN WeSpeaker embedding vs torch reference emb."""
import sys, numpy as np, torch, ttnn
sys.path.insert(0, "/home/ubuntu/diar-work/tt-inference-server/tt-media-server/tt_port/wespeaker")

d = np.load("/home/ubuntu/diar-work/tt_port_wespeaker/parity_input.npz")
feats = torch.from_numpy(d["feats"]).float()
emb_torch = d["emb_torch"]
sd_npz = np.load("/home/ubuntu/diar-work/tt_port_wespeaker/state_dict.npz")
# WeSpeakerNumpyRef expects objects with .numpy(); wrap numpy arrays as torch tensors
state_dict = {k: torch.from_numpy(sd_npz[k]) for k in sd_npz.files}

from ttnn_wespeaker import TTNNWeSpeaker

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    model = TTNNWeSpeaker(state_dict, dev)
    emb_tt = model.forward(feats).numpy()
finally:
    ttnn.close_device(dev)

cos = float(np.dot(emb_tt[0], emb_torch[0]) / (np.linalg.norm(emb_tt[0]) * np.linalg.norm(emb_torch[0])))
maxabs = float(np.max(np.abs(emb_tt - emb_torch)))
print(f"PARITY cos={cos:.5f} max_abs={maxabs:.4f}")
assert cos > 0.99, f"cosine too low {cos}"
print("PASS: ttnn WeSpeaker embedding matches torch reference on p150")
