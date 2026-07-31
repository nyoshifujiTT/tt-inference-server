"""On-device (p150) parity: TTNN PyanNet segmentation logits vs torch reference."""
import sys, numpy as np, torch, ttnn
sys.path.insert(0,"/home/ubuntu/diar-work/tt-inference-server/tt-media-server/tt_port/pyannet")
d = np.load("/home/ubuntu/diar-work/tt_port_pyannet/seg_parity.npz")
wav = d["wav"]; sinc = d["sinc"]; logits_torch = d["logits_torch"]
state_dict = {k[4:]: torch.from_numpy(d[k]) for k in d.files if k.startswith("sd::")}
from ttnn_pyannet import TTNNPyanNet
dev = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    net = TTNNPyanNet(state_dict, sinc, dev)
    net.use_device_sincnet = True
    logits_tt = net.forward(wav)
finally:
    ttnn.close_device(dev)
T=min(logits_torch.shape[1], logits_tt.shape[1])
a=logits_torch[0,:T]; b=logits_tt[0,:T]
cos=float(np.dot(a.flatten(),b.flatten())/(np.linalg.norm(a)*np.linalg.norm(b)))
maxabs=float(np.max(np.abs(a-b)))
print(f"SEG_PARITY cos={cos:.5f} max_abs={maxabs:.4f} frames={T}")
# also compare argmax (powerset class per frame) agreement
agree=float((a.argmax(1)==b.argmax(1)).mean())
print(f"argmax_agreement={agree:.4f}")
assert cos>0.99, f"cosine too low {cos}"
print("PASS: ttnn PyanNet segmentation (incl. BiLSTM on device) matches torch on p150")
