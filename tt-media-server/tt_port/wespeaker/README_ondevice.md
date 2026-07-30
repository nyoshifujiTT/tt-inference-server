# On-device (p150) ttnn WeSpeaker embedding — how to run

Verified: full WeSpeakerResNet34 embedding (stem + 16 BasicBlocks + TSTP + seg_1)
runs on Blackhole p150 via ttnn, matching the torch reference with
**cosine 0.99971 / max_abs 0.0032** (HiFi4 math + fp32 dest accumulation).

## Files
- ttnn_wespeaker.py   : TTNNWeSpeaker (conv2d on device; bn folded on host;
  residual/relu/TSTP on host in this first milestone; conv is the dominant cost).
- run_ttnn_parity.py  : loads a saved fbank + torch embedding + state_dict and
  runs the ttnn forward on device, asserting cosine > 0.99.
- wespeaker_numpy_ref.py / golden_embeddings.json / PORTING_PLAN.md : step-1
  device-independent reference and plan.

## Environment (this host, reconstructed)
- OpenMPI-ULFM: `dpkg -i` from
  github.com/tenstorrent/ompi/releases/download/v5.0.7/openmpi-ulfm_5.0.7-1_amd64.deb
  -> /opt/openmpi-v5.0.7-ulfm
- ttnn: reuse built tree /data/wt/qwen3asr (build_Release/lib, ttnn, tools) +
  its site-packages (torch 2.11+cpu, tracy).
- device: `sudo chmod 666 /dev/tenstorrent/3` (UMD needs read/write); open with
  `ttnn.open_device(device_id=0, l1_small_size=32768)`.

## Run
```
SP=/data/wt/qwen3asr/python_env/lib/python3.12/site-packages
export TT_METAL_HOME=/data/wt/qwen3asr \
  PYTHONPATH=$SP:/data/wt/qwen3asr:/data/wt/qwen3asr/ttnn:/data/wt/qwen3asr/tools \
  LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:/data/wt/qwen3asr/build_Release/lib \
  ARCH_NAME=blackhole
/home/ubuntu/diar-work/ttnnvenv/bin/python run_ttnn_parity.py
```

## Next (precision/perf)
- Move residual-add / relu / TSTP onto device (ttnn.add / ttnn.relu / ttnn.mean+std)
  to make it fully on-device (currently conv on device, glue on host).
- Wire as a DiarizationBackend embedding provider (embedding_exclude_overlap path)
  so community-1 diarization uses the TT embedding on p150.
