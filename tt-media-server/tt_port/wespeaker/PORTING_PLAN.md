# WeSpeaker ResNet34 (community-1 embedding) -> ttnn port plan

## Scope
Port the speaker-embedding NN of pyannote/speaker-diarization-community-1 to
Tenstorrent ttnn (Blackhole p150). This is homework 3, first target (embedding);
segmentation (SincNet + BiLSTM4) is the harder second target and is out of scope
for this first step (LSTM has no ttnn op yet).

## Exact architecture (verified from the real checkpoint)
- module: pyannote.audio.models.embedding.wespeaker.WeSpeakerResNet34
- input: log-mel Fbank, 80 bins (frontend stays on CPU/torchaudio; ttnn boundary
  is mel-features -> ResNet34)
- resnet.conv1: Conv2d(1->32, k3, pad1) + bn1 + relu
- layer1: 3x BasicBlock(32->32)   (blocks = [3,4,6,3])
- layer2: 4x BasicBlock(32->64, stride2 on first)
- layer3: 6x BasicBlock(64->128, stride2 on first)
- layer4: 3x BasicBlock(128->256, stride2 on first)
  BasicBlock = conv3x3+bn+relu -> conv3x3+bn -> (+identity/downsample) -> relu
- TSTP (temporal statistics pooling): mean & std over time -> concat = 5120-dim
  (256 channels x freq-collapsed 20? -> 5120 = 256 * 10 mean + 256*10 std;
  exact: flatten(channels*freq) then mean+std over time -> 2560*2 = 5120)
- resnet.seg_1: Linear(5120 -> 256)  => 256-dim embedding (NOT L2-normalized at
  model output; pyannote pipeline L2-normalizes downstream)

## ttnn op mapping (all available per Glean/tt-metal survey)
- Conv2d 3x3 (stride1/2, pad1): ttnn.conv2d (+ Conv2dConfig), template = ResNet50
  demo models/demos/vision/classification/resnet50 (ttnn_functional_resnet50.py)
- BatchNorm2d (inference): fold into conv (bn-fuse) or ttnn batch_norm; ResNet50
  demo folds bn into conv weights -> reuse that flow
- ReLU: ttnn.relu ; residual add: ttnn.add
- TSTP mean/std over time: ttnn.mean + ttnn.std (or mean of x and x^2 -> var);
  reshape/transpose via ttnn ops
- Linear seg_1: ttnn.linear (matmul + bias)
- L2 norm (if needed): ttnn.div by ttnn.sqrt(ttnn.sum(x*x))
No LSTM/SincNet/STFT needed for embedding (those are segmentation-side).

## Parity target (golden)
- golden_embeddings.json: 256-dim reference embeddings (CPU, real weights) for
  deterministic inputs d2/d3/d5. The ttnn port must match within tolerance
  (target: cosine > 0.999 / max_abs diff < 1e-2 bf16).
- gen_golden.py regenerates them.

## Porting steps (each a commit, parity-gated)
1. bn-fold: fold resnet.bn* into conv weights on CPU, verify CPU-folded == CPU
   reference (parity). [device-independent]
2. ttnn conv stem (conv1+bn+relu) parity on device vs CPU intermediate.
3. ttnn BasicBlock (one block) parity; then each layer.
4. ttnn TSTP parity.
5. ttnn seg_1 linear -> full 256-dim embedding parity vs golden.
6. integrate as models/demos/audio/wespeaker + expose to DiarizationBackend as a
   device embedding provider (embedding_exclude_overlap path).

## Device/runtime note
- Requires a built tt-metal/ttnn + exclusive p150. On this host the device is
  shared with the Qwen3-ASR agent (frequent tt-smi -r / ipmitool power cycles),
  so on-device steps (2-5) must be scheduled when the device is free. Steps 1 and
  the golden reference are device-independent and done here.
