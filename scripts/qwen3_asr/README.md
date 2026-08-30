# Qwen3-ASR TT p150 server supervisor

## Serving with `run.py --docker-server`

`--docker-server` is the standard delivery path, but Qwen3-ASR has no published
GHCR image yet, and its code is not upstream. The image must therefore be built
locally from the bring-up forks.

### 1. Base image

tt-metal's `dockerfile/Dockerfile` declares its tool/venv layers as
`FROM scratch` stubs that Bake substitutes; a plain `docker build` fails with
`COPY --from=cmake-layer /install/: lstat /install: no such file or directory`.
`scripts/build_docker_images.py` issues a plain `docker build` for the base, so
build the base with Bake first and tag it the way the script expects:

```
cd $TT_METAL_HOME
docker buildx bake -f dockerfile/docker-bake.hcl \
  --set ci-build.tags=local/tt-metal/tt-metalium/ubuntu-22.04-amd64:3b1b9ad \
  --set ci-build.output=type=docker \
  ci-build
```

The script then sees the base locally and only builds the dev image.

### 2. Dev image (from the bring-up forks)

The spec pins the bring-up branch heads, which live on the forks until they land
upstream, so both clone URLs have to be overridden:

```
cd $TT_INFERENCE_SERVER
TT_VLLM_REPO_URL=https://github.com/nyoshifujiTT/vllm.git \
TT_METAL_REPO_URL=https://github.com/nyoshifujiTT/tt-metal.git \
  python3 scripts/build_docker_images.py --build-metal-commit 3b1b9ad --single-threaded
```

Without `TT_METAL_REPO_URL` the image lacks the vLLM adapter and the server dies
with `ModuleNotFoundError: models.demos.audio.qwen3_asr.tt.generator_vllm`;
without `TT_VLLM_REPO_URL` it lacks the HF-config fix and dies with
`AttributeError: 'Qwen3ASRConfig' object has no attribute 'thinker_config'`.

Disk: the dev image is ~21 GB and a rebuild keeps the previous generation until
it is replaced, so keep at least 60 GB free.

### 3. Run

```
python3 run.py --model Qwen3-ASR-1.7B-JA --tt-device p150 --workflow server \
  --docker-server --dev-mode --no-auth --service-port 8110 --host-hf-cache \
  --override-docker-image ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.13.0-3b1b9ad-5e69638
```

`/health` turns 200 after ~12 minutes. Requests use the HF repo id, not the
spec's model name:

```
curl -X POST http://127.0.0.1:8110/v1/audio/transcriptions \
  -F file=@clip.wav -F model=neosophie/Qwen3-ASR-1.7B-JA -F language=ja
```

The very first transcription JIT-compiles kernels into the container's cache and
can take minutes; subsequent ones settle at ~2 s for an 11 s clip. Do not mistake
that first request for a hang.

The tt-metal decode path has a **non-deterministic device hang** in the same
SDPA/decode class tracked upstream (tt-metal issues #40592, #45052, #4752, also
seen for Mistral / gpt-oss / Falcon3). It is a platform-level bug, not specific
to the Qwen3-ASR adapter: memory is flat (no leak), single-device CCL/fabric is
short-circuited, and a watchdog (`TT_METAL_OPERATION_TIMEOUT_SECONDS`) shows the
stall in a decode device op (`device timeout, potential hang detected,
unrecoverable`).

Scope note (important): this hang was reproducible on the original board
(10.160.20.103), where a reused decode trace wedged the service within ~9-26
requests — which is why the adapter/spec historically shipped with decode
tracing OFF. On the delivery p150 (172.27.44.85) it does **not** reproduce:
decode trace ON (the current default) sustained conc=4 soaks of 300 and 600
requests (900 total) with zero wedges and `/health` 200 throughout, and the
earlier non-traced conc=4 x300 runs were also clean. The default now serves the
fast decode path (`trace_mode=decode_only`, `QWEN3ASR_DECODE_TRACE=1`); set
`QWEN3ASR_DECODE_TRACE=0` to fall back to untraced decode on any board where the
hang does reproduce.

`asr_supervisor.sh` remains as defense-in-depth: it keeps the ASR server
available in production the same way the Qwen3-Embedding fullbench supervisor
does on this hardware, so a wedge from any residual platform hang still
self-recovers rather than taking the service down:

1. Launch the standard `run.py --local-server` (our TT vLLM Qwen3-ASR adapter).
2. Wait for `/health` + a served model.
3. Monitor liveness with a bounded canary transcription every 20s.
4. On a wedge (2 consecutive canary failures): recover the device
   (`tt-smi -r`, then `ipmitool chassis power cycle` if that is insufficient)
   and relaunch.

`qwen3asr-supervisor.service` runs the supervisor under systemd so it
auto-starts on boot — including after a power-cycle recovery — making the
recovery loop fully self-sustaining.

## Install
```
sudo cp qwen3asr-supervisor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3asr-supervisor.service
```

Verified: injecting sustained variable-length load wedged the board; the
supervisor detected it, power-cycled, systemd restarted the supervisor on boot,
the server came back healthy, and `POST /v1/audio/transcriptions` returned the
correct transcript again — with no human intervention. (This wedge was induced
on the original board; the delivery p150 has not reproduced it across the
900-request decode-trace-on soaks above, but the supervisor is kept as a safety
net for the residual platform hang.)
