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
  --set ci-build.tags=local/tt-metal/tt-metalium/ubuntu-22.04-amd64:1395635 \
  --set ci-build.output=type=docker \
  ci-build
```

The script then sees the base locally and only builds the dev image.

### Where the pins live

`workflows/model_specs/dev/audio_tts.yaml` carries no `version`,
`tt_metal_commit` or `vllm_commit`, and cannot: `_build_template` constructs dev
entries as `ModelSpecTemplate`, which has no pin fields, so a dev entry that
sets one fails to load. Pins belong to `ProdModelSpecTemplate`, where they are
required.

Prod entries are release artifacts, written by
`scripts/release/promote_dev_spec_to_prod.py` for the leaves listed in
`.github/workflows/models-ci-config.json`. A bring-up is not in that file, so
there is no promotion path for this model yet and no committed place for its
pins.

`scripts/build_docker_images.py` reads both pins from the catalog, so building
before release needs a prod entry to exist temporarily. That is the second half
of the manual patch below: applied, built from, and reverted -- never committed.

### Why the pin may lag the branch head

`tt_metal_commit` names the tree the image is BUILT from, so it only has to move
when a commit changes something the image contains. Commits that touch only
`models/demos/audio/qwen3_asr/tests/` (host-only tests, run from a checkout and
never copied into the image) produce a byte-identical image, so bumping the pin
for them would force a ~7 h rebuild that cannot change the result.

Before leaving the pin behind the branch head, verify there is no runtime diff:

```
git diff --name-only <pinned> <head> -- models/demos/audio/qwen3_asr \
  | grep -v '/tests/'      # must print nothing
```

If that prints anything, bump the pin and rebuild.

### 2. Dev image (from the bring-up forks)

The bring-up branch heads live on forks until they land upstream. Every
repository in this bring-up uses the same branch name:

| repository | branch | pinned by |
|---|---|---|
| `nyoshifujiTT/tt-metal` | `nyoshifujiTT/qwen3-asr-17b_p150x1` | `tt_metal_commit` |
| `nyoshifujiTT/vllm-tt-plugin` | `nyoshifujiTT/qwen3-asr-17b_p150x1` | `vllm_commit` |

The clone checks out the pinned commit, not the branch, so the branch only has
to *contain* it.

#### The one manual patch

Two things have to change to build before release, and both are temporary:

1. the two clone URLs, which the committed Dockerfile always points upstream;
2. a prod catalog entry, because `scripts/build_docker_images.py` reads the
   release pins from the catalog and the dev contract forbids them there.

The second point is worth stating precisely, since editing prod is normally out
of bounds. Prod entries are release artifacts written by
`scripts/release/promote_dev_spec_to_prod.py`, which promotes leaves selected by
`.github/workflows/models-ci-config.json` -- a bring-up is not in that file, so
promotion cannot produce this entry. The entry below is therefore added by the
patch and reverted straight after, never committed. The same shape was used for
the pyannote bring-up on this repo.

Apply, build, then restore -- the recipe PR#4837 established:

```
cd $TT_INFERENCE_SERVER
git apply <<'PATCH'
diff --git a/vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile b/vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile
--- a/vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile
+++ b/vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile
@@ -86,7 +86,7 @@ ENV UV_HTTP_RETRIES=10
 # A full-history clone of tt-metal has taken over an hour on CI, connection dropped ("fatal: early
 # EOF"). Only the pinned commit is needed, so fetch just that (matches the shallow
 # clone already used by tt-media-server/Dockerfile).
-RUN /bin/bash -c "git clone --depth 1 https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} \
+RUN /bin/bash -c "git clone --depth 1 https://github.com/nyoshifujiTT/tt-metal.git ${TT_METAL_HOME} \
     && cd ${TT_METAL_HOME} \
     && git fetch --depth 1 origin ${TT_METAL_COMMIT_SHA_OR_TAG} \
     && git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} \
@@ -101,7 +101,7 @@ RUN /bin/bash -c "git clone --depth 1 https://github.com/tenstorrent-metal/tt-me
 # Build vllm-tt-plugin - clone with minimal history and clean.
 # The plugin owns the vLLM version pin and its dependency overrides, so the
 # install is delegated to its own docs/install-vllm-tt.sh rather than restated here
-RUN /bin/bash -c "git clone https://github.com/tenstorrent/vllm-tt-plugin.git ${vllm_tt_plugin_dir} \
+RUN /bin/bash -c "git clone https://github.com/nyoshifujiTT/vllm-tt-plugin.git ${vllm_tt_plugin_dir} \
     && cd ${vllm_tt_plugin_dir} \
     && git checkout ${TT_VLLM_COMMIT_SHA_OR_TAG} \
     && source ${PYTHON_ENV_DIR}/bin/activate \
diff --git a/workflows/model_specs/prod/audio_tts.yaml b/workflows/model_specs/prod/audio_tts.yaml
--- a/workflows/model_specs/prod/audio_tts.yaml
+++ b/workflows/model_specs/prod/audio_tts.yaml
@@ -272,3 +272,27 @@ templates:
       env_vars:
         TT_MESH_GRAPH_DESC_PATH: "../../tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
   status: EXPERIMENTAL
+- weights:
+    - neosophie/Qwen3-ASR-1.7B-JA
+  version: "0.1.0"
+  tt_metal_commit: "1395635"
+  vllm_commit: "50695d8"
+  impl: tt_vllm_plugin
+  min_disk_gb: 15
+  min_ram_gb: 6
+  model_type: AUDIO
+  inference_engine: VLLM
+  model_display_name: Qwen3-ASR-1.7B
+  has_builtin_warmup: true
+  device_model_specs:
+    - device: P150
+      max_concurrency: 4
+      max_context: 2048
+      default_impl: true
+      env_vars:
+        MESH_DEVICE: "P150"
+        HF_HUB_OFFLINE: "1"
+        TRANSFORMERS_OFFLINE: "1"
+      vllm_args:
+        additional-config: '{"tt": {"trace_mode": "decode_only"}}'
+  status: EXPERIMENTAL
PATCH

python3 scripts/build_docker_images.py --build-metal-commit 1395635 --single-threaded

git checkout vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile \
             workflows/model_specs/prod/audio_tts.yaml
```

`--build-metal-commit` only filters which combinations get built; the commit
values themselves come from the catalog, which is why the pins have to be in
the patch rather than on the command line.

Without the tt-metal half the image lacks the vLLM adapter and the server dies
with `ModuleNotFoundError: models.demos.audio.qwen3_asr.tt.generator_vllm`.
Without the plugin half it lacks the TT adapter registration and the engine
fails to resolve the architecture. The URL halves disappear once the commits
are upstream; the prod entry disappears once the model is part of a release.

There is deliberately no build arg for either URL. Build args for exactly this
were added earlier in the bring-up and withdrawn: they let an image built from
a fork look as though it came from the committed Dockerfile.

#### What `vllm_commit` names

A *vllm-tt-plugin* commit, despite the name. The dev Dockerfile clones only
`tenstorrent/vllm-tt-plugin`, which owns the vLLM version pin and installs it
via its own `docs/install-vllm-tt.sh` -- the same layout `tt-inference-server`
main uses. The field keeps its old name from when the image cloned the
`tenstorrent/vllm` fork.

**That fork is deprecated.** `tenstorrent/vllm`'s README says "This repository
is deprecated. Do not use it"; TT-specific issues are redirected to
`vllm-tt-plugin` (tenstorrent/vllm#477), and `tt-inference-server` itself
switched off it in PR #4907, which this branch has now merged. `run.py` still
accepts `--vllm-dir` but reports it as "deprecated and ignored".

Nothing was lost in the switch. Every Qwen3-ASR change the fork carried has an
equivalent on the standalone plugin (TT adapter registration, the
audio/transcription wiring in `TTModelRunner`, surfacing the real
`execute_model` error, forced eager execution), and its remaining change --
ordering `thinker_config` before `super().__init__()` in the HF config -- ships
in the `vllm==0.24.0` release the plugin pins.

#### Evidence that the move changed no output

The fork route was run one last time before it was retired, on the same board,
same clips, same shipped defaults (decode trace on, eager, conc=4, seq=4), to
show the migration is output-preserving:

| | plugin route | fork route (retired) |
|---|---|---|
| golden transcript | identical | identical |
| TED 509 CER | 0.1002 (494 ok) | 0.1002 (494 ok) |
| MagicHub 600 CER | 0.1668 (600 ok) | 0.1668 (600 ok) |
| bench 128 req | 11.97 audio-s/s, p50 2.24 s | 12.73 audio-s/s, p50 2.02 s |

Accuracy matches to four decimal places on both corpora. The throughput gap is
session drift on this board, not a route difference: a later plugin-route run
measured 12.86 audio-s/s at p50 2.03 s. An older run also had LibriSpeech WER
6.7288 on both.

This table is a record of the migration, not an invitation to run the fork.
There is one supported route: upstream vLLM plus `vllm-tt-plugin`.

#### Relationship to upstream

This branch has `tt-inference-server` main merged in, so the layout here is the
upstream one: catalogs are YAML under `workflows/model_specs/{dev,prod}/`, the
eval and benchmark harnesses live under `reference_config/` and `llm_module/`,
reporting is in `report_module/`, and the dev Dockerfile clones only
`vllm-tt-plugin`.

Two things are deliberately *not* followed:

- **`tt-metal` stays on the model's PR branch** (`upstream/yito/qwen3_asr_pr`),
  not on tt-metal main. The ttnn implementation is still in review there;
  rebasing onto main ahead of that PR would put this bring-up in conflict with
  it. Once the PR lands, the pin becomes an ordinary main commit.
- **`vllm-tt-plugin` stays on its current base.** Upstream has moved on and
  raised its vLLM pin from 0.24.0 to 0.26.0. That is not a layout change: this
  tree carries a `TTUniProcExecutor` that exists specifically because vLLM
  0.24.0 dropped `UniProcExecutor`'s async output thread and serialised decode
  read-back. Whether 0.26.0 still needs it, and what it does to throughput, has
  to be measured -- so that upgrade is its own piece of work, not a side effect
  of this one.

#### If a branch is not pushed yet

The patch above assumes the pinned commits are reachable on the forks. While a
branch only exists locally, serve it over the loopback and point that half of
the patch at it instead -- the build still performs an ordinary `git clone`,
only the URL differs:

Both clones need this, not just tt-metal — the patch points the plugin at a
fork too:

```
# once, on the docker host: one bare repo per fork, both under /tmp
git clone --bare <local tt-metal checkout>       /tmp/ttmetal-src.git
git clone --bare <local vllm-tt-plugin checkout> /tmp/vllmttplugin-src.git
for d in /tmp/ttmetal-src.git /tmp/vllmttplugin-src.git; do
  (cd "$d" && git update-server-info)
done

# one daemon exports everything under the base path
git daemon --reuseaddr --base-path=/tmp --export-all --enable=upload-pack \
  --listen=0.0.0.0 --port=9418 &

# confirm the pinned commits are reachable before starting a multi-hour build
git ls-remote git://172.17.0.1:9418/ttmetal-src.git
git ls-remote git://172.17.0.1:9418/vllmttplugin-src.git
```

Then substitute both URLs in the patch (172.17.0.1 is the docker0 gateway):
`git://172.17.0.1:9418/ttmetal-src.git` and
`git://172.17.0.1:9418/vllmttplugin-src.git`. Push the branches and drop this
step before handing the recipe over: an image built from a daemon on one host
is not reproducible by anyone else.

Disk: the dev image is ~21 GB and a rebuild keeps the previous generation until
it is replaced, so keep at least 60 GB free.

### 3. Run

```
python3 run.py --model Qwen3-ASR-1.7B-JA --tt-device p150 --workflow server \
  --docker-server --dev-mode --no-auth --service-port 8110 --host-hf-cache \
  --override-docker-image ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.21.0-1395635-50695d8
```

`/health` turns 200 after ~12 minutes. Requests use the HF repo id, not the
spec's model name:

```
curl -X POST http://127.0.0.1:8110/v1/audio/transcriptions \
  -F file=@clip.wav -F model=neosophie/Qwen3-ASR-1.7B-JA -F language=ja
```

#### The clip to check with

Use a clip that has a published reference transcript, so the output can actually
be judged. The sanity clip used throughout this bring-up is FLEURS `ja_jp`
`test[0]`; fetch it rather than copying a wav from someone's scratch directory:

```python
from datasets import load_dataset
import soundfile as sf, librosa, numpy as np

ds = load_dataset("google/fleurs", "ja_jp", split="test", streaming=True)
ex = next(iter(ds))
w = np.asarray(ex["audio"]["array"], dtype="float32")
sr = ex["audio"]["sampling_rate"]
if sr != 16000:
    w = librosa.resample(w, orig_sr=sr, target_sr=16000)
sf.write("real_ja.wav", w, 16000)   # 10.44 s
print(ex["transcription"])
```

Reference: `インターネットで 敵対的環境コース について検索すると おそらく現地企業の住所が出てくるでしょう`

Served output on p150 (bf8 weights, greedy) differs from the reference only in
homophones, e.g. `インターネットで適体適環境コースについて検索すると、おそらく現地企業の住所が出てくるでしょう。`
That is the expected result; treat a materially different string as a
regression.

The snippet resamples to 16 kHz because that is what the checkpoint's
`preprocessor_config.json` declares (`sampling_rate: 16000`, and `n_samples`
480000 / `nb_max_frames` 3000 x `hop_length` 160 are all consistent with it), so
16 kHz is what the mel front-end is calibrated for. It is *not* a requirement on
what you may POST: the server resamples and downmixes for you. Verified against
the running server -- 44.1 kHz mono, 44.1 kHz stereo and 16 kHz stereo copies of
this clip all return the same golden transcript.

Writing the file at 16 kHz mono just makes it byte-identical to the clip these
numbers were measured with (md5 `3d43ec3ac2562231ec7c8c9ce4087ba4`).

**Do not use `ja_words.wav` or `test15s.wav` for this.** Both are synthetic
fixtures with no reference transcript: `ja_words.wav` is a concatenation of
isolated words made during bring-up to compare the TT and CPU decoders against
each other, and `test15s.wav` is the warmup waveform pointed at by
`QWEN3ASR_WARMUP_WAV` (looped English, which the JA model ends immediately).
They are fine for "does the server answer at all", but any accuracy claim based
on them is meaningless. Accuracy is measured with the corpus evals (TED,
MagicHub, LibriSpeech), not with a single clip.

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
