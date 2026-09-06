# Qwen3-ASR TT p150 server supervisor

## Checkouts this runbook refers to

Three trees are used below, always by these names. Export them before following
anything else -- the commands `cd` into them and will otherwise land wherever
the variable is empty:

```
export TT_METAL_HOME=/path/to/tt-metal                  # nyoshifujiTT/tt-metal
export TT_INFERENCE_SERVER=/path/to/tt-inference-server # this repo
export VLLM_TT_PLUGIN=/path/to/vllm-tt-plugin           # nyoshifujiTT/vllm-tt-plugin
```

All three carry the same branch, `nyoshifujiTT/qwen3-asr-17b_p150x1`.

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
  --set ci-build.tags=local/tt-metal/tt-metalium/ubuntu-22.04-amd64:e7929dcf5dcf2ad8f8f98dc945d012f61ad075cb \
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

#### `tt_metal_commit` must be the full 40-character SHA

`build_docker_images.py` expands whatever is written here by running
`git ls-remote https://github.com/tenstorrent/tt-metal.git | grep <pin>` and
taking the first hit. That is a substring match against **upstream**, which
does not have this branch, so a short pin can silently resolve to an unrelated
object. `e7929dc` matched upstream's `7e7929dcd898...` (`refs/pull/9507/head`)
and the build died cloning a commit that does not exist on the fork:

```
Resolved e7929dc to full SHA via ls-remote (first match): 7e7929dcd898a0c655f56b77fb147742829fc26f
... returned non-zero exit status 1
```

A full SHA passes through that grep as itself, so it is the only safe form
while the branch lives on a fork. `vllm_commit` is not resolved this way and
may stay short. Note the pin is also the image tag, so the tag carries the full
SHA too, and `--build-metal-commit` must be given the identical string -- it is
an exact-equality filter over catalog entries.

`tt_metal_commit` and `vllm_commit` name the trees the image is BUILT from, so
they only have to move when a commit changes something the served code path
reads. Both clones are whole working trees, so a `tests/` directory *is* present
inside the image -- verified on a running container:

```
$ docker exec <container> ls /home/container_app_user/tt-metal/models/demos/audio/qwen3_asr/tests | wc -l
18
$ docker exec <container> ls /home/container_app_user/vllm-tt-plugin/tests/*.py | wc -l
30
```

What makes a test-only commit safe to leave behind the pin is not absence from
the image but absence from the import graph: nothing the server loads imports
those modules. `vllm_tt_plugin` resolves to `.../vllm-tt-plugin/src/vllm_tt_plugin`
(an editable install rooted at `src/`, so `tests/` is outside it), and the
tt-metal demo package is imported as `models.demos.audio.qwen3_asr.tt.*`.
A commit that only adds or edits test modules therefore cannot change what the
server executes, and bumping the pin for it would force a ~7 h rebuild that
cannot change the result.

That is checkable on a serving container rather than argued: Python writes
`__pycache__` only for modules it actually imports, so the directories that
exist say which ones were loaded.

```
$ docker exec <container> ls .../qwen3_asr/tt/__pycache__ | head -4
__init__.cpython-310.pyc
audio_encoder.cpython-310.pyc
generator_vllm.cpython-310.pyc
qwen3_asr_decoder.cpython-310.pyc
$ docker exec <container> ls .../qwen3_asr/tests/__pycache__
ls: cannot access '.../qwen3_asr/tests/__pycache__': No such file or directory
```

So the served `tt/` package is byte-compiled and `tests/` was never imported
once, on a container that has served every corpus run above. Run this check
instead of trusting the reasoning if you ever need to defend leaving the pin
where it is.

Before leaving a pin behind its branch head, verify there is no runtime diff:

```
# tt-metal
git diff --name-only <pinned> <head> -- models/demos/audio/qwen3_asr \
  | grep -vE '/tests/|^.*/(README\.md|reference/dump_reference\.py|reference/extract_text_decoder\.py|eval/corpus_eval\.py)$'

# vllm-tt-plugin
git diff --name-only <pinned> <head> \
  | grep -v '^tests/'      # must print nothing
```

If either prints anything, it is not yet a verdict -- run the comment-only
check below on each file printed. Only a file that survives *both* forces the
pin to move.

At the current pins the tt-metal command is not silent: it prints
`tt/generator_vllm.py` and `tt/qwen3_asr_decoder.py`. That is expected and is
the worked example of the next paragraph -- both commits rewrote comment blocks
(the ND-hang issue references, and the prompt-length bound) and nothing else,
so the pin stays. The plugin command does print nothing.

One exception the filename filter cannot express: a commit that touches a
served module but changes only comments. The rule is "does the server execute
something different", and it does not. Confirm it rather than assuming, then
leave the pin:

```
git diff <pinned> <head> -- <the file> \
  | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)' \
  | grep -vE '^[+-][[:space:]]*(#|$)'                   # must print nothing
```

Anything printed is a real code change and the pin has to move.

`[[:space:]]*` is load-bearing: the earlier form anchored `#` directly after
the `+`/`-`, so it only recognised a comment in column 0. Every comment inside
a function is indented, so an indented comment-only diff was reported as a real
code change -- which is what happened to `tt/qwen3_asr_decoder.py`, whose
comment edit printed until this was fixed. Trailing `|$` drops blank-line
changes for the same reason.

The extra tt-metal exclusions are the same rule, not exceptions to it. Those
files ship inside the image but nothing the server loads imports them: the
served path enters at `models.demos.audio.qwen3_asr.tt.*`, while
`reference/dump_reference.py` and `reference/extract_text_decoder.py` are
golden-generation scripts run by hand from a separate CPU venv, and
`eval/corpus_eval.py` is the offline demo-side eval. Confirm the claim rather
than trusting the list -- a file is only safe to exclude if no module reachable
from `tt/` imports it:

```
grep -rn '<basename without .py>' --include='*.py' models/demos/audio/qwen3_asr/tt/
```

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

Apply, build, then restore. The recipe comes from a review comment on PR #4837
(`issuecomment-5390617037`), not from that PR's merged diff -- the commit
itself only adds Qwen3.5/3.6-27B specs. There it rewrote two existing pins:

```
git apply <<'EOF'
diff --git a/workflows/model_specs/prod/llm.yaml b/workflows/model_specs/prod/llm.yaml
@@ -1192,7 +1192,7 @@ templates:
   tt_metal_commit: "de59f8a"
-  vllm_commit: "03fa3af"
+  vllm_commit: "b95c0501e62f"
EOF
```

A bring-up has no prod entry to rewrite, so the same shape is used to add one
temporarily -- as the pyannote bring-up on this repo did:

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
+  tt_metal_commit: "e7929dcf5dcf2ad8f8f98dc945d012f61ad075cb"
+  vllm_commit: "c0c4842"
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
+      override_tt_config:
+        trace_mode: decode_only
+  status: EXPERIMENTAL
PATCH

python3 scripts/build_docker_images.py --build-metal-commit e7929dcf5dcf2ad8f8f98dc945d012f61ad075cb --single-threaded

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
`tenstorrent/vllm-tt-plugin` for the vLLM side -- it owns the vLLM version pin
and installs it via its own `docs/install-vllm-tt.sh`, the same layout
`tt-inference-server` main uses. (The Dockerfile clones three repositories in
total: tt-metal, vllm-tt-plugin and tt-smi. "Only" here means there is no
second vLLM source.) The field keeps its old name from when the image cloned
the `tenstorrent/vllm` fork.

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
in the vLLM release the plugin pins (0.24.0 at the time; 0.26.0 today).

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
measured 12.86 audio-s/s at p50 2.03 s.

An older run also had LibriSpeech WER 6.7288 on both routes. **That figure is
not reproducible on this tree** and is kept only as a historical
cross-check: it came from an lmms-eval config in `evals/eval_config.py`, and
upstream deleted that file (#4678 / #4630) when the v1 workflows went. The
successor catalog `reference_config/evals/eval_config.py` carries whisper
entries but no Qwen3-ASR one, and our `qwen3_asr_openai` model adapter still
sits under `evals/lmms_eval_models/` with **nothing referencing it** -- that
directory is all that is left of `evals/`.

Re-enabling it is not just a catalog entry, because the way lmms-eval models
are installed changed underneath us. Traced through the tree:

- The successor catalog has a working template -- the `whisper-large-v3` entry
  at `reference_config/evals/eval_config.py` uses
  `eval_class="whisper_tt"`, `WorkflowVenvType.EVALS_AUDIO` and
  `score_task_single_key` on `wer,none`. A Qwen3-ASR entry would be the same
  shape with `eval_class="qwen3_asr_openai"`.
- But our adapter reached the venv through `evals/lmms_eval_models/install.py`,
  which copied it into the venv's `lmms_eval` package and patched its registry.
  That was driven by `setup_evals_audio()`, and **upstream deleted that hook**
  (`workflow_venvs.py` now declares `EVALS_AUDIO` with no `setup_function`).
- Upstream gets `whisper_tt` from a **TT fork of lmms-eval** instead
  (`requirements/evals-audio.txt` pins
  `git+https://github.com/bgoelTT/lmms-eval.git@ben/samt/whisper-tt`), so the
  model lives in the dependency, not in a post-install copy step.

So the work is: get `qwen3_asr_openai` into that fork (or restore a
`setup_function` for `EVALS_AUDIO`), then add the catalog entry. Leaving the
adapter where it is and only adding the entry would fail at model resolution.

Note `requirements/evals-audio.txt` still says "Used by: setup_evals_audio()
in workflows/workflow_venvs.py", which no longer exists anywhere in the repo --
that comment is upstream's, and it is what sent me looking for a hook that had
already been removed.

Accuracy on this tree is measured with the corpus evals (TED, MagicHub), which
do run.

This table is a record of the migration, not an invitation to run the fork.
There is one supported route: upstream vLLM plus `vllm-tt-plugin`.

#### Relationship to upstream

This branch has `tt-inference-server` main merged in, so the layout here is the
upstream one: catalogs are YAML under `workflows/model_specs/{dev,prod}/`, the
eval and benchmark harnesses live under `reference_config/` and `llm_module/`,
reporting is in `report_module/`, and the dev Dockerfile takes its vLLM from
`vllm-tt-plugin` alone -- the `tenstorrent/vllm` clone is gone.

`vllm-tt-plugin` has upstream merged in too, which raises the vLLM it installs
from 0.24.0 to 0.26.0. Three of the plugin commits this bring-up carried were
dropped in that merge because upstream had already fixed the same things: the
launcher optional-import shim, the `sample_tokens` empty-deque guard, and a
Gemma-4 tool-parser test upstream deleted outright.

`TTUniProcExecutor` stays. Upstream pins `distributed_executor_backend` to
`"uni"` for lane-DP and adds no executor of its own, and vLLM still finalizes
the decode read-back inline -- `UniProcExecutor`'s async output thread was
removed in 0.24.0 and has not returned in 0.26.0. Measured across the upgrade:
TED CER 0.1002 and MagicHub CER 0.1668 unchanged, `rtfx` 12.24 -> 12.61, p99
7.73 s -> 5.89 s.

One thing is deliberately *not* followed:

- **`tt-metal` stays on the model's PR branch** (`upstream/yito/qwen3_asr_pr`),
  not on tt-metal main. The ttnn implementation is still in review there;
  rebasing onto main ahead of that PR would put this bring-up in conflict with
  it. Once the PR lands, the pin becomes an ordinary main commit.

#### After any upstream merge, check for silently dropped work

A merge can take a change of ours *and* the test that covered it, and then
nothing fails. That happened in `vllm-tt-plugin`: the first merge kept
`enforce_eager = True` but dropped the `compilation_config` pin that must
accompany it, and dropped `test_check_and_update_config_forces_eager` in the
same commit. The branch ran for weeks on half the change, spending ~209 s per
start building a graph the ttnn path never uses (restored in `acae5aa`).

A green suite does not detect this, because the assertion left with the code.
Run this in each repo after merging. `TESTS` is the test root, which differs
per repo -- the scan silently finds nothing if it is wrong, so set it
deliberately:

```
TESTS=tests                                        # tt-inference-server, vllm-tt-plugin
TESTS=models/demos/audio/qwen3_asr/tests           # tt-metal

git log --format='%h %an' HEAD | grep -i codex | cut -d' ' -f1 | while read c; do
  for t in $(git show $c --name-only --format= | grep -E "^$TESTS/.*\.py\$"); do
    for fn in $(git show $c -- "$t" | grep -oP '^\+\s*def \Ktest_\w+' | sort -u); do
      grep -rq "def $fn" "$TESTS/" || echo "LOST: $fn <- $c $t"
    done
  done
done | sort -u
```

Every line has to be accounted for as one of: renamed, replaced by a later
commit of ours, deliberately withdrawn (say where that is written down), or a
real loss to restore. Do not treat a long list as noise -- run across all
three repos this produced 33 lines here, 2 in tt-metal and 0 in the plugin
(after `acae5aa`), and every one of the 35 resolved to the first three
categories. That is only meaningful because each was checked individually.

On tt-metal, restrict the history to our own commits; the whole log is ~200k
commits of upstream and the scan is quadratic in what you feed it. Resolve the
base by remote-tracking ref rather than by name -- the PR branch is
`upstream/yito/qwen3_asr_pr` in some checkouts and `origin/yito/qwen3_asr_pr`
in others, and a bad revision aborts the scan with `fatal: ambiguous
argument`:

```
BASE=$(git rev-parse --verify -q upstream/yito/qwen3_asr_pr \
       || git rev-parse --verify -q origin/yito/qwen3_asr_pr)
git log --format='%h %an' "$BASE..HEAD" | grep -i codex | ...
```

**The test scan alone is not enough.** The plugin's loss was an
implementation line, not a test -- the test went with it, so scanning tests
found it, but the reverse case exists too: a file of ours can be deleted
upstream while every test still passes because the tests went to the same
place. Scan the implementation side as well:

```
git log --format='%h %an' HEAD | grep -i codex | cut -d' ' -f1 | while read c; do
  for f in $(git show $c --name-only --format= \
             | grep -vE "^$TESTS/" | grep -E '\.(py|sh|yaml)$'); do
    [ -e "$f" ] || echo "FILE GONE: $f <- $c"
  done
done | sort -u
```

Here that prints one line: `workflows/run_reports.py`, which upstream deleted
in #4630 ("route all workflows to v2 and delete dead v1 code"). Our change to
it made the reports workflow tolerate a missing `functional_ttft` on an
eval-only audio run. Checked before writing it off: `functional_ttft` no
longer appears anywhere in the repo, and the v2 audio path carries ttft as
`Optional[float]` with a `is not None` filter rather than a dict subscript, so
that `KeyError` cannot recur. Verdict: obsoleted by the rewrite, not lost.

A `FILE GONE` line needs the same treatment as a `LOST` one -- find where the
behaviour went, and confirm the problem it solved cannot come back. Do not
restore the file.

Disk: the dev image is ~21 GB and a rebuild keeps the previous generation until
it is replaced, so keep at least 60 GB free.

### 3. Run

```
MODEL_SPECS_ENV=dev python3 run.py --model Qwen3-ASR-1.7B-JA --tt-device p150 \
  --workflow server --docker-server --dev-mode --no-auth --service-port 8110 \
  --host-hf-cache \
  --override-docker-image ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.21.0-e7929dcf5dcf2ad8f8f98dc945d012f61ad075cb-c0c4842
```

`MODEL_SPECS_ENV=dev` is required here for the same reason as in the build: the
catalog defaults to prod, which has no Qwen3-ASR entry, and `run.py` would exit
saying the model is unknown.

`/health` turns 200 in **140 s to ~12 minutes**, and which end you get depends
on the kernel cache, not on the machine:

| start | timed | why |
|---|---|---|
| cold kernel cache | 7-12 min (**460 s** measured) | tt-metal JIT-compiles kernels |
| warm kernel cache | **140 s** measured | the cache is a docker volume (`volume_id_tt_vllm_plugin-Qwen3-ASR-1.7B-JA`, ~169 MB at `~/.cache/tt-metal-cache`) and survives container restarts |

So a fast start is normal on a machine that has served before, and a slow one
is normal after `docker volume rm` or on a new host. The supervisor's budget is
sized for the slow end (20 min) rather than for whichever figure you happen to
measure.

A second, separate cost sits inside that window on the image pinned above:

```
init engine (profile, create kv cache, warmup model) took 214.11 s (compilation: 208.97 s)
compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, ...}
```

The plugin sets `enforce_eager=True`, but `VllmConfig.__post_init__` derives
the compilation mode *before* the platform hook runs, so on this pin the mode
stays `VLLM_COMPILE` and ~209 s goes into a compiled graph the ttnn hot path
never uses. `vllm-tt-plugin` `acae5aa` restores the pin that forces
`CompilationMode.NONE`; a rebuild past it should drop most of that 209 s.

This does **not** invalidate the throughput and accuracy numbers below.
Upstream's own later check (`Enforce eager set, disabling torch.compile and
CUDAGraphs`) still lands, and `cudagraph_mode` is already `NONE`, so execution
was eager either way -- the loss is startup time, not steady-state speed.

Requests use the HF repo id, not the spec's model name:

```
curl -X POST http://127.0.0.1:8110/v1/audio/transcriptions \
  -F file=@clip.wav -F model=neosophie/Qwen3-ASR-1.7B-JA -F language=ja
```

#### `ARCH_NAME` is absent from the engine, and that is fine

Checking the spec's `env_vars` per process rather than by reading the startup
log, `ARCH_NAME` is the one that does not reach the worker:

| variable | pid 1 (entrypoint) | APIServer | EngineCore |
|---|---|---|---|
| `MESH_DEVICE`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | -- | set | set |
| `ARCH_NAME` | `wormhole_b0` (from the base image) | `blackhole` | **absent** |

The spec is right -- the runtime spec carries `ARCH_NAME: blackhole` and the
entrypoint logs `overriding with blackhole` -- so this looks like a leak. It is
not: UMD reads the architecture off PCIe,

```
UMD | Creating TopologyDiscovery for architecture: blackhole
```

and `model_spec.py` marks the variable itself as transitional
(`TODO: Remove once all model specs are uplifted to tt-metal >= 0.60.0`).
Do not "fix" it by exporting `ARCH_NAME` globally: the base image sets
`wormhole_b0`, so a global export is how the wrong value would reach a worker.

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

The snippet resamples to 16 kHz because that is the rate the mel front-end is
calibrated for. The two checkpoints say so differently, which is worth knowing
before quoting either file:

| checkpoint | `sampling_rate` in `preprocessor_config.json` | `feature_extractor.sampling_rate` |
|---|---|---|
| `neosophie/Qwen3-ASR-1.7B-JA` (served) | `16000`, declared | 16000 |
| `Qwen/Qwen3-ASR-1.7B` (reference dumps) | **absent** | 16000, from `WhisperFeatureExtractor`'s default |

So only the JA checkpoint declares the rate; the base one inherits it. Either
way the geometry in both files agrees with 16 kHz -- `n_samples` 480000 =
`chunk_length` 30 x 16000, and `nb_max_frames` 3000 x `hop_length` 160 = 480000.
Prefer that arithmetic over the key if you need to re-derive the rate, because
it is present in both files.

16 kHz is *not* a requirement on what you may POST: the server resamples and
downmixes for you. Verified against the running server -- 44.1 kHz mono,
44.1 kHz stereo and 16 kHz stereo copies of this clip all return the same
golden transcript.

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

The tt-metal decode path has a **non-deterministic device hang** in a class
tracked upstream. The closest matches by signature, checked rather than
recalled:

| issue | what it actually is | same class? |
|---|---|---|
| [tt-metal#37543](https://github.com/tenstorrent/tt-metal/issues/37543) | "[GPT-OSS] ND hang in SDPA decode" -- hangs after 20-120 min, decode traced, only under vLLM | yes: SDPA decode, ND, trace-related |
| [tt-metal#36395](https://github.com/tenstorrent/tt-metal/issues/36395) | "[GPT-OSS] ND hangs" -- mostly prefill, some decode | partly |
| [tt-metal#45052](https://github.com/tenstorrent/tt-metal/issues/45052) | gpt-oss decode hang after `paged_fill_cache`, isl>=1024 | shares the watchdog signature, but that one is 100% deterministic and reported only on P300x2 |

Two details on that last row, since they were previously stated more strongly
than the tracker supports:

- The **defect** is not established as P300x2-only. The report is specific to a
  Blackhole P300x2 `(1,4)` mesh, and the stuck op is GPT-OSS MoE
  `SparseMatmulDeviceOperation`; a later issue
  ([#45943](https://github.com/tenstorrent/tt-metal/issues/45943)) describes the
  same sparse-matmul deadlock as architecture-agnostic. "Reported only on
  P300x2" is the accurate claim -- this deployment is single-device p150 and
  runs no MoE, which is why it is still the wrong class for us.
- PR #44118 is the **first bad tested version** (`7eff69a85a0`, against
  `747215b` last known good), not a proven root cause; triage names #43682 as
  the better bisection target. Cite it as a regression boundary, not a culprit.

Two issues quoted here previously do **not** belong: #40592 is a Mistral
*AllGatherAsync* hang on T3K, and #4752 is a tt-inference-server *eval accuracy*
issue for Falcon3, not a hang at all. Neither supports the claim.

What is ours, and measured here: it is not specific to the Qwen3-ASR adapter --
memory is flat (no leak), single-device CCL/fabric is short-circuited, and a
watchdog (`TT_METAL_OPERATION_TIMEOUT_SECONDS`) puts the stall in a decode
device op (`device timeout, potential hang detected, unrecoverable`).

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

The soaks were deliberate; the accumulated evidence since is larger. Read the
counter off a server that has been through the eval and benchmark suites:

```
$ curl -s http://127.0.0.1:8110/metrics | grep '^vllm:request_success_total'
vllm:request_success_total{...,finished_reason="stop",...} 6675.0
```

6675 transcriptions finished in one engine process -- TED 509 and MagicHub 600
several times over, plus the benchmark batches -- with decode tracing on, no
restart, and no `potential hang` / `unrecoverable` line in the container log.

**That figure is a high-water mark from one engine process, not a value to
match.** The counter resets when the engine restarts, so a fresh server
legitimately reads far lower; a later session measured 3064 after one pass of
each suite. What carries over between sessions is the shape, not the number:

- `error` and `abort` stay at `0.0` (they have on every run recorded here),
- `stop` advances by exactly the requests you issued, and
- no `potential hang` / `unrecoverable` appears in the container log
  (`docker logs <container> | grep -cE 'potential hang|unrecoverable'` -> `0`).

Checking the arithmetic is what makes it evidence rather than a big number:
one pass of TED 509 + MagicHub 600 + benchmark 128, minus the 15 empty-wav
failures, plus one golden clip, is +1223 -- which is what the delta was.

If the two perf probes ran as well, add **6**, not 120: each one transcribes
the clip 3 times before it starts timing (`for _ in range(3)` in both
`asr_perf_probe.py` and `asr_perf_stream.py`), and those warm-up requests are
counted by the server even though they are excluded from the reported figures.
A full pass of everything above therefore lands on
`1 + 494 + 600 + 128 + 63 + 63 = 1349`, measured as 1844 -> 3193. Forget the
warm-ups and the sum comes out 6 short, which looks like six dropped requests.

That counter is the cheapest wedge check there is: it is monotonic per process,
so a restart resets it, and a stall shows up as it ceasing to advance while
`vllm:num_requests_running` stays non-zero.

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

### The plugin's server-facing tests

`vllm-tt-plugin` ships `tests/tt`, which drive a running server over
`/v1/completions`. They apply here -- the ASR model answers that endpoint too,
even though its text output is meaningless -- and they are the only coverage of
per-request sampling isolation on this deployment:

```
cd $VLLM_TT_PLUGIN
pytest tests/tt --tt-server-url=http://127.0.0.1:8110 \
  --tt-model-name=neosophie/Qwen3-ASR-1.7B-JA
```

Two presence-penalty cases fail, and it is not a plugin defect. `presence`
subtracts its value once, capped at 2.0 by the OpenAI schema, while `frequency`
subtracts value x occurrence count and `repetition` divides. Measured on this
model, the gap between the top and second token is 3.5-5.8 nats at every step:

```
step 1 top3: [(' b', -0.08), (' a', -5.58), (' c', -5.83)]
step 3 top3: [(' a', -0.50), (' ',  -4.00), ('\n', -4.25)]
```

So -2.0 cannot reorder the top two, and the greedy output is identical for
presence_penalty 0.0 and 2.0 -- while frequency_penalty 2.0 does change it once
a token has repeated three times (-6.0 > gap), and repetition_penalty 2.0
changes it immediately. The tests assert that different presence penalties give
different text, which needs a flatter logit distribution than this model has.

Run them with `--deselect` on those two if a clean run is wanted:

```
pytest tests/tt --tt-server-url=http://127.0.0.1:8110 \
  --tt-model-name=neosophie/Qwen3-ASR-1.7B-JA \
  --deselect tests/tt/test_tt_penalties.py::TestPresencePenalty::test_different_presence_penalties \
  --deselect tests/tt/test_tt_penalties.py::TestPresencePenalty::test_presence_penalty_mixed_batch
```

That gives **71 passed, 1 skipped, 2 deselected** (~15 min). The skip is
`test_all_vocab_logprobs`, which asks for `top_logprobs=-1`; the server answers

```
Requested sample logprobs of 151936, which is greater than max allowed: 20
```

`max_logprobs` defaults to 20 in vLLM and the spec does not raise it. That is
deliberate: whole-vocabulary logprobs are a debugging aid with a per-token cost
proportional to the vocabulary, and nothing in the transcription path asks for
them. The test skips itself on exactly this error rather than failing.

### 4. Eval and benchmark

The upstream audio harnesses do not fit this model. `run_audio_eval` /
`run_audio_benchmark` in `test_module/` branch on `_is_whisper(ctx)`; everything
else falls to a generic path that POSTs a JSON body
(`{"file": "<base64>", ...}`) and, per `llm_module/eval_command.py`, omits the
`/v1` prefix because "audio models use tt-media-server". Both halves of that
shape fail here, and they fail differently -- measured against the running
server:

| what the upstream path sends | result |
|---|---|
| JSON body to `/audio/transcriptions` (no `/v1`, as `eval_command` builds it) | **HTTP 404** `{"detail":"Not Found"}` |
| JSON body to `/v1/audio/transcriptions` (path corrected by hand) | **HTTP 400**, `body.file` missing |

```
{"error":{"message":"1 validation error:
  {'type':'missing','loc':('body','file'),'msg':'Field required', ...}}}
```

So fixing the prefix alone is not enough: the route then exists but rejects the
body, because vLLM parses `file` as an upload rather than as a base64 string.
Note the 400's `input` echo lists the transcription defaults (`response_format`,
`temperature`, ...) with no `file` -- the JSON keys were dropped entirely, not
mistyped.

vLLM's OpenAI-compatible `/v1/audio/transcriptions` takes multipart/form-data.
Making the upstream harness speak it is a feature addition to that harness, not
part of this bring-up, so accuracy and throughput are measured with the two
scripts carried here instead.

Corpus accuracy (character error rate) — TED and MagicHub manifests, `conc=4`:

```
python3 reference_config/evals/asr_ja_eval.py --host http://127.0.0.1:8110 \
  --model neosophie/Qwen3-ASR-1.7B-JA \
  --manifest <corpus>/manifest.jsonl --concurrency 4 --output ted.json
```

**Discard the first corpus run after a restart, and run one measurement at a
time.** `--concurrency 4` matches `max_num_seqs`, so the queue is already
saturated: a second client makes requests wait, and the first run on a fresh
server also pays for kernel compilation and cache warm-up on top. Either way
some clips cross the eval's 120 s timeout, which inflates `fail` without moving
`corpus_cer` -- that is computed over the clips that did return.

Measured on TED against one server, back to back:

| run | ok / fail | p50 | p99 | CER |
|---|---|---|---|---|
| first after `/health` turns 200 | 490 / 19 | 3.363 s | **16.637 s** | 0.1000 |
| second, same server | 494 / 15 | 3.230 s | 8.847 s | **0.1002** |

The p99 is what moves; the extra 4 failures are its tail crossing the timeout.
Reaching the steady 15 needs no restart, just a second pass.

The first run *can* land on the steady numbers, so do not read 19 as the
expected first result. On a later restart the first pass came in at 494 / 15,
CER 0.1002, p99 4.924 s -- because a single golden clip had been transcribed
first, which is enough to pay the JIT compilation the first corpus run
otherwise absorbs. So:

- warm the server with one request (any clip) before measuring, or
- discard the first corpus pass.

Either works; what does not work is trusting the first pass on a server whose
very first request is a corpus clip.

On TED those 15 are expected and are not a model result. They are manifest
artifacts -- zero-length wavs, which the server rejects:

```
$ grep -o 'ERROR: [^"]*' samples.jsonl | sort | uniq -c
     15 ERROR: HTTP Error 400: Bad Request
# each one: frames 0, dur 0.0
```

So the accepted TED figure reads "494 ok / 15 download artifacts".

#### Where the two manifests come from

Neither corpus can be redistributed, so the manifests are built locally. A
manifest is JSON Lines, one clip per line, absolute wav path plus reference
text:

```
{"id": "-6K2nN9aWsg-00002686-00002940",
 "wav": "/path/ted_eval/clips/-6K2nN9aWsg-00002686-00002940.wav",
 "ref": "今大学教員をやってるんですけど"}
```

- **TED — 509 clips, monologue.** TEDxJP-10K (laboroai) ships no audio; it is
  reconstructed from YouTube with the project's own `compose_tedxjp10k.py`
  (v1.1, `utt_id_table.csv` + `diffs`), which emits Kaldi `text` and
  `segments`. Cut each segment out of the 16 kHz mono source (`sox`) and pair
  it with its `text` line. 14 talks yield 509 clips (mean 3.24 s, max 9.92 s
  measured over the manifest actually used).
  Note the download needs an IP YouTube will serve; the reconstruction itself
  is offline. 15 of the 509 fail to decode and are reported as `fail`, which is
  why the accepted figure reads "494 ok".
- **MagicHub — 600 clips, spontaneous conversation.**
  `MagicHub/Japanese_Spontaneous_Conversation_Training_Dataset` (ungated on the
  Hub) records each speaker on their own microphone, with `TXT/` giving
  `[start,end] speaker gender transcript`. Take each speaker's own turns from
  their own channel — no diarization involved — 30 turns per channel over 20
  channels, sampled with seed 42 from the 5920 candidates, keeping 0.4–30 s
  turns and dropping any turn carrying `[*]`, `[PII]` or `[NPS]`. The resulting
  600 clips run 0.42–14.98 s, mean 3.21 s.

A conversational corpus recorded as one mixed track is not usable here: CABank
Sakura was tried and its per-clip CER came out at 2.0, because a single
speaker's time window still contains the others' overlapping speech while the
reference holds only that speaker's line. That is a property of the corpus, not
of the model.

Throughput (LibriSpeech, downloaded by the script):

```
python3 reference_config/benchmarking/asr_openai_benchmark.py \
  --host http://127.0.0.1:8110 --model neosophie/Qwen3-ASR-1.7B-JA \
  --samples 32 --num-requests 128 --concurrency 4 --output bench.json
```

It fetches the clips from `datasets-server.huggingface.co` on every run, before
it touches the server, so a network hiccup shows up as a traceback rather than
a result:

```
Fetching HF dataset metadata: https://datasets-server.huggingface.co/rows?...
TimeoutError: The read operation timed out
```

That is the download, not the model -- no request reached port 8110. Rerun it.

Serving-level timings — TTFT, prefill, decode TPS and TPS/user — come from two
more probes, both driving one fixed clip so the token count per request does
not move between runs. Arguments are positional:
`<host> <model> <wav> <requests> <concurrency> <max_tokens>`.

```
# non-streaming: reads vllm:* counters from /metrics before and after
python3 reference_config/benchmarking/asr_perf_probe.py \
  http://127.0.0.1:8110 neosophie/Qwen3-ASR-1.7B-JA clip.wav 60 4 100

# streaming: client-side, first SSE chunk = TTFT, inter-chunk gap = TPOT
python3 reference_config/benchmarking/asr_perf_stream.py \
  http://127.0.0.1:8110 neosophie/Qwen3-ASR-1.7B-JA clip.wav 60 4 100
```

The non-streaming probe exists because the customer's client sets
`stream=false`, so per-token timings are not observable from the client and
have to be read off the server's own counters. The streaming probe measures the
same quantities the ordinary way and is the cross-check on them.

Measured on the delivery p150 with the image above:

| | value |
|---|---|
| TED 509 clips | CER 0.1002, 494 ok / 15 download artifacts |
| MagicHub 600 clips | CER 0.1668, 600 ok |
| LibriSpeech 128 req | 128 ok, rtfx ~12.5, p50 ~2.1 s |

And the serving-level timings the two probes report, 60 requests at
`concurrency 4` on the FLEURS clip, so the cross-check can be judged:

| | non-streaming (`/metrics`) | streaming (client-side) |
|---|---|---|
| requests ok | 60 / 60 | 60 / 60 |
| mean TTFT | 1.243 s | 1.338 s |
| mean E2E | 2.245 s | 2.326 s |
| decode TPS/user | 23.91 | 22.22 |
| decode TPS aggregate | 41.45 | 39.52 |
| tokens per request | 24.0 | 23.0 |

The two agree to within a few percent, which is the point of running both: the
streaming column is measured the ordinary way and corroborates counters the
customer's `stream=false` client cannot observe. It reads slightly slower
because the client sees SSE framing and scheduling delay on top of what the
counters attribute to decode, and it counts one fewer token per request (the
final chunk carries no new token). Treat a gap of tens of percent, or the two
columns moving in opposite directions, as a regression worth chasing.

The same CER, to four decimal places, has come out of every build of this model
so far -- across the vLLM 0.24->0.26 upgrade, three separate image builds at the
earlier pins, a device reset, and the current pins. The ~1 % spread in rtfx is
session-to-session drift on this board.

**How the current image was built.** The pinned commits are not on the forks
yet, so `git clone https://github.com/nyoshifujiTT/...` cannot reach them. This
image was built with both clone URLs pointed at a local `git daemon` instead;
everything else -- the Bake base, the patch, the build command -- is exactly
what is written above. Once the branches are pushed, the build has to be
repeated with the fork URLs unchanged, which is the only step of this runbook
that has not been executed as written at these pins. Earlier pins were
reproduced that way (fork clone, and fork clone with the base rebuilt from
Bake), and produced the same CER.

## Install
```
sudo cp $TT_INFERENCE_SERVER/scripts/qwen3_asr/qwen3asr-supervisor.service \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3asr-supervisor.service
```

### What the supervisor reads from the environment

Every path it uses is overridable, and it checks them before launching -- a
missing one exits with `supervisor: missing path: <p>` rather than failing
later inside `run.py`. Defaults are under the service user's home:

| variable | default | note |
|---|---|---|
| `TTIS` | `$HOME/tt-inference-server` | this repo. **Not** `TT_INFERENCE_SERVER`: that name is this runbook's, and the script predates it |
| `TT_METAL_HOME` | `$HOME/tt-metal` | same variable the runbook exports |
| `VENV` | `$TT_METAL_HOME/python_env` | the interpreter `run.py` is launched with |
| `SNAP` | the HF hub snapshot of `neosophie/Qwen3-ASR-1.7B-JA`, **by revision SHA** (`987bda16...`) | passed as `MODEL_WEIGHTS_DIR`. The SHA is literal, not globbed, so a re-download at a newer revision makes the path missing and the guard exits before launch -- override `SNAP` then, rather than editing the script |
| `MODEL_NAME` | `Qwen3-ASR-1.7B-JA` | used for both the launch and the canary, so they cannot drift apart |
| `CANARY_WAV` | `$HOME/real_ja.wav` | the clip from "The clip to check with" above -- verified the same file, md5 `3d43ec3ac2562231ec7c8c9ce4087ba4`. The canary only checks for HTTP 200 and a `"text"` field, so it does not depend on the transcript, but keeping it to that clip means a wrong answer is visible by eye in the log |
| `TTSMI` | `$(command -v tt-smi)`, else `$HOME/ttvenv/bin/tt-smi` | |
| `LOG` | `$HOME/asr_supervisor.log` | |

The unit file sets `TTIS` and nothing else, because the rest resolve correctly
for a service running as `ubuntu` on this host. Override there when they do
not.

### What has and has not been verified

An end-to-end recovery was demonstrated once on the **original** board: induced
load wedged it, the supervisor power-cycled, systemd restarted it on boot, and
transcription came back without human intervention.

That run predates the upstream merge, and auditing the script afterwards found
five defects that would each have broken it on the delivery host — stale run.py
flags, `/data`-only paths, a `tt-smi` path that does not exist here, a wedge
test grepping for a string this `tt-smi` never prints, and a startup budget
shorter than the 7–12 minute startup. They are fixed, and each piece is
exercised against the live host:

| checked | result |
|---|---|
| `run.py` invocation | resolves `Qwen3-ASR-1.7B-JA`, no argument errors |
| `TTSMI` resolution | `/home/ubuntu/ttvenv/bin/tt-smi`, executable |
| `device_ok` | reports the healthy board |
| `in_container` | spares the engine of a running `--docker-server` |
| `canary_ok` | 200 + text against the live server |

What is **not** re-verified is the full wedge → power-cycle → reboot → recover
loop on this host, because that needs a wedged board and this one has not
reproduced the hang across the 900-request decode-trace-on soaks above. Treat
the recovery path as reviewed and unit-exercised, not as re-demonstrated.
