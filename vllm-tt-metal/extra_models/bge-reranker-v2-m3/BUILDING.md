# Building a bge-reranker-v2-m3 image from an unmerged branch

The reranker model code lives in tt-metal and the serving glue lives in
vllm-tt-plugin. While either is still on a branch that upstream has not taken,
the image has to be built from that branch. Do it the same way every other model
does (see tt-inference-server PR #4837): apply a local patch, build, revert.

Never commit the fork URLs or the fork commit pins. An image tag is only worth
something if it names revisions that exist upstream, and a committed fork pin
silently makes every later build reproduce a personal branch.

Equally, never bind-mount model sources over a prebuilt image to pick up local
changes. The image bakes in a whole tt-metal tree, so overlaying code from
another revision mixes it with a C++ runtime and Python packages built from a
different one: the run proves nothing about the image, and the tag stops
describing what actually ran.

## 1. Patch

Pin the commits by **full 40-character SHA**. The Dockerfile shallow-clones the
default branch and then runs `git fetch --depth 1 origin <pin>`; GitHub serves
any full SHA that way, including one that only exists on a topic branch, but it
cannot resolve an abbreviated one because that is not a ref:

```
$ git fetch --depth 1 origin ad499a943ab            # abbreviated
fatal: couldn't find remote ref ad499a943ab
$ git fetch --depth 1 origin ad499a943ab9273c...    # full
 * branch  ad499a943ab9273c... -> FETCH_HEAD
```

Most pins in the prod catalog are abbreviated, which works only because those
commits sit on the default branch and are already in the shallow clone. A
branch that is not upstream yet has no such luck, so use the full SHA.

```bash
git apply <<'PATCH'
diff --git a/vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile b/vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile
index dafb609aa..2f51cb793 100644
--- a/vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile
+++ b/vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile
@@ -86,7 +86,7 @@ ENV UV_HTTP_RETRIES=10
 # A full-history clone of tt-metal has taken over an hour on CI, connection dropped ("fatal: early
 # EOF"). Only the pinned commit is needed, so fetch just that (matches the shallow
 # clone already used by tt-media-server/Dockerfile).
-RUN /bin/bash -c "git clone --depth 1 https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} \
+RUN /bin/bash -c "git clone --depth 1 https://github.com/<you>/tt-metal.git ${TT_METAL_HOME} \
     && cd ${TT_METAL_HOME} \
     && git fetch --depth 1 origin ${TT_METAL_COMMIT_SHA_OR_TAG} \
     && git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} \
@@ -101,7 +101,7 @@ RUN /bin/bash -c "git clone --depth 1 https://github.com/tenstorrent-metal/tt-me
 # Build vllm-tt-plugin - clone with minimal history and clean.
 # The plugin owns the vLLM version pin and its dependency overrides, so the
 # install is delegated to its own docs/install-vllm-tt.sh rather than restated here
-RUN /bin/bash -c "git clone https://github.com/tenstorrent/vllm-tt-plugin.git ${vllm_tt_plugin_dir} \
+RUN /bin/bash -c "git clone https://github.com/<you>/vllm-tt-plugin.git ${vllm_tt_plugin_dir} \
     && cd ${vllm_tt_plugin_dir} \
     && git checkout ${TT_VLLM_COMMIT_SHA_OR_TAG} \
     && source ${PYTHON_ENV_DIR}/bin/activate \
diff --git a/workflows/model_specs/prod/embedding.yaml b/workflows/model_specs/prod/embedding.yaml
index dbd01d8cb..a8ff370f5 100644
--- a/workflows/model_specs/prod/embedding.yaml
+++ b/workflows/model_specs/prod/embedding.yaml
@@ -258,8 +258,8 @@ templates:
 # both with git checkout -- never commit the fork pins.
 - weights:
     - BAAI/bge-reranker-v2-m3
-  tt_metal_commit: "f32a181affd"
-  vllm_commit: "e3fc849"
+  tt_metal_commit: "<full 40-char tt-metal sha>"
+  vllm_commit: "<full 40-char vllm-tt-plugin sha>"
   version: "0.20.0"
   impl: tt_vllm_plugin
   min_disk_gb: 15
PATCH
```

Both branches must be pushed: the Dockerfile clones them from GitHub by SHA, so
a local-only commit cannot end up in the image.

## 2. Build

```bash
python3 scripts/build_docker_images.py --build-metal-commit <tt-metal sha>
```

This produces
`ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:<VERSION>-<tt_metal>-<vllm>`.

## 3. Revert

```bash
git checkout vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile \
             workflows/model_specs/prod/embedding.yaml
```

## 4. Serve

```bash
python3 run.py --model bge-reranker-v2-m3 --workflow server --tt-device p150 \
  --docker-server --no-auth --service-port 8010 \
  --override-docker-image <tag from step 2>
```

Do not pass `--dev-mode`: it overlays host sources onto the image, so the run
would no longer validate the image being tested.

Check with:

```bash
curl -s localhost:8010/rerank -H 'Content-Type: application/json' -d '{
  "model": "BAAI/bge-reranker-v2-m3",
  "query": "what is tenstorrent",
  "documents": ["Tenstorrent builds AI processors.", "Paris is in France."]}'
```

The relevant document must score far above the irrelevant one.
