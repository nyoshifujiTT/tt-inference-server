# Derives a plugin-enabled image from the prebuilt src-dev image by installing
# tt-vllm-plugin on top of the existing tt-metal + vLLM(fork) build. This lets
# us use the official plugin pooling path (TTPlatform + TTModelRunnerPooling +
# official Qwen3ForEmbedding) without rebuilding tt-metal and without pulling
# the media-server image (whose pull was chronically stalling).
#
# Build context must be the tt-inference-server repo root so that the
# tt-vllm-plugin/ directory is available to COPY.
#   docker build -f vllm-tt-metal/vllm.tt-metal.src.dev.ttplugin.Dockerfile \
#     -t <image>:0.10.0-555f240-22be241-ttplugin .
#
# --no-deps is used so the plugin does not drag in pypi vllm==0.10.1.1 and
# clobber the fork's editable vLLM install already present in the image.
# Base src-dev image to layer the plugin on. Defaults to the last published
# pin, but override with --build-arg BASE_IMAGE=<tag> to layer on a locally
# built src-dev image (e.g. one built from a fork via build_single_docker.sh).
ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.10.0-555f240-22be241
FROM ${BASE_IMAGE}
USER root
COPY tt-vllm-plugin ${TT_METAL_HOME}/tt-vllm-plugin
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && cd ${TT_METAL_HOME}/tt-vllm-plugin \
    && uv pip install --no-deps ."
