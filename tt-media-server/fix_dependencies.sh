# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# Remove packages that might contain CUDA
uv pip uninstall xformers diffusers torch torchvision torchaudio

# Install CPU-only versions.
# The floors keep pyannote-audio 4.x (required by the speaker-diarization
# service) satisfiable: it needs torchaudio>=2.8.0, and torch/torchvision have
# to move with it or their C extensions fail to load. Without them this step
# silently reinstalls the older CPU wheels and undoes the resolve above.
uv pip install "torch>=2.8.0" "torchvision>=0.23.0" "torchaudio>=2.8.0" --index-url https://download.pytorch.org/whl/cpu

# Install xformers without its CUDA sub-deps (--no-deps avoids re-pulling
# the CUDA torch wheels we just replaced with CPU-only ones above).
uv pip install xformers --no-deps

uv pip install diffusers
