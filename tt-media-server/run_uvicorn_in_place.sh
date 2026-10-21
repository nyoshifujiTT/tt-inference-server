#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# Start the media server from wherever this tree was copied, for launchers that
# exec a command line rather than sourcing run_uvicorn.sh from the server dir.
#
# The server resolves several paths against its working directory -- the
# /static mount, pydantic-settings' .env, the job database -- and imports its
# packages as top-level modules, so it has to run from this directory. The
# interpreter is whichever one is first on PATH: the launcher owns the venv.
#
#   run_uvicorn_in_place.sh --host <host> --port <port> [uvicorn args...]
set -eo pipefail

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
exec python -m uvicorn main:app --lifespan on "$@"
