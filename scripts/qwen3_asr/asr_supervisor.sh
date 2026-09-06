#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Qwen3-ASR server supervisor for TT p150.
#
# The tt-metal decode path has a non-deterministic device hang (see worklog /
# tt-metal #40592, #45052, #4752). It was reproducible on the original board
# (10.160.20.103) but has NOT reproduced on the delivery p150 across 900-request
# decode-trace-on soaks, so it no longer gates the default (see
# scripts/qwen3_asr/README.md). This supervisor is kept as defense-in-depth so
# the ASR server survives any residual wedge in production: it (re)launches the
# standard run.py --local-server, watches liveness with a lightweight canary
# transcription, and on a wedge recovers the device (tt-smi -r, then ipmitool
# power cycle as a fallback) and relaunches -- the same pattern used for the
# Qwen3-Embedding fullbench supervisor on this hardware.
#
# Usage: asr_supervisor.sh [PORT]
set -u

PORT="${1:-8101}"

# Every path is overridable. The defaults below are for the original bring-up
# board, which kept everything under /data; that directory does not exist on
# the delivery host, so hardcoding it made this script a no-op there. Set the
# variables (or edit the defaults) to match the host this runs on.
TTIS="${TTIS:-$HOME/tt-inference-server}"
TT_METAL_HOME="${TT_METAL_HOME:-$HOME/tt-metal}"
VENV="${VENV:-$TT_METAL_HOME/python_env}"
# Where the weights are. This is passed as --host-weights-dir with
# MODEL_SOURCE=local, which is what makes run.py actually use THIS directory.
#
# Setting MODEL_WEIGHTS_DIR alone did nothing: setup_host.py only reads it on
# the `model_source == local` branch, and model_source defaults to
# `huggingface` (setup_host.py: os.getenv("MODEL_SOURCE", HUGGINGFACE)). With
# neither MODEL_SOURCE nor --host-weights-dir/--host-hf-cache passed, the
# revision pinned in this path was never the one that got served -- run.py
# resolved the repo through the HF cache and would happily start on a different
# snapshot.
SNAP="${SNAP:-$HOME/.cache/huggingface/hub/models--neosophie--Qwen3-ASR-1.7B-JA/snapshots/987bda160f2dabfa6757550bcff7cdda2ba0648c}"
MODEL_NAME="${MODEL_NAME:-Qwen3-ASR-1.7B-JA}"
# Liveness only asks "did a transcription come back", so any short clip works.
# Fetch it with the snippet in README.md ("The clip to check with") rather than
# pointing at a scratch file: the synthetic fixtures used during bring-up have
# no reference transcript and must not be mistaken for an accuracy check.
CANARY_WAV="${CANARY_WAV:-$HOME/real_ja.wav}"
TTSMI="${TTSMI:-$(command -v tt-smi || echo "$HOME/ttvenv/bin/tt-smi")}"
LOG="${LOG:-$HOME/asr_supervisor.log}"
SERVER_LOG_DIR="${TTIS}/workflow_logs/local_server"

for _p in "$TTIS" "$TT_METAL_HOME" "$VENV" "$SNAP" "$CANARY_WAV"; do
  [ -e "$_p" ] || { echo "supervisor: missing path: $_p" >&2; exit 1; }
done

log() { echo "$(date -u +%FT%TZ) [supervisor] $*" | tee -a "$LOG"; }

[ -f ~/.codex/hf.env ] && set -a && . ~/.codex/hf.env && set +a

# Is the device usable? tt-smi -s prints a JSON snapshot; if the board is wedged
# the call itself fails or reports no device. The older check grepped for
# "should be reset", a string this tt-smi build never emits, so the escalation
# below could never trigger.
device_ok() {
  sudo "$TTSMI" -s 2>/dev/null | grep -q "BOARD_ID_HIGH"
}

recover_device() {
  log "recovering device (tt-smi -r) ..."
  sudo "$TTSMI" -r >/dev/null 2>&1
  sleep 5
  if device_ok; then
    log "device recovered by tt-smi -r"
    sudo chmod 666 /dev/tenstorrent/* 2>/dev/null
    return 0
  fi

  log "tt-smi -r insufficient; ipmitool chassis power cycle (host will reboot)"
  sudo ipmitool chassis power cycle >/dev/null 2>&1
  # The host reboots under us, so this loop only matters if the power cycle was
  # refused. systemd restarts the supervisor after the reboot (see the unit).
  for _ in $(seq 1 40); do
    sleep 30
    # /dev/tenstorrent/0 is the p150; the old check looked for device 2, which
    # does not exist on a single-board host and so never became true.
    if [ -e /dev/tenstorrent/0 ] && device_ok; then
      log "device back after power cycle"
      break
    fi
  done
  sudo chmod 666 /dev/tenstorrent/* 2>/dev/null
}

# Kill a previous run and make sure nothing still holds the device.
#
# pkill on the run.py pattern only matches the parent: vLLM's engine runs as a
# separate "VLLM::EngineCore" process, which survives and keeps
# /dev/tenstorrent/* open. A later launch then hangs forever in "Starting
# devices in cluster", and tt-smi -r does not help because the orphan
# reacquires the device right after the reset.
# True when the pid belongs to a docker container rather than to a local-server
# run of ours. A --docker-server deployment's engine looks identical to ours in
# the process table, so without this check the supervisor would kill an
# unrelated containerised server.
in_container() {
  grep -qE '/docker-|/docker/' "/proc/$1/cgroup" 2>/dev/null
}

# Pids holding a Tenstorrent device, found by walking /proc/*/fd rather than by
# asking lsof for a path.
#
# `lsof -t /dev/tenstorrent/*` matches on the host's device node, and a
# container gets its own node for the same chip. Measured on this host with a
# --docker-server engine running: the host node is dev=5 inode=666, the fd
# inside the container resolves to dev=67 inode=13, and
# `lsof -t /dev/tenstorrent/*` prints nothing at all while
# `/proc/<pid>/fd/17 -> /dev/tenstorrent/0` is plainly open. So the path form
# reports "device free" for exactly the holder that will make the next launch
# hang in "Starting devices in cluster".
device_holders() {
  local fd pid
  for fd in /proc/[0-9]*/fd; do
    pid="${fd#/proc/}"; pid="${pid%/fd}"
    if sudo readlink "$fd"/* 2>/dev/null | grep -q '^/dev/tenstorrent/'; then
      echo "$pid"
    fi
  done | sort -un
}

stop_server() {
  pkill -f "run.py --model Qwen3-ASR" 2>/dev/null
  pkill -f "run_vllm_api_server.py" 2>/dev/null
  # Not a bare pkill: the same pattern matches a containerised server's engine.
  for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null); do
    in_container "$pid" || kill "$pid" 2>/dev/null
  done
  sleep 3

  # Whatever still holds the device blocks the next launch: tt-metal then hangs
  # in "Starting devices in cluster", and tt-smi -r does not help because the
  # holder reacquires it. Escalate, but again only for processes we own.
  local holders stubborn
  holders=$(device_holders)
  stubborn=""
  for pid in $holders; do
    if in_container "$pid"; then
      # Not ours to kill, but the launch below cannot succeed while it holds
      # the chip, so say so instead of hanging without explanation.
      log "device held by containerised pid $pid -- leaving it alone; a local launch will not get the device"
    else
      stubborn="$stubborn $pid"
    fi
  done
  if [ -n "$stubborn" ]; then
    log "device still held by:$stubborn -- sending SIGKILL"
    # shellcheck disable=SC2086
    kill -9 $stubborn 2>/dev/null
    sleep 3
  fi
}

launch_server() {
  stop_server
  sudo chmod 666 /dev/tenstorrent/* 2>/dev/null
  log "launching run.py --local-server on port $PORT"
  # MODEL_SPECS_ENV=dev: the spec lives only in the dev catalog (prod entries are
  # release artifacts written by promote_dev_spec_to_prod.py), and run.py defaults
  # to prod, where this model does not exist.
  # No --vllm-dir: run.py reports it as deprecated and ignored -- vLLM is an
  # ordinary package in the tt-metal venv and the TT platform comes from
  # vllm-tt-plugin.
  ( cd "$TTIS" && \
    MODEL_SPECS_ENV=dev HF_TOKEN="${HF_TOKEN:-}" \
    MODEL_SOURCE=local MODEL_WEIGHTS_DIR="$SNAP" \
    nohup "$VENV/bin/python" run.py \
      --model "$MODEL_NAME" --tt-device p150 --workflow server --local-server \
      --tt-metal-home "$TT_METAL_HOME" \
      --tt-metal-python-venv-dir "$VENV" \
      --host-weights-dir "$SNAP" \
      --service-port "$PORT" --no-auth --skip-system-sw-validation --dev-mode \
      > /tmp/asr_supervisor_run.log 2>&1 & )
}

wait_healthy() {
  # Startup is slow: the adapter loads weights and captures the decode trace
  # before serving. Measured on the delivery p150 at 7-12 minutes (460 s on one
  # timed restart), so the old 5-minute budget expired every time -- the
  # supervisor would declare the server dead and start power-cycling a board
  # that was merely still warming up. Allow 20 minutes.
  local deadline=$(( $(date +%s) + 20 * 60 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    sleep 5
    if curl -s -m 5 "http://127.0.0.1:${PORT}/health" -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q 200; then
      # confirm a model is actually served
      if curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q Qwen3-ASR; then
        log "server healthy on port $PORT"
        return 0
      fi
    fi
  done
  log "server did not become healthy in time"
  return 1
}

canary_ok() {
  # lightweight liveness: a bounded transcription must return 200 with text
  local out
  out=$(curl -s -m 45 "http://127.0.0.1:${PORT}/v1/audio/transcriptions" \
        -F "file=@${CANARY_WAV}" -F "model=neosophie/${MODEL_NAME}" -F language=ja \
        -w '\n%{http_code}' 2>/dev/null)
  local code="${out##*$'\n'}"
  [ "$code" = "200" ] && echo "$out" | grep -q '"text"'
}

log "=== supervisor start (port $PORT) ==="
while true; do
  launch_server
  if ! wait_healthy; then
    recover_device
    continue
  fi
  # monitor loop
  fails=0
  while true; do
    sleep 20
    if canary_ok; then
      fails=0
    else
      fails=$((fails+1))
      log "canary failed ($fails)"
      if [ "$fails" -ge 2 ]; then
        log "server wedged; recovering + relaunching"
        recover_device
        break
      fi
    fi
  done
done
