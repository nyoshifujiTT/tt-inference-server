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
TTIS="/data/repo/tt-inference-server"
TT_METAL_HOME="/data/wt/qwen3asr"
VLLM_DIR="/data/vllm_tt/vllm"
VENV="/data/wt/qwen3asr/python_env"
SNAP="/data/qwen3asr_hf/hub/models--neosophie--Qwen3-ASR-1.7B-JA/snapshots/987bda160f2dabfa6757550bcff7cdda2ba0648c"
CANARY_WAV="/data/ja_words.wav"
TTSMI="/home/ubuntu/ttsmi-venv/bin/tt-smi"
LOG="/data/vllm_tt/asr_supervisor.log"
SERVER_LOG_DIR="${TTIS}/workflow_logs/local_server"

log() { echo "$(date -u +%FT%TZ) [supervisor] $*" | tee -a "$LOG"; }

[ -f ~/.codex/hf.env ] && set -a && . ~/.codex/hf.env && set +a

recover_device() {
  log "recovering device (tt-smi -r) ..."
  sudo "$TTSMI" -r >/dev/null 2>&1
  sleep 3
  if sudo "$TTSMI" -s 2>&1 | grep -q "should be reset"; then
    log "tt-smi -r insufficient; ipmitool chassis power cycle (host will reboot)"
    sudo ipmitool chassis power cycle >/dev/null 2>&1
    # wait for host to come back
    for i in $(seq 1 40); do
      sleep 30
      if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no localhost true >/dev/null 2>&1 || true; then :; fi
      if [ -e /dev/tenstorrent/2 ] && ! (sudo "$TTSMI" -s 2>&1 | grep -q "should be reset"); then
        log "device back after power cycle"
        break
      fi
    done
  fi
  sudo chmod 666 /dev/tenstorrent/* 2>/dev/null
}

launch_server() {
  pkill -f "run_vllm_api_server.py" 2>/dev/null
  pkill -f "run.py --model Qwen3-ASR" 2>/dev/null
  sleep 3
  sudo chmod 666 /dev/tenstorrent/* 2>/dev/null
  log "launching run.py --local-server on port $PORT"
  ( cd "$TTIS" && \
    HF_TOKEN="${HF_TOKEN:-}" MODEL_WEIGHTS_DIR="$SNAP" \
    nohup "$VENV/bin/python" run.py \
      --model Qwen3-ASR-1.7B --device p150 --workflow server --local-server \
      --tt-metal-home "$TT_METAL_HOME" \
      --tt-metal-python-venv-dir "$VENV" \
      --vllm-dir "$VLLM_DIR" \
      --service-port "$PORT" --no-auth --skip-system-sw-validation --dev-mode \
      > /tmp/asr_supervisor_run.log 2>&1 & )
}

wait_healthy() {
  # up to ~5 min for startup (model load + decode warmup)
  for i in $(seq 1 60); do
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
        -F "file=@${CANARY_WAV}" -F "model=Qwen/Qwen3-ASR-1.7B" -F language=ja \
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
