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

# Make the chip nodes writable by the service user after a reset.
#
# Not `chmod 666 /dev/tenstorrent/*`: that glob also matches the by-id/
# directory udev creates for the stable `blackhole-<asic_id>` symlinks, and
# 666 on a directory drops its execute bit, so nothing non-root can traverse
# it any more. Measured on this host after the supervisor had run --
# /dev/tenstorrent/by-id was drw-rw-rw- with a ctime matching the supervisor's
# log, and `stat /dev/tenstorrent/by-id/*` returned Permission denied. It stays
# broken until udev recreates it at the next boot. udev's own rule
# (`SUBSYSTEM=="tenstorrent", MODE="0666"`) applies to the device nodes only,
# which is exactly the scope wanted here.
relax_device_perms() {
  local node
  for node in /dev/tenstorrent/*; do
    [ -c "$node" ] && sudo chmod 666 "$node" 2>/dev/null
  done
}

recover_device() {
  # Never reset a chip a container is serving on.
  #
  # The pre-launch guard keeps us out of that state, but recover_device is also
  # reached from the monitor loop, where a container could have been started
  # underneath us since. `tt-smi -r` does not care who holds the device: it
  # would reset the board out from under a deployment that is answering
  # requests, and device_ok would then report success because the board reads
  # fine. Bail out and let the main loop's guard wait for the container
  # instead.
  local held pid
  held=""
  for pid in $(device_holders); do
    in_container "$pid" && held="$held $pid"
  done
  if [ -n "$held" ]; then
    log "not recovering: device held by containerised pid(s):$held -- \
tt-smi -r would reset the chip that deployment is serving on"
    return 1
  fi

  log "recovering device (tt-smi -r) ..."
  sudo "$TTSMI" -r >/dev/null 2>&1
  sleep 5
  if device_ok; then
    log "device recovered by tt-smi -r"
    relax_device_perms
    return 0
  fi

  # Escalate to a power cycle -- but only report what actually happened.
  #
  # This host has no BMC (`ipmitool chassis power status` ->
  # "Could not open device at /dev/ipmi0 ..."), and the old form discarded both
  # the output and the status with `>/dev/null 2>&1`. It then entered a wait
  # loop whose condition -- /dev/tenstorrent/0 exists and device_ok -- is
  # already true here, so 30 s later it logged "device back after power cycle"
  # having neither power-cycled nor done anything beyond the tt-smi -r above.
  # Measured: the whole tail returned in 30 s with that line in the log. An
  # operator reading it would conclude the board had been recovered.
  local ipmi_err
  log "tt-smi -r insufficient; escalating to ipmitool chassis power cycle"
  if ! ipmi_err=$(sudo ipmitool chassis power cycle 2>&1); then
    log "power cycle UNAVAILABLE: ${ipmi_err:-ipmitool failed}"
    log "device is still wedged and this host cannot recover it automatically \
-- a human has to power-cycle it (see scripts/qwen3_asr/README.md)"
    relax_device_perms
    return 1
  fi

  # The host reboots under us, so this loop only matters if the power cycle was
  # accepted but deferred. systemd restarts the supervisor after the reboot
  # (see the unit).
  local came_back=1
  for _ in $(seq 1 40); do
    sleep 30
    # /dev/tenstorrent/0 is the p150; the old check looked for device 2, which
    # does not exist on a single-board host and so never became true.
    if [ -e /dev/tenstorrent/0 ] && device_ok; then
      log "device back after power cycle"
      came_back=0
      break
    fi
  done
  [ "$came_back" = 0 ] || log "device did not come back within 20 minutes of the power cycle"
  relax_device_perms
  return "$came_back"
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

# Kill our own processes matching a pattern, never a containerised one.
#
# A bare `pkill -f` is wrong for every pattern here, not just for
# VLLM::EngineCore. Host /proc shows processes inside containers too, so
# `pgrep -f run_vllm_api_server.py` on a host running --docker-server returns
# the live production server's pid -- measured: pid 2714943, cgroup
# /system.slice/docker-9c2677b2....scope, identical to the container's
# .State.Pid. pkill would have killed it. The EngineCore loop below was given
# this guard after that mistake once already; the guard has to cover the whole
# of stop_server, or the accident just moves to another line.
kill_ours() {
  local pattern="$1" pid
  for pid in $(pgrep -f "$pattern" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    if in_container "$pid"; then
      log "leaving containerised pid $pid alone (matched: $pattern)"
    else
      kill "$pid" 2>/dev/null
    fi
  done
}

stop_server() {
  kill_ours "run.py --model Qwen3-ASR"
  kill_ours "run_vllm_api_server.py"
  kill_ours "VLLM::EngineCore"
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
  relax_device_perms
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
  # Liveness: a bounded transcription must return 200 with text.
  #
  # The timeout is an argument because the FIRST transcription after a launch
  # is not bounded by inference: tt-metal JIT-compiles kernels into the
  # container/host cache, measured at 6m25s-6m45s across fifteen runs on this
  # board. /health turns 200 long before that (route published 02:35:52, first
  # transcription done 02:45 = 9.1 min later), so a 45 s canary run straight
  # out of wait_healthy always times out, and two of those declare a wedge
  # after 2.2 minutes -- resetting the device mid-compile and starting over,
  # forever.
  local timeout="${1:-45}" out
  out=$(curl -s -m "$timeout" "http://127.0.0.1:${PORT}/v1/audio/transcriptions" \
        -F "file=@${CANARY_WAV}" -F "model=neosophie/${MODEL_NAME}" -F language=ja \
        -w '\n%{http_code}' 2>/dev/null)
  local code="${out##*$'\n'}"
  [ "$code" = "200" ] && echo "$out" | grep -q '"text"'
}

# One transcription with the compile budget, so the steady-state canary can
# stay short. README: "budget 7 minutes for it, and do not put a shorter
# --max-time on that request".
CANARY_FIRST_TIMEOUT="${CANARY_FIRST_TIMEOUT:-600}"

warm_first_transcription() {
  log "warming: first transcription JIT-compiles kernels (up to ${CANARY_FIRST_TIMEOUT}s)"
  if canary_ok "$CANARY_FIRST_TIMEOUT"; then
    log "warming: first transcription returned; steady-state canary starts now"
    return 0
  fi
  log "warming: first transcription did not return within ${CANARY_FIRST_TIMEOUT}s"
  return 1
}

log "=== supervisor start (port $PORT) ==="
while true; do
  # Do not launch while a container owns the chip.
  #
  # stop_server deliberately spares containerised processes, so the device is
  # still held when a --docker-server deployment is running. Launching anyway
  # does not fail cleanly: run.py hangs in "Starting devices in cluster",
  # wait_healthy burns its full 20 minutes (and is watching a different port
  # from the container's), and recover_device then runs `tt-smi -r` -- which
  # resets the chip out from under the deployment that is serving traffic. Since
  # device_ok only asks whether tt-smi can read the board, the reset "succeeds"
  # and the loop repeats: a working production server wedged every 20 minutes,
  # forever. Wait for the container to go away instead, and say so.
  while true; do
    held=""
    for pid in $(device_holders); do
      in_container "$pid" && held="$held $pid"
    done
    [ -z "$held" ] && break
    log "device held by containerised pid(s):$held -- not launching; \
waiting for that deployment to stop (do not run this supervisor beside a \
--docker-server server)"
    sleep 60
  done

  launch_server
  if ! wait_healthy; then
    # A refusal here (container appeared underneath us) is fine: continue
    # returns to the guard above, which waits for it to go.
    recover_device || true
    continue
  fi
  # /health being 200 does not mean a transcription can complete yet -- the
  # first one compiles kernels for ~6.5 min. Spend that here, with the long
  # budget, so the monitor's short canary is only ever asked about a server
  # that has already served once. Without this the monitor declares a wedge
  # 2.2 min in, every single launch.
  if ! warm_first_transcription; then
    recover_device || true
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
        recover_device || true
        break
      fi
    fi
  done
done
