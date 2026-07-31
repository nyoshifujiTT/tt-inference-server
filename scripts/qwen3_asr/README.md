# Qwen3-ASR TT p150 server supervisor

The tt-metal decode path has a known **non-deterministic device hang** that can
wedge the board under sustained load (see tt-metal issues #40592, #45052,
#4752 — same class of SDPA/decode ND-hang observed for Mistral / gpt-oss /
Falcon3). It is a platform-level bug, not specific to the Qwen3-ASR adapter:
memory is flat (no leak) and single-device CCL/fabric is short-circuited, and a
watchdog (`TT_METAL_OPERATION_TIMEOUT_SECONDS`) shows the stall happens in a
decode device op (`device timeout, potential hang detected, unrecoverable`).

Until the tt-metal fix lands (or tt-metal is pinned to a pre-regression commit),
`asr_supervisor.sh` keeps the ASR server available in production the same way
the Qwen3-Embedding fullbench supervisor does on this hardware:

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
correct transcript again — with no human intervention.
