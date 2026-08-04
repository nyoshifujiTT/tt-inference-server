# Qwen3-ASR TT p150 server supervisor

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
