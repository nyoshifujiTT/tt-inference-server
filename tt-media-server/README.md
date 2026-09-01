# TT non-LLM inference server

This server is built to serve non-LLM models. Currently supported models:

1. SDXL-trace
2. SDXL-image-to-image
3. SDXL-edit
4. SD3.5
5. Flux1
6. Mochi1
7. Wan2.2
8. Motif-Image-6B-Preview
9. Qwen-Image
10. Whisper
11. Microsoft Resnet (Forge)
12. VLLM with TT Plugin
13. bge-large-en-v1.5
14. Qwen3-Embedding-8B

# Repo structure

1. Config - config files that can be overridden by environment variables.
2. Domain - Domain and transfer objects
3. Model services - Services for processing models, scheduler for models and a runner
4. Open_ai_api - controllers in OpenAI flavor
5. Resolver - creator of scheduler and model, depending on the config creates singleton instances of scheduler and model service
6. Security - Auth features
7. Tests - general end to end tests
8. Model runners - runners for devices and models. Runner_fabric is responsible for creating a needed runner

More details about each folder will be provided below

# API Versioning

All API endpoints use the `/v1` prefix to match the OpenAI API standard. Legacy paths without the `/v1` prefix are still supported during a deprecation period but will be removed after **2026-06-30**.

## Versioned vs legacy paths

| Primary (use this)                                          | Legacy (deprecated)                                  | Method | Description                                |
|-------------------------------------------------------------|------------------------------------------------------|--------|--------------------------------------------|
| `/v1/images/generations`                                    | `/image/generations`                                 | POST   | Text-to-image generation                   |
| `/v1/images/image-to-image`                                 | `/image/image-to-image`                              | POST   | Image-to-image (SDXL img2img)              |
| `/v1/images/edits`                                          | `/image/edits`                                       | POST   | Image editing (SDXL edit)                  |
| `/v1/audio/transcriptions`                                  | `/audio/transcriptions`                              | POST   | Speech-to-text                             |
| `/v1/audio/translations`                                    | `/audio/translations`                                | POST   | Speech-to-English (translation)            |
| `/v1/diarize`                                               | —                                                    | POST   | Speaker diarization job (who spoke when)   |
| `/v1/jobs/{jobId}`                                          | —                                                    | GET    | Diarization job status and result          |
| `/v1/media/input`                                           | —                                                    | POST   | Pre-signed upload url for `media://` audio |
| `/v1/audio/speech`                                          | `/audio/speech`                                      | POST   | Text-to-speech                             |
| `/v1/videos/generations`                                    | `/video/generations`                                 | POST   | Text-to-video generation                   |
| `/v1/videos/generations/i2v`                                | `/video/generations/i2v`                             | POST   | Image-to-video generation (Wan2.2 I2V)     |
| `/v1/videos/generations/{job_id}`                           | `/video/generations/{job_id}`                        | GET    | Get video job metadata                     |
| `/v1/videos/generations/{job_id}/download`                  | `/video/generations/{job_id}/download`               | GET    | Download generated video                   |
| `/v1/videos/generations/{job_id}/cancel`                    | `/video/generations/{job_id}/cancel`                 | POST   | Cancel video job and assets                |
| `/v1/videos/jobs`                                           | `/video/jobs`                                        | GET    | List all video jobs                        |
| `/v1/cnn/search-image`                                      | `/cnn/search-image`                                  | POST   | CNN image search                           |
| `/v1/fine_tuning/catalog`                                   | n/a                                                  | GET    | Available fine-tuning catalog              |
| `/v1/fine_tuning/jobs`                                      | n/a                                                  | POST   | Create fine-tuning job                     |
| `/v1/fine_tuning/jobs`                                      | n/a                                                  | GET    | List fine-tuning jobs                      |
| `/v1/fine_tuning/jobs/{job_id}`                             | n/a                                                  | GET    | Get fine-tuning job metadata               |
| `/v1/fine_tuning/jobs/{job_id}/cancel`                      | n/a                                                  | POST   | Cancel fine-tuning job                     |
| `/v1/fine_tuning/jobs/{job_id}/metrics`                     | n/a                                                  | GET    | Get training metrics                       |
| `/v1/fine_tuning/jobs/{job_id}/logs`                        | n/a                                                  | GET    | Get job logs                               |
| `/v1/fine_tuning/jobs/{job_id}/checkpoints`                 | n/a                                                  | GET    | List checkpoints for a job                 |
| `/v1/fine_tuning/jobs/{job_id}/checkpoints/{checkpoint_id}` | n/a                                                  | GET    | Download checkpoint as a zip               |
| `/v1/tokenize`                                              | `/tokenize`                                          | POST   | Tokenize text                              |
| `/v1/detokenize`                                            | `/detokenize`                                        | POST   | Detokenize token ids back to text          |

The following endpoints were already on `/v1` and have no legacy path:

| Endpoint                       | Method | Description                                                     |
|--------------------------------|--------|-----------------------------------------------------------------|
| `/v1/completions`              | POST   | OpenAI-compatible text completions (vLLM)                       |
| `/v1/chat/completions`         | POST   | OpenAI-compatible chat completions (vLLM)                       |
| `/v1/embeddings`               | POST   | OpenAI-compatible text embeddings                               |

## Deprecation headers

Requests to legacy paths return three extra HTTP headers per RFC 8594 and RFC 8288:

```
Deprecation: true
Sunset: 2026-06-30
Link: </v1/images/generations>; rel="successor-version"
```

- **`Deprecation: true`** -- signals the endpoint is deprecated.
- **`Sunset: 2026-06-30`** -- the date after which the legacy path will be removed.
- **`Link`** -- points to the replacement `/v1/...` endpoint.

## Maintenance and observability endpoints

These endpoints are always registered (regardless of `MODEL_SERVICE`) and do not use the `/v1` prefix:

| Endpoint            | Method | Description                                                                                  |
|---------------------|--------|----------------------------------------------------------------------------------------------|
| `/tt-liveness`      | GET    | Service liveness + model readiness payload (Tenstorrent-specific)                            |
| `/health`           | GET    | OpenAI/vLLM-compatible health check. 200 when ready, 503 when not                            |
| `/tt-deep-reset`    | POST   | Schedules a deep reset of the service and its model                                          |
| `/tt-reset-device`  | POST   | Schedules a reset for a single device. Requires `device_id` query parameter                  |
| `/metrics`          | GET    | Prometheus metrics (path configurable via `PROMETHEUS_ENDPOINT`)                              |
| `/docs`             | GET    | Swagger UI (only when `ENVIRONMENT=development`)                                             |
| `/redoc`            | GET    | ReDoc UI (only when `ENVIRONMENT=development`)                                               |
| `/openapi.json`     | GET    | OpenAPI schema (only when `ENVIRONMENT=development`)                                         |
| `/static/*`         | GET    | Static assets served from `tt-media-server/static/`                                           |

Examples:

```bash
# Liveness (no auth required)
curl 'http://127.0.0.1:8000/tt-liveness'

# Health (200 when model is ready, 503 otherwise)
curl -i 'http://127.0.0.1:8000/health'

# Schedule a deep reset of the service and its model
curl -X POST 'http://127.0.0.1:8000/tt-deep-reset'

# Reset a specific device by ID
curl -X POST 'http://127.0.0.1:8000/tt-reset-device?device_id=0'

# Prometheus metrics scrape
curl 'http://127.0.0.1:8000/metrics'
```

# Installation instructions

To just run a server build a docker file and run it.

For development running:

1. Setup tt-metal and all the needed variables for it
2. Make sure you're in tt-metal's python env
3. Clone tt-inference-server repo and switch to dev branch
4. ```sudo apt update && sudo apt install -y ffmpeg && uv pip install -r requirements.txt``` from tt-media-server
5. ```uvicorn main:app --lifespan on --port 8000``` (lifespan methods are needed to init device and close the devices)

## SDXL setup

### Standard SDXL Setup
1. ```export MODEL_RUNNER=tt-sdxl-trace```
2. Run the server ```uvicorn main:app --lifespan on --port 8000```

### SDXL with Tensor Parallelism (TP2)
1. ```export TP2=true```
2. ```export MODEL_RUNNER=tt-sdxl-trace```
3. Run the server ```source run_uvicorn.sh```

**Note:** TP2 configuration requires exactly 2 TT devices and is only supported for SDXL models.

### SDXL Image To Image Setup
1. ```export MODEL_RUNNER=tt-sdxl-image-to-image```
2. Run the server ```uvicorn main:app --lifespan on --port 8000```

### SDXL Edit Setup
1. ```export MODEL_RUNNER=tt-sdxl-edit```
2. Run the server ```uvicorn main:app --lifespan on --port 8000```


## SD-3.5 setup

Its easiest to use the [Special Environment Variable Overrides](#special-environment-variable-overrides) to help create the necessary setup for the target device.

### Standard SD-3.5 Setup
1. Set the model special env variable ```export MODEL=stable-diffusion-3.5-large```
2. Set device special env variable ```export DEVICE=galaxy``` or ```export DEVICE=t3k```
3. Run the server ```uvicorn main:app --lifespan on --port 8000```

### SD-3.5 with Custom Device Mesh Configurations

For optimized performance, you can use pre-configured device mesh setups:

#### Base Configuration (8 devices: 2x4 mesh)
```bash
export SD_3_5_BASE=true
export MODEL=stable-diffusion-3.5-large
export DEVICE=galaxy
source run_uvicorn.sh
```

#### Fast Configuration (32 devices: 4x8 mesh)
```bash
export SD_3_5_FAST=true
export MODEL=stable-diffusion-3.5-large
export DEVICE=galaxy
source run_uvicorn.sh
```

**Important Notes:**
- Base configuration requires 8 TT devices arranged in a 2x4 mesh
- Fast configuration requires 32 TT devices arranged in a 4x8 mesh
- Only Galaxy and T3K hardware with sufficient devices is supported
- Choose the configuration based on your hardware availability and performance requirements


## Supported DiT models
The setup for other supported DiT models is very similar to [Standard SD-3.5 Setup](#standard-sd-35-setup). Choose a configuration from the table below, and run the server.

| MODEL | Supported device options|
|-------|--------|
| stable-diffusion-3.5-large | galaxy, t3k |
| flux.1-dev | galaxy, t3k, p300, qbge |
| flux.1-schnell | galaxy, t3k, p300, qbge |
| motif-image-6b-preview | galaxy, t3k |
| qwen-image | galaxy, t3k |
| qwen-image-2512 | galaxy, t3k |
| mochi-1-preview | galaxy, t3k |
| Wan2.2-T2V-A14B-Diffusers | galaxy, t3k, qbge |
| Wan2.2-I2V-A14B-Diffusers | galaxy, t3k, p150x4, p150x8, p300x2 |

For example, to run flux.1-dev on t3k
1. Set the model special env variable e.g ```export MODEL=flux.1-dev```.
2. Set device special env variable e.g ```export DEVICE=t3k```.
3. Run the server ```uvicorn main:app --lifespan on --port 8000```.

## VLLM with TT Plugin Setup

The server supports running large language models using VLLM with the Tenstorrent plugin.

### Prerequisites

1. **Install the TT-VLLM Plugin**

   Follow the installation instructions from the repository:
   https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-vllm-plugin

2. **Required Environment Variables**

   ```bash
   # Specify the Hugging Face model to use
   export HF_MODEL='meta-llama/Llama-3.1-8B-Instruct'

   # Enable VLLM V1 API
   export VLLM_USE_V1=1

   # Set the model runner
   export MODEL_RUNNER=vllm-forge
   ```

3. **Run the Server**

### Testing VLLM Completions

Once the server is running, you can test text completion using curl. The VLLM endpoint supports streaming responses by default. Tokens will be returned as they are generated:


```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/completions' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Write a short story about a robot",
    "max_tokens": 500,
    "temperature": 0.8
  }' \
  --no-buffer
```

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable.

## Audio Preprocessing Setup and Model Terms

When setting `allow_audio_preprocessing` for the first time and testing audio models, you must:

**Accept Terms for All Required Models:**
1. Main diarization model: https://hf.co/pyannote/speaker-diarization-3.0
2. Segmentation model: https://hf.co/pyannote/segmentation-3.0

- For Company/University, enter: `Tenstorrent Inc.`
- For Website, enter: `https://tenstorrent.com`

**Hugging Face Token Setup:**
- Create a Hugging Face token on the HF website with read permission.
- Export the token as an environment variable:

```bash
export HF_TOKEN=[copied token]
```

This is required for downloading and using the models during audio preprocessing.


## Testing instructions

If server is running in development mode (ENVIRONMENT=development), OpenAPI endpoint is available on /docs URL.

# Chat completions test call

Available when running an LLM via the VLLM TT plugin (see [VLLM with TT Plugin Setup](#vllm-with-tt-plugin-setup)). OpenAI-compatible endpoint. Set `"stream": true` to receive Server-Sent Events.

```bash
curl -X POST 'http://127.0.0.1:8000/v1/chat/completions' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": false
  }'
```

# Text completions test call

OpenAI-compatible legacy text completion endpoint. Streaming is enabled with `"stream": true`.

```bash
curl -X POST 'http://127.0.0.1:8000/v1/completions' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Once upon a time",
    "max_tokens": 200,
    "temperature": 0.8,
    "stream": true
  }' \
  --no-buffer
```

# Embeddings test call

Available when `MODEL_RUNNER` is set to an embedding runner (e.g. `tt-bge-large-en-v1.5`, `tt-qwen3-embedding-8b`). OpenAI-compatible response shape.

```bash
curl -X POST 'http://127.0.0.1:8000/v1/embeddings' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BAAI/bge-large-en-v1.5",
    "input": "Hello world"
  }'
```

# Tokenize / detokenize test call

The tokenizer endpoints load the HuggingFace tokenizer for the requested `model` (must be a value present in `SupportedModels`). They do not run on the device and do not require the model to be loaded as the active runner.

```bash
# Tokenize text
curl -X POST 'http://127.0.0.1:8000/v1/tokenize' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Hello world",
    "add_special_tokens": true,
    "return_token_strs": true
  }'

# Detokenize
curl -X POST 'http://127.0.0.1:8000/v1/detokenize' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "tokens": [128000, 9906, 1917]
  }'
```

# Image API

The image router exposes one of three endpoints depending on `MODEL_RUNNER`:

| `MODEL_RUNNER`             | Active endpoint                     |
|----------------------------|-------------------------------------|
| `tt-sdxl-image-to-image`   | `POST /v1/images/image-to-image`    |
| `tt-sdxl-edit`             | `POST /v1/images/edits`             |
| any other image runner     | `POST /v1/images/generations`       |

## Image generation test call

Sample for calling the endpoint for image generation via curl:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/images/generations' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Volcano on a beach",
  "negative_prompt": "low quality",
  "num_inference_steps": 20,
  "seed": 0,
  "guidance_scale": 7.0,
  "number_of_images": 1
}'
```

The response includes the list of base64-encoded images and total `generation_time` in seconds.

## Image-to-image test call

Available when `MODEL_RUNNER=tt-sdxl-image-to-image`.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/images/image-to-image' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Make it look like a watercolor painting",
  "image": "[base64 encoded image]",
  "num_inference_steps": 20,
  "guidance_scale": 7.0,
  "strength": 0.7
}'
```

## Image edit test call

Available when `MODEL_RUNNER=tt-sdxl-edit`.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/images/edits' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Replace the masked area with a sunset",
  "image": "[base64 encoded image]",
  "mask": "[base64 encoded mask]",
  "num_inference_steps": 20,
  "guidance_scale": 7.0
}'
```

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable.

# Audio transcription and translation test call

The audio transcription and translation API supports multiple audio formats and input methods with automatic format detection and conversion.

The active endpoint depends on the `AUDIO_TASK` environment variable:

| `AUDIO_TASK`     | Active endpoint                  |
|------------------|----------------------------------|
| `transcribe` (default) | `POST /v1/audio/transcriptions` |
| `translate`            | `POST /v1/audio/translations`   |

Both endpoints accept the same payload (base64 JSON or multipart file upload) and produce the same response shape.

- Base64 JSON Request: Send a JSON POST request to `/v1/audio/transcriptions` or `/v1/audio/translations`
Sample for calling the audio transcription/translations endpoint via curl:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/audio/transcriptions' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  --data-binary @server/tests/test_data.json \
  --no-buffer
```

test_data.json file example:
```bash
{
    "stream": false,
    "file": "[base64 audio file]"
}
```

- File Upload (WAV/MP3): Send a multipart form data POST request to `/v1/audio/transcriptions` or `/v1/audio/translations`
```bash
# WAV file upload
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@/path/to/audio.wav" \
  -F "stream=true" \
  -F "is_preprocessing_enabled=true" \
  -F "perform_diarization=false" \
  -F "temperatures=0.0,0.2,0.4,0.6,0.8,1.0" \
  -F "compression_ratio_threshold=2.4" \
  -F "logprob_threshold=-1.0" \
  -F "no_speech_threshold=0.6" \
  -F "return_timestamps=true" \
  -F "prompt=test" \
  --no-buffer
```

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable.

*Please note that test_data.json is within docker container or within tests folder*


# Speaker diarization test call

Served when the model resolves to the diarization service — `MODEL=speaker-diarization-community-1`, which is all `run.py` passes. Speaker diarization answers "who spoke when": it returns speaker turns only (no transcript). The request/response schema follows the **pyannoteAI cloud diarization API** — the request matches [`DiarizeRequest`](https://docs.pyannote.ai/api-reference/diarize) and the response matches [`DiarizationJobOutput`](https://docs.pyannote.ai/api-reference/get-job) / `DiarizationSegment` (machine-readable spec: https://docs.pyannote.ai/openapi.json) — so a pyannoteAI client can switch base URL only. The `tests/test_diarization_pyannoteai_schema_conformance.py` test fetches the official OpenAPI spec live on every run and fails if these fields drift. The diarization neural nets run on the Tenstorrent device the catalog resolved (`settings.device_ids`), so no device id has to be passed. `DIARIZATION_TT_SEGMENTATION` additionally offloads the segmentation net, which is slower and mainly useful for device coverage; it is declared in the model spec's `env_vars` because the host environment does not reach the container, so exporting it in a shell has no effect. If the device cannot be opened the service fails to start rather than continuing on CPU: this model is served because it runs on the accelerator, and a silent fallback would keep answering while delivering none of that.

Weights default to the `pyannote/speaker-diarization-community-1` repo id, which is gated: export `HF_TOKEN` after accepting the terms on the model page, or point `HF_MODEL` at the ungated mirror `pyannote-community/speaker-diarization-community-1`, which carries the same checkpoints.

Diarization is a job: `POST /v1/diarize` creates one and `GET /v1/jobs/{jobId}` returns it. That is the only shape the official API has, so there is no synchronous convenience route here — one published at `/v1/audio/diarize` was a path no pyannoteAI client would ever call.

**Endpoint:** `POST /v1/diarize`
**Content-Type:** `application/json`

The body is the pyannoteAI `DiarizeRequest`. Audio is referenced by `url`: a public `http(s)://` URL, or a `media://<object-key>` staged via the media API (see below). There is no multipart file field, matching pyannoteAI. Whichever form is used, the payload is capped at `media_url_max_bytes` (64 MiB for this model, about 35 minutes of 16 kHz mono audio); over that the server answers 413.

> **Non-standard extension:** this server *also* accepts the audio itself as inline base64 in the same `url` field. **The pyannoteAI cloud API does not** — it documents `url` as "URL of the audio file to be processed" and takes a fetchable location only, so a client written against this extension will fail the moment it is pointed at the official service. It exists because the official request carries audio in exactly one field and that field is a location, which leaves a deployment with no object storage the server can reach no way to send audio at all. It is expressible without emitting a request the spec would reject only because the official schema types `url` as a bare string with no pattern, and it follows the same "URL or base64 in one field" shape the video endpoint uses for images.

### Choosing between inline base64 and `media://`

**Both forms are capped at 1 GiB**, matching [what the pyannoteAI cloud API accepts](https://docs.pyannote.ai/support/faqs) for a diarization job, so a client that works against the official service is not refused here for a reason the official service would not have refused it. `MEDIA_URL_MAX_BYTES` bounds fetched audio (`media://`, `http(s)://`); `MEDIA_INLINE_MAX_BYTES` bounds inline base64.

An oversized request is refused on its declared `Content-Length` **before the body is received** (`BodySizeLimitMiddleware`). That matters more than it sounds: an endpoint cannot defend itself against a large body, because the ASGI layer has already buffered the whole request by the time a handler runs. Measured on a p150 — a 1 GiB inline body that the *endpoint* rejected still cost **+1289 MiB** RSS, and a 900 MiB one OOM-killed a 6 GiB container; with the middleware, a 900 MiB announced body against a 64 MiB limit costs **+1 MiB** instead of +894 MiB. Verified on the running server: a 2 GiB declared body is answered 413 with **no measurable memory increase**.

A fetched object is additionally refused on the declared `Content-Length` when the store sends one, and otherwise mid-stream once the bytes read pass the cap. Neither limit can be dodged by switching form.

*One caveat on inline.* An **accepted** body still has to be held: 1 GiB of audio is ~1.33 GiB of base64 resident while it decodes, and it is the one input that cannot be streamed. Measured: a 200 MiB inline body (109 minutes of PCM16) submitted in 5 s at **+634 MiB** RSS and diarized in 739 s. On a tight memory allotment, lower `MEDIA_INLINE_MAX_BYTES` — it is there to be lowered. Prefer `media://` regardless: it is the only form the official API takes, it does not inflate the payload by a third, and the upload does not sit inside the inference request.

### Where the numbers come from

Nothing below the application enforces a body size: `run_uvicorn.sh` sets no `--limit-*` flags and there is no proxy in the container, so these settings are the only thing standing between a client and an arbitrarily large upload. Removing them is not an option — with the caps raised to 1 TiB inside an 8 GiB memory limit, a 256 MiB inline body was accepted, 1 GiB reached 5.9 GiB RSS, and 3 GiB killed the server outright (`OOMKilled=true`, exit 137, no recovery). An unbounded request body is a way for one client to take the service down, so the question is only where the limit sits.

That 8 GiB is not a hypothetical: `run.py` sets no memory limit, but the Helm chart does, and the nearest precedent — `whisper-large-v3`, the other audio model — is allotted **6 GiB** (`charts/tt-inference-server/values.yaml`; media models cluster at 6–32 GiB, with the large LLMs at 175–300 GiB). Idle diarization measures 1.6 GiB resident, leaving ~4.4 GiB of headroom in a whisper-sized allotment. That is the figure to size an inline body against, since an accepted one is resident: 200 MiB inline measured +634 MiB, so the 1 GiB default assumes a roomier allotment than whisper's and should be lowered where that is not available. A fetched object is streamed and stays near the audio size regardless. Under `docker run` with no limit the ceiling is the host instead, which is why the caps do the work rather than the container. Diarization has no chart entry yet; when one is added, 6 GiB is the figure to start from, on the evidence above.

The defaults come from the official API rather than from a guess about what a recording ought to be, and were then checked against what they cost here:

| | Setting | Default | Chosen because |
|---|---|---|---|
| `media://`, `http(s)://` | `MEDIA_URL_MAX_BYTES` | 1 GiB | The official pyannoteAI limit for a diarization job. Verified on a p150: a 300 MiB file (163 min of PCM16) diarized successfully in 802 s, 12.3× realtime. |
| inline base64 | `MEDIA_INLINE_MAX_BYTES` | 1 GiB | The same official limit. Safe to match only because the declared `Content-Length` is refused before the body arrives; an accepted body is still ~1.33× resident, so lower it on a tight allotment. Measured: 200 MiB inline, +634 MiB RSS, diarized in 739 s. |
| multipart | — | — | No such route: pyannoteAI has no file field, so one was never added. |

The byte cap is deliberately *not* the binding constraint at that size — `request_processing_timeout_seconds` (1000 s) is. The official API also allows 24 hours of audio and this server cannot: at ~12–18× realtime a p150 needs over an hour for 24 hours of audio. Compressed input makes the gap wider, since 1 GiB of ~100 kbps mp3 *is* the full 24 hours while 1 GiB of PCM16 is 9.3 hours. A long file inside the cap can therefore still exceed the request deadline, and that is the timeout's job to report: refusing on a byte count a file the official API accepts would be the wrong error for the wrong reason. Requests are also serialised (one pipeline, `max_batch_size: 1`), so a long recording blocks the queue for its whole run — the 300 MiB run above held it for 13 minutes. And the inline path must stay well inside the container's memory, since the whole body is resident before decoding — the measurements above put the practical inline limit at a fraction of available RAM, not at whatever number looks generous. Sizes are for uncompressed PCM16; a compressed input of the same byte count carries far more audio and takes proportionally longer.

*These are the only size limits on this path.* `max_audio_size_bytes` (50 MiB) is **not** a second, universal check underneath it: it is enforced by `AudioManager`, which only the transcription service goes through (`audio_service.py` → `to_audio_array` → `_validate_file_size`). `DiarizationService.pre_process` decodes with `decode_to_wav` directly and never enters `AudioManager`. Measured on a p150: a 300 MiB file — far above `max_audio_size_bytes`, below `media_url_max_bytes` — diarizes successfully via `media://`. So to bound diarization audio, change `MEDIA_URL_MAX_BYTES` and `MEDIA_INLINE_MAX_BYTES`; changing `max_audio_size_bytes` affects transcription only.

Verified on a p150: with `MEDIA_INLINE_MAX_BYTES=1 MiB` a 3 MiB clip is refused as base64 and still accepted via `media://`, and with `MEDIA_URL_MAX_BYTES=1 MiB` the same clip is refused on both.

What actually differs:

| | Inline base64 | `media://` (or `http(s)://`) |
|---|---|---|
| Portable to pyannoteAI | **No** — non-standard extension | Yes |
| Needs object storage | No | Yes (`media://`); or any reachable URL |
| Bytes on the wire | **1.333×** the audio | 1.0× |
| Upload path | Through this server, inside the request | Straight to the store, separate from the request |
| Re-diarizing the same audio | Re-uploads every time | Stage once, reference by key |

Prefer `media://` (or a plain `http(s)://` URL) — it is what the official API takes, it does not inflate the payload, and the upload does not sit inside the request to the inference server. Reach for base64 when there is no object store the server can reach, or for a one-off small clip where standing up storage is not worth it.

## Request parameters

| Parameter      | Required | Description |
|----------------|----------|-------------|
| `url`          | Yes      | Audio location: `http(s)://…` or `media://<object-key>` (capped at 64 MiB). Inline base64 audio is also accepted as a non-standard extension (not supported by pyannoteAI), capped at 16 MiB. |
| `numSpeakers`  | No       | Exact number of speakers, if known (pyannoteAI `numSpeakers`). |
| `minSpeakers`  | No       | Lower bound on the number of speakers (pyannoteAI `minSpeakers`). |
| `maxSpeakers`  | No       | Upper bound on the number of speakers (pyannoteAI `maxSpeakers`). |
| `exclusive`    | No       | When `true` (default), also return non-overlapping turns as `exclusiveDiarization` (pyannoteAI `exclusive`). |

```bash
JOB=$(curl -s -X POST 'http://127.0.0.1:8000/v1/diarize' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com/audio.wav", "exclusive": true}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["jobId"])')

curl -s "http://127.0.0.1:8000/v1/jobs/$JOB" -H 'Authorization: Bearer your-secret-key'
```

The job's `output` is the pyannoteAI `DiarizationJobOutput` (`exclusiveDiarization` present when `exclusive=true`):

```json
{
  "diarization": [
    {"speaker": "SPEAKER_00", "start": 0.5, "end": 4.2},
    {"speaker": "SPEAKER_01", "start": 4.2, "end": 7.8}
  ],
  "exclusiveDiarization": [
    {"speaker": "SPEAKER_00", "start": 0.5, "end": 4.2},
    {"speaker": "SPEAKER_01", "start": 4.2, "end": 7.8}
  ]
}
```

## Model and unsupported options

`model` follows the pyannoteAI `DiarizeRequest.model` enum. This server serves **`community-1`** only; `model=precision-2` (the paid cloud model) or any unknown value is rejected with HTTP 400.

The following pyannoteAI options are **precision-2-only** and cannot be produced by community-1, so requesting them returns HTTP 400 (they are never silently ignored): `confidence`, `turnLevelConfidence`, `transcription`, `transcriptionConfig`. Correspondingly, the precision-2-only response fields (`confidence`, `wordLevelTranscription`, `turnLevelTranscription`) are never emitted. See https://docs.pyannote.ai/openapi.json.

## Job API (pyannoteAI-native)

- `POST /v1/diarize` — create a job. Body is the pyannoteAI `DiarizeRequest` (`url` plus optional `numSpeakers`/`minSpeakers`/`maxSpeakers`/`exclusive`/`model` and `webhook`/`webhookStatusOnly`). Returns `201` with `JobCreated` (`{jobId, status}`).
- `GET /v1/jobs/{jobId}` — returns the `DiarizationJob` (`{jobId, status, createdAt, updatedAt, output}`); `output` is the `DiarizationJobOutput` once `status` is `succeeded`. Status values are the pyannoteAI enum `created|running|succeeded|failed|canceled`.
- `webhook` / `webhookStatusOnly` — when `webhook` is set, the job payload is POSTed to that URL on completion (`webhookStatusOnly=true` sends only `{jobId, status}`).

```bash
# 1. stage a private file (optional; or pass a public https url, or inline
#    base64 via this server's non-standard extension)
UP=$(curl -s -X POST http://127.0.0.1:8000/v1/media/input \
  -H 'Authorization: Bearer your-secret-key' -H 'Content-Type: application/json' \
  -d '{"url":"media://sess/audio.wav"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["url"])')
# the signature in $UP is the only credential this needs, and it goes to the
# object store rather than back through the inference server
curl -s -X PUT "$UP" --upload-file /path/to/audio.wav

# 2. create the job
JOB=$(curl -s -X POST http://127.0.0.1:8000/v1/diarize \
  -H 'Authorization: Bearer your-secret-key' -H 'Content-Type: application/json' \
  -d '{"url":"media://sess/audio.wav","exclusive":true}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["jobId"])')

# 3. poll for the result
curl -s http://127.0.0.1:8000/v1/jobs/$JOB -H 'Authorization: Bearer your-secret-key'
```

## Media input (staging private files)

`POST /v1/media/input` with `{"url":"media://<object-key>"}` returns a **pre-signed `PUT` url on an S3-compatible object store**; `PUT` the file bytes straight there, then pass `media://<object-key>` as the diarize `url`. This is the pyannoteAI temporary media storage flow (https://docs.pyannote.ai/api-reference/upload-media), including the part that matters operationally: the upload goes to storage, not through this server, so a long recording never occupies the process that is scheduling device work. The signature is SigV4, so plain `curl -T` is enough — no SDK, no credentials on the client.

This is optional. With no store configured the endpoint answers `501` and names the two inputs that need none: an `http(s)://` url, or the audio inline as base64 (the non-standard extension above). Configure it with:

| Variable | Description |
|---|---|
| `MEDIA_STORAGE_ENDPOINT` | Base url of the S3-compatible service, e.g. `http://rustfs:9000`. Empty disables `media://`. |
| `MEDIA_STORAGE_BUCKET` | Bucket the objects are staged in. |
| `MEDIA_STORAGE_ACCESS_KEY` / `MEDIA_STORAGE_SECRET_KEY` | Credentials used to sign. Never sent to the client — only the signature is. |
| `MEDIA_STORAGE_REGION` | Region string SigV4 signs over; self-hosted services ignore the value but it has to match on both sides. Default `us-east-1`. |
| `MEDIA_STORAGE_PRESIGN_EXPIRY_SECONDS` | Lifetime of a signed url. Default `3600`. |

The store's hostname is downloadable without appearing in `MEDIA_URL_ALLOWED_DOMAINS`: a `media://` key resolves to a url this server signed against its own endpoint, so it is not a client-chosen destination. Client-supplied `http(s)` urls still need that allowlist.

Retention is a **bucket lifecycle policy on the store**, not a sweeper in this server — that is where the official "at least 24 hours" guarantee belongs, and it keeps working while this server is down.

Any S3-compatible service works. For a self-hosted one, [RustFS](https://github.com/rustfs/rustfs) is the straightforward pick: Apache-2.0 (MinIO's community edition is AGPL and archived, Garage is AGPL), a single container configured by two variables, and S3-native lifecycle. Staged media is temporary, so backing it with `tmpfs` keeps it off disk and clears it with the container:

The image runs as uid/gid 10001, and a `tmpfs` mount defaults to root-owned `0755`, so the ownership has to be given on the mount or the server exits with `Permission denied (os error 13)`:

```bash
docker run -d --name rustfs \
  --tmpfs /data:rw,size=8g,uid=10001,gid=10001 \
  --tmpfs /logs:rw,size=64m,uid=10001,gid=10001 \
  -e RUSTFS_ACCESS_KEY=media -e RUSTFS_SECRET_KEY=media-secret \
  -p 9000:9000 rustfs/rustfs:latest

# create the bucket and expire its objects after a day
AWS_ACCESS_KEY_ID=media AWS_SECRET_ACCESS_KEY=media-secret AWS_DEFAULT_REGION=us-east-1 \
  aws --endpoint-url http://127.0.0.1:9000 s3 mb s3://media
AWS_ACCESS_KEY_ID=media AWS_SECRET_ACCESS_KEY=media-secret AWS_DEFAULT_REGION=us-east-1 \
  aws --endpoint-url http://127.0.0.1:9000 s3api put-bucket-lifecycle-configuration \
  --bucket media --lifecycle-configuration \
  '{"Rules":[{"ID":"expire-staged-media","Status":"Enabled","Filter":{"Prefix":""},"Expiration":{"Days":1}}]}'
```

Then point the server at it. `run.py` passes the repository-root `.env` to the container with `--env-file`, so that is where these belong:

```bash
cat >> .env <<'VARS'
MEDIA_STORAGE_ENDPOINT=http://10.0.0.5:9000
MEDIA_STORAGE_BUCKET=media
MEDIA_STORAGE_ACCESS_KEY=media
MEDIA_STORAGE_SECRET_KEY=media-secret
VARS
```

The endpoint has to be reachable **from inside the container**, which `run.py` starts on the default bridge network with no `--link` and no shared user-defined network. A container name will not resolve; use an address the container can route to, such as the host address the store's port is published on. Both the client that `PUT`s and this server resolve the same endpoint string, so it has to be valid for both.

## Benchmarking and accuracy

`benchmarking/run_benchmarks.py` covers the served endpoint: `ModelType.DIARIZATION` selects `BenchmarkTaskDiarization`, so the sweep is a single diarization-typed run rather than the LLM prompt-length sweep.

`evals/run_evals.py` covers it too. Media model types are dispatched to `run_media_evals`, which calls the client's `run_eval` directly rather than going through `lm-evaluation-harness` — so the absence of an lm-eval diarization task is no obstacle, the same way the TTS model scores itself.

The metric is the diarization error rate, the standard for this task: the fraction of speaking time attributed to the wrong speaker, plus missed speech and false alarm. The reference is the hand annotation (`sample.rttm`) that ships beside pyannote's sample recording, so the eval scores the served pipeline against ground truth rather than against another run of itself. The speaker count is reported and checked alongside the DER, because a pipeline that splits or merges speakers can still post an acceptable DER.

Measured on a p150: `DER = 0.052` with the two annotated speakers. That is not a porting error — the all-CPU pipeline scores **the same 0.052** against the same annotation, and the device run reproduces the CPU run exactly (`DER = 0.000` between them). It is the ordinary gap between any diarizer and a human transcript at turn boundaries. Nor does it contradict community-1's published 0.17: that figure is measured on AMI / DIHARD / VoxConverse, full corpora of hard meeting and broadcast audio, whereas this is a clean 30 s two-speaker clip.

The scoring is shared with tt-metal rather than reimplemented: both this eval and `models/demos/audio/pyannote_diarization/tests/test_diarization_e2e_ondevice.py` call `models.demos.audio.pyannote_diarization.accuracy`, so the served model and the on-device tests report the same figure against the same thresholds. That test asserts both directions — `DER < 0.05` for the device run against the host run (fidelity: the port has not drifted from the reference implementation) and `DER < 0.15` against the human annotation (accuracy: the pipeline is right in absolute terms, which fidelity alone cannot show, since host and device could be wrong together).



# Text-to-Speech (TTS) test call

The Text-to-Speech API converts text to speech audio using the SpeechT5 model. The response is binary audio (WAV, MP3, OGG) or JSON with base64 audio and metadata.

**Endpoint:** `POST /v1/audio/speech`
**Content-Type:** `application/json`

## Request parameters

| Parameter           | Required | Description |
|--------------------|----------|-------------|
| `text`             | Yes      | Input text to convert to speech. |
| `response_format`  | No       | Output format: `wav` (default), `mp3`, `ogg`, `json`, or `verbose_json`. |

## Response formats

- **`wav`** (default) – Binary WAV (`Content-Type: audio/wav`). No ffmpeg required.
- **`mp3`** – Binary MP3 (`Content-Type: audio/mpeg`). Requires ffmpeg on the server.
- **`ogg`** – Binary OGG (`Content-Type: audio/ogg`). Requires ffmpeg on the server.
- **`json`** / **`verbose_json`** – JSON body with base64-encoded audio (`audio`), `duration`, `sample_rate`, `format`. No ffmpeg required.

If `response_format` is `mp3` or `ogg` but ffmpeg is not in PATH (or encoding fails), the server logs a warning and **falls back to WAV** (HTTP 200, `Content-Type: audio/wav`).

**Prerequisite for MP3/OGG:** Install ffmpeg so the server can encode to MP3/OGG. From tt-media-server: `sudo apt update && sudo apt install -y ffmpeg`. Same as in [For development running](#for-development-running) step 4.

## Content-Disposition and curl -J -O

The server sends `Content-Disposition: attachment; filename=speech.<format>` (e.g. `speech.wav`, `speech.mp3`, `speech.ogg`) so the suggested filename matches the actual format. Use **`curl -J -O`** to save with that filename and avoid extension mismatch (e.g. requesting ogg but saving as `output.mp3`).

## Examples

**Default (WAV):**

```bash
curl -X POST 'http://127.0.0.1:8000/v1/audio/speech' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, this is a test of the text to speech system."}' \
  --output output.wav \
  --silent \
  --show-error
```

**MP3:**

```bash
curl -X POST 'http://127.0.0.1:8000/v1/audio/speech' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, this is a test of the text to speech system.", "response_format": "mp3"}' \
  --output output.mp3 \
  --silent \
  --show-error
```

**OGG (or use -J -O to save as speech.ogg from Content-Disposition):**

```bash
curl -X POST 'http://127.0.0.1:8000/v1/audio/speech' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, this is a test of the text to speech system.", "response_format": "ogg"}' \
  -J -O
```

**JSON response (base64 audio + metadata):**

```bash
curl -X POST 'http://127.0.0.1:8000/v1/audio/speech' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, this is a test of the text to speech system.", "response_format": "verbose_json"}' \
  --silent
```

**Request body examples (Swagger/OpenAPI):**

```json
{"text": "Hello, this is a test of the text to speech system."}
```

```json
{"text": "Hello world", "response_format": "wav"}
```

```json
{"text": "Hello world", "response_format": "mp3"}
```

```json
{"text": "Hello world", "response_format": "ogg"}
```

```json
{"text": "Hello world", "response_format": "json"}
```

```json
{"text": "Hello world", "response_format": "verbose_json"}
```

# Image search test call

The image search API uses a CNN model to search for similar images. It supports multiple input methods.

- Base64 JSON Request: Send a JSON POST request to `/search-image`
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/search-image' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "[base64 encoded image]",
  "response_format": "json",
  "top_k": 3,
  "min_confidence": 70.0
}'
```

- File Upload: Send a multipart form data POST request to `/v1/cnn/search-image`
```bash
curl -X POST "http://localhost:8000/v1/cnn/search-image" \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@/path/to/image.jpg" \
  -F "response_format=json" \
  -F "top_k=5" \
  -F "min_confidence=80.0"
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` / `file` | string / file | required | Base64-encoded image (JSON) or image file (multipart) |
| `response_format` | string | `"json"` | Response format for results |
| `top_k` | integer | `3` | Number of top results to return |
| `min_confidence` | float | `70.0` | Minimum confidence threshold (0-100) |

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable.

# Video generation API

The video API supports both **async** (job-based) and **sync** modes, controlled via the `USE_ASYNC_VIDEO` environment variable:

- `USE_ASYNC_VIDEO=True` (default): the server creates a job and returns metadata with a `job_id`. Use the `GET /v1/videos/generations/{job_id}` and `/download` endpoints to track progress and retrieve the video.
- `USE_ASYNC_VIDEO=False`: the server processes synchronously and streams the MP4 back directly in the same request.

## Submit text-to-video generation job

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/videos/generations' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Volcano on a beach",
  "negative_prompt": "low quality",
  "num_inference_steps": 20
}'
```

**Response example (async mode, HTTP 202):**
```json
{
  "id": "video_id_1",
  "object": "video",
  "status": "queued",
  "created_at": 1702860000,
  "model": "Wan2.2-T2V-A14B-Diffusers"
}
```

Save the `id` field from the response (e.g., `video_id_1`) to use as `{video_id}` in subsequent requests.

In sync mode (`USE_ASYNC_VIDEO=False`) the response is the raw MP4 (`Content-Type: video/mp4`) with an `X-Generation-Time` header.

## Submit image-to-video generation job

Available with the Wan2.2 I2V runner. Requires at least one entry in `image_prompts`.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/videos/generations/i2v' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Camera slowly zooms in",
  "negative_prompt": "low quality",
  "num_inference_steps": 20,
  "image_prompts": ["[base64 encoded image]"]
}'
```

## Get video job metadata

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/v1/videos/generations/{video_id}' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key'
```

## List all video jobs

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/v1/videos/jobs' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key'
```

## Download generated video

The `/v1/videos/generations/{video_id}/download` endpoint for downloading a video file

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/v1/videos/generations/{video_id}/download' \
  -H 'Authorization: Bearer your-secret-key' \
  -o output.mp4
```

## Cancel video job and assets

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/videos/generations/{video_id}/cancel' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key'
```

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable.

# Fine-tuning API

All fine-tuning endpoints require the API key **and** an organization header (default name `X-TT-Organization`, configurable via `ORG_ID_HEADER`). The org id is used to scope jobs to a tenant.

## Get fine-tuning catalog

Lists available models, datasets, trainers, optimizers, and clusters that this server instance can fine-tune.

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/v1/fine_tuning/catalog' \
  -H 'Authorization: Bearer your-secret-key'
```

## Create fine-tuning job

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/fine_tuning/jobs' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'X-TT-Organization: my-org' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "training_file": "file-abc123",
  "hyperparameters": {
    "n_epochs": 3,
    "batch_size": 4,
    "learning_rate_multiplier": 1.0
  }
}'
```

**Response example:**
```json
{
  "id": "ftjob-abc123",
  "object": "training",
  "status": "queued",
  "created_at": 1702860000,
  "model": "meta-llama/Llama-3.1-8B-Instruct"
}
```

Save the `id` field from the response (e.g., `ftjob-abc123`) to use as `{job_id}` in subsequent requests.

## List fine-tuning jobs

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/v1/fine_tuning/jobs' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'X-TT-Organization: my-org'
```

## Get fine-tuning job details

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/v1/fine_tuning/jobs/{job_id}' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'X-TT-Organization: my-org'
```

## Get training metrics

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/v1/fine_tuning/jobs/{job_id}/metrics' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'X-TT-Organization: my-org'
```

## Get fine-tuning job logs

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/v1/fine_tuning/jobs/{job_id}/logs' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'X-TT-Organization: my-org'
```

## Cancel fine-tuning job

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/fine_tuning/jobs/{job_id}/cancel' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'X-TT-Organization: my-org'
```

## List fine-tuning job checkpoints

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/v1/fine_tuning/jobs/{job_id}/checkpoints' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'X-TT-Organization: my-org'
```

## Download checkpoint adapter weights

Returns a zip archive of the adapter weights for the requested checkpoint.

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/v1/fine_tuning/jobs/{job_id}/checkpoints/{checkpoint_id}' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'X-TT-Organization: my-org' \
  -o adapter_{checkpoint_id}.zip
```

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable, and `my-org` with the value clients should send for `X-TT-Organization` (or the header name configured via `ORG_ID_HEADER`).

## Unit Testing Setup in VS Code

To set up and run unit tests in VS Code with pytest support, follow these steps:

### 1. Install Required Extension

Install the **Python Extension Pack** from VS Code extensions marketplace. This provides complete Python development support including testing capabilities.

### 2. Create VS Code Settings File

Create a `.vscode/settings.json` file in your workspace root with the following configuration:

```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "--rootdir=.",
        "resolver/",
        "tests/",
        "."
    ],
    "python.testing.cwd": "${workspaceFolder}",
    "python.defaultInterpreterPath": "/opt/venv/bin/python",
    "python.testing.autoTestDiscoverOnSaveEnabled": true,
    "python.languageServer": "Pylance",
    "python-envs.pythonProjects": [],
    "python.envFile": "${workspaceFolder}/.env.test"
}
```

**Note:** Update `python.defaultInterpreterPath` to match your tt-metal Python environment location.

### 3. Create Test Environment File

Create a `.env.test` file in the project root with the following configuration:

```bash
PYTHONPATH=[path to tt-metal]:[path to tt-media-server]
TT_METAL_PATH=[path to tt-metal]
```

**Note:** Update the paths to match your local environment setup.

### 4. Configure Python Interpreter

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Search for "Python: Select Interpreter"
3. Choose the Python interpreter from your tt-metal environment

### 5. Running and Debugging Tests

Once configured, you should be able to run and debug (all or some specific) tests directly from VS Code. In order to do that you can open the Testing sidebar or open a test file in the editor.

# Configuration

The TT Inference Server can be configured using environment variables or by modifying the settings file. All parameter names should be **UPPERCASED** when used as environment variables.

## General Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `LOG_LEVEL` | `"INFO"` | Sets the logging level for the application. Valid values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `ENVIRONMENT` | `"development"` | Specifies the runtime environment. Used for environment-specific configurations |
| `LOG_FILE` | `None` | Optional path to log file. If not set, logs are output to console only |

## Device Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `DEVICE_IDS` | `"(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15),(16),(17),(18),(19),(20),(21),(22),(23),(24),(25),(26),(27),(28),(29),(30),(31)"` | Comma-separated list of device IDs available for inference. Defines which TT devices can be used |
| `IS_GALAXY` | `True` | Boolean flag indicating if running on Galaxy hardware. Used for graph device split and class initialization |
| `DEVICE_MESH_SHAPE` | `(1, 1)` | Tuple defining the device mesh topology. Format: `(rows, columns)` for multi-device setups |
| `RESET_DEVICE_COMMAND` | `"tt-smi -r"` | Command used to reset TT devices when needed |
| `RESET_DEVICE_SLEEP_TIME` | `5.0` | Time in seconds to wait after device reset before attempting reconnection |
| `ALLOW_DEEP_RESET` | `False` | Boolean flag that gates the **internal worker-health auto-recovery** path. When `True`, the scheduler may trigger `deep_restart_workers()` after a worker has exceeded `MAX_WORKER_RESTART_COUNT`. Note: this flag does **not** gate the externally-triggered `/tt-deep-reset` endpoint, which is always available |
| `USE_GREEDY_BASED_ALLOCATION` | `True` | Boolean flag to enable greedy-based device allocation strategy. When enabled with single device mesh shape (1,1), automatically allocates all available devices from the system |

## Model Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `MODEL_RUNNER` | [`ModelRunners.TT_SDXL_TRACE.value`](config/constants.py ) | Specifies which model runner implementation to use for inference |
| `MODEL_SERVICE` | `None` | Specifies which model service implementation to use for inference. If not set, the default service for the selected model runner will be used |
| `MODEL_WEIGHTS_PATH` | `""` | Path to the main model weights. Used if `HF_HOME` is not set. |
| `PREPROCESSING_MODEL_WEIGHTS_PATH` | `""` | Path to preprocessing model weights (e.g., for audio preprocessing). Used if `HF_HOME` is not set. |
| `TRAINING_MODEL` | `None` | HuggingFace model ID used by the fine-tuning catalog when `MODEL_SERVICE=training` |
| `CHAT_TEMPLATE_KWARGS` | `{}` | Extra kwargs passed to `tokenizer.apply_chat_template` for chat completions (e.g. Qwen3 thinking mode flags) |
| `SDXL_IMAGE_RESOLUTION` | `(1024, 1024)` | Output resolution for SDXL text-to-image. Must be one of the values in `SDXL_VALID_IMAGE_RESOLUTIONS` |
| `TRACE_REGION_SIZE` | `34541598` | Memory size allocated for model tracing operations (in bytes) |
| `DOWNLOAD_WEIGHTS_FROM_SERVICE` | `True` | Boolean flag to enable downloading weights when initializing service. When enabled, ensures that weights are downloaded once per instance of the server |


## Queue and Batch Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `MAX_QUEUE_SIZE` | `5000` | Maximum number of requests that can be queued for processing |
| `MAX_BATCH_SIZE` | `1` | Maximum batch size for inference requests. Currently limited to 1 for stability |
| `MAX_BATCH_DELAY_TIME_MS` | `None` | Maximum wait time in milliseconds after the first request before a batch is executed, allowing more requests to accumulate without adding significant latency |
| `USE_DYNAMIC_BATCHER` | `False` | Boolean flag to enable dynamic batching for improved throughput. When enabled, the server attempts to batch multiple requests together for more efficient processing |
| `USE_QUEUE_PER_WORKER` | `False` | Boolean flag to enable per-worker result queues. When enabled, each worker has its own dedicated result queue instead of a shared queue, which can improve performance in high-concurrency scenarios by reducing queue contention |
| `QUEUE_FOR_MULTIPROCESSING` | `TTQueue` | Selects the queue implementation for inter-process communication. Options: `TTQueue` (default, Python's multiprocessing.Queue), `FasterFifo` (high-performance, uses faster-fifo library). |

### Dynamic Batching

The `USE_DYNAMIC_BATCHER` setting controls whether the server uses dynamic batching to improve throughput:

- **When `False` (default)**: While one request is in process, new requests are not added
- **When `True`**: The server attempts to add multiple requests during the inference

**Usage:**
```bash
# Enable dynamic batching for higher throughput scenarios
export USE_DYNAMIC_BATCHER=true
export MAX_BATCH_SIZE=4
export MAX_BATCH_DELAY_TIME_MS=50
```

**Note:** Dynamic batching is currently experimental and may not be supported by all model runners. Check your specific model runner documentation for batching support.

## Worker Management

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `NEW_DEVICE_DELAY_SECONDS` | `0` | Delay in seconds before initializing a new device worker, 0 by default |
| `NEW_RUNNER_DELAY_SECONDS` | `2` | Delay in seconds before initializing a new CPU worker |
| `MOCK_DEVICES_COUNT` | `5` | Number of mock devices to create when running in mock/test mode |
| `MAX_WORKER_RESTART_COUNT` | `5` | Maximum number of times a worker can be restarted before being marked as failed |
| `WORKER_CHECK_SLEEP_TIMEOUT` | `30.0` | Time in seconds between worker health checks |
| `DEFAULT_THROTTLE_LEVEL` | `"5"` | Controls the maximum number of concurrent tasks or requests a worker can handle before throttling is applied |

## Timeout Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `REQUEST_PROCESSING_TIMEOUT_SECONDS` | `1000` | Default timeout for processing requests in seconds |
| `WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS` | `1200` | Maximum time in seconds to wait for weights to be distributed to all workers before failing startup |
| `VIDEO_REQUEST_TIMEOUT_SECONDS` | `300.0` | SHM response deadline used by the video sub-process runner (`SPRunner` proxy to `video_runner`). Tune for long video generations |

## Job Management Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `MAX_JOBS` | `10000` | Maximum number of jobs allowed in the job manager. |
| `JOB_CLEANUP_INTERVAL_SECONDS` | `300` | Interval in seconds between automatic job cleanup checks. The background cleanup task runs at this frequency to remove old jobs and cancel stuck jobs |
| `JOB_RETENTION_SECONDS` | `86400` | Duration in seconds to keep completed or failed jobs before automatic removal. Jobs older than this threshold are cleaned up to free memory. Default is 1 day |
| `JOB_MAX_STUCK_TIME_SECONDS` | `10800` | Maximum time in seconds a job can remain in "in_progress" status before being automatically cancelled as stuck. Helps prevent zombie jobs from consuming resources. Default is 3 hours |
| `ENABLE_JOB_PERSISTENCE` | `False` | Boolean flag to enable persistent job storage to database. When enabled, jobs are saved to disk and can survive server restarts |
| `JOB_DATABASE_PATH` | `./jobs.db` | The file system path where the job database is stored. This setting is only applicable when job persistence is enabled |

## VLLM Settings

These settings configure VLLM-based model runners and are grouped under `settings.vllm` in the configuration.

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `VLLM__MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | Hugging Face model identifier for VLLM inference. |
| `VLLM__MIN_CONTEXT_LENGTH` | `32` | Sets the minimum number of tokens that can be processed per sequence. Must be a power of two. Must be less than max_model_length. Min value is 32. |
| `VLLM__MAX_MODEL_LENGTH` | `2048` | Sets the maximum number of tokens that can be processed per sequence, including both input and output tokens. Determines the model's context window size. |
| `VLLM__MAX_NUM_BATCHED_TOKENS` | `max_model_length * max_num_seqs` | Sets the maximum total number of tokens processed in a single iteration across all active sequences. Higher values improve throughput but increase memory usage and latency. |
| `VLLM__MAX_NUM_SEQS` | `1` | Defines the maximum number of sequences that can be batched and processed simultaneously in one iteration. Note: tt-xla currently only supports max_num_seqs=1. |
| `VLLM__GPU_MEMORY_UTILIZATION` | `0.1` | Fraction of GPU memory to use for model weights and KV cache. |
| `MAX_MODEL_LENGTH` | `4096` | Top-level alias used when constructing `VLLMSettings`; if set, takes effect even before nested `VLLM__*` parsing (used as the default for `vllm.max_model_length`) |
| `MAX_NUM_SEQS` | `1` | Top-level alias used when constructing `VLLMSettings`; if set, takes effect even before nested `VLLM__*` parsing (used as the default for `vllm.max_num_seqs`) |
| `GPU_MEMORY_UTILIZATION` | `0.1` | Top-level alias used when constructing `VLLMSettings`; if set, takes effect even before nested `VLLM__*` parsing (used as the default for `vllm.gpu_memory_utilization`) |

## Audio Processing Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `ALLOW_AUDIO_PREPROCESSING` | `True` | Boolean flag to allow audio preprocessing capabilities |
| `AUDIO_CHUNK_DURATION_SECONDS` | Auto-calculated | Duration in seconds for audio chunks during processing. If not set, automatically calculated based on worker count: 3s for 8+ workers, 15s for 4-7 workers, 30s for 1-3 workers. Can be overridden by setting this environment variable |
| `MAX_AUDIO_DURATION_SECONDS` | `60.0` | Maximum allowed audio duration (in seconds) |
| `MAX_AUDIO_DURATION_WITH_PREPROCESSING_SECONDS` | `300.0` | Maximum allowed audio duration (in seconds) when audio preprocessing (e.g., speaker diarization) is enabled |
| `MAX_AUDIO_SIZE_BYTES` | `52428800` | Maximum allowed audio file size (50 MB in bytes) |
| `DEFAULT_SAMPLE_RATE` | `16000` | Default audio sample rate for processing (16 kHz) |
| `AUDIO_TASK` | `"transcribe"` | Specifies the audio processing task: transcription (speech-to-text in original language) or translation (speech-to-English or other supported language) |
| `AUDIO_LANGUAGE` | `"English"` | Specifies the language for audio processing (transcription or translation). Supported languages depend on the selected Whisper model. |

## Video Generation Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `USE_ASYNC_VIDEO` | `True` | When `True`, video generation creates a job and returns metadata; when `False`, the request blocks and the MP4 is streamed back directly |
| `TT_VIDEO_SHM_INPUT` | `"tt_video_in"` | Name of the shared-memory segment used to send requests to the video runner sub-process |
| `TT_VIDEO_SHM_OUTPUT` | `"tt_video_out"` | Name of the shared-memory segment used to receive results from the video runner sub-process |
| `TT_VIDEO_FILE_DIR` | `"/dev/shm"` | Directory used by the video pipeline to write intermediate / output video files |
| `TT_VIDEO_EXPORT_CRF` | `"23"` | x264 CRF used when exporting MP4 (lower = higher quality) |
| `TT_VIDEO_EXPORT_PRESET` | `"ultrafast"` | x264 preset used when exporting MP4 (e.g. `ultrafast`, `fast`, `medium`) |

## Operational TT Settings

These environment variables are typically set automatically by the worker bootstrap (`utils/runner_utils.py`) but can be overridden when needed:

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `TT_METAL_HOME` | _required_ | Path to the tt-metal install. Used by `run_uvicorn.sh` to activate the Python env and by workers to locate cache and mesh descriptors |
| `SERVICE_PORT` | `8000` | Port used by `run_uvicorn.sh` when launching uvicorn |
| `TT_DIT_CACHE_DIR` | `/tmp/TT_DIT_CACHE` | Cache directory used by DiT model runners (set automatically by the runner) |
| `TT_SMI_TIMEOUT` | `30` | Timeout in seconds for `tt-smi` calls performed by `DeviceManager` |
| `TT_SYSTEM_HEALTH_TIMEOUT` | `60` | Timeout in seconds for `Cluster.ReportSystemHealth` based device discovery |
| `TT_VISIBLE_DEVICES` | _set per worker_ | Set internally by the worker bootstrap to expose a single device id to a worker process |
| `TT_METAL_CACHE` | _set per worker_ | Set internally to `${TT_METAL_HOME}/built/<device_ids>` so each worker uses a distinct cache directory |
| `TT_MM_THROTTLE_PERF` | _runner-dependent_ | Set internally based on `DEFAULT_THROTTLE_LEVEL`; some DiT runners disable throttling automatically |
| `TT_MESH_GRAPH_DESC_PATH` | _runner-dependent_ | Set internally to point at the correct mesh graph descriptor for the current `device_mesh_shape` |

### Telemetry Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `ENABLE_TELEMETRY` | `True` | Boolean flag to enable or disable telemetry collection. When disabled, no metrics are recorded and background telemetry processes are not started |
| `PROMETHEUS_ENDPOINT` | `"/metrics"` | HTTP endpoint path where Prometheus metrics are exposed for scraping by monitoring systems |

## Authentication Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `API_KEY` | `"your-secret-key"` | Secret key used for API authentication. All requests must include `Authorization: Bearer <API_KEY>` header |
| `ORG_ID_HEADER` | `"X-TT-Organization"` | Name of the HTTP header that fine-tuning endpoints read to scope jobs to a tenant. Requests to `/v1/fine_tuning/*` must include this header |

## Hugging Face Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `HF_TOKEN` | `None` | Hugging Face token with read permission for accessing private models and datasets |
| `HF_HOME` | `None` | Directory path for Hugging Face cache and model storage |

## Special Environment Variable Overrides

The server supports special environment variable combinations that can override multiple settings at once:

| Environment Variable | Description |
|---------------------|-------------|
| `MODEL` | Specifies the model to run. Combined with `DEVICE`, overrides configuration based on predefined ModelConfigs |
| `DEVICE` | Specifies the target device type for model execution. Combined with `MODEL`, overrides configuration based on predefined ModelConfigs |

When both `MODEL` and `DEVICE` are set, the server will look up the corresponding configuration in [`ModelConfigs`](config/constants.py ) and apply all associated settings automatically.

## Telemetry

The TT Media Server provides comprehensive Prometheus metrics for monitoring performance and operational health. Telemetry can be enabled/disabled via the `ENABLE_TELEMETRY` environment variable.

### Observability stack

Metrics emission lives with each server; collection and visualization
live in a shared, top-level [`monitoring/`](./monitoring) directory:

- [`telemetry/`](./telemetry) — Python instrumentation that exposes `/metrics`.
- [`cpp_server/`](./cpp_server) — C++ instrumentation that exposes `/metrics`.
- [`monitoring/`](./monitoring) — Prometheus + Grafana + process-exporter
  Docker Compose stack that scrapes whichever server is running. Picks
  the dashboard via `SERVER_SERVICE` (`cpp` | `python`).

Quick start: see [`monitoring/README.md`](./monitoring/README.md).

### Available Metrics

#### Request Processing Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `tt_media_server_requests_base_counter` | Counter | Total base service requests | `model_type` |
| `tt_media_server_requests_base_duration_seconds` | Histogram | Base service request duration | `model_type` |
| `tt_media_server_requests_base_total` | Counter | Total base service method calls | `model_type` |
| `tt_media_server_requests_base_duration_seconds_total` | Histogram | Total base service method duration | `model_type` |

#### Processing Pipeline Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `tt_media_server_pre_processing_duration_seconds` | Histogram | Pre-processing stage duration | `model_type`, `preprocessing_enabled` |
| `tt_media_server_post_processing_duration_seconds` | Histogram | Post-processing stage duration | `model_type`, `post_processing_enabled` |

#### Model & Device Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `tt_media_server_model_inference_duration_seconds` | Histogram | Model inference execution time | `model_type`, `device_id` |
| `tt_media_server_model_inference_total` | Counter | Total model inference operations | `model_type`, `device_id`, `status` |
| `tt_media_server_device_warmup_duration_seconds` | Histogram | Device warmup time | `model_type`, `device_id` |
| `tt_media_server_device_warmup_total` | Counter | Total device warmup operations | `model_type`, `device_id`, `status` |

### Labels Description

Labels are part of the metrics. Example:
tt_media_server_device_warmup_duration_seconds_sum{device_id="2",model_type="tt-sdxl-trace"} 505.4703781604767

- **`model_type`**: The type of model being used (e.g., `SDXL`, `TT_SDXL_IMAGE_TO_IMAGE`)
- **`device_id`**: Logical index of the Tenstorrent device for that worker (devices are ordered by PCI bus address; on Galaxy this stays the same across reset). Not the same as the number in `/dev/tenstorrent/N`.
- **`status`**: Operation status (`success` or `failure`)
- **`preprocessing_enabled`**: Whether preprocessing is enabled (`true` or `false`)
- **`post_processing_enabled`**: Whether post-processing is enabled (`true` or `false`)

### Accessing Metrics

Metrics are available at the configured endpoint (default: `http://localhost:8000/metrics`) in Prometheus format.

## Device Mesh Configuration

The server supports special environment variables for configuring device mesh shapes for specific model configurations:

| Environment Variable | Device Mesh Shape | Description |
|---------------------|-------------------|-------------|
| `SD_3_5_FAST` | `None` | Configures device mesh for SD-3.5 in fast configuration (4x8 mesh = 32 devices total) when set to `"true"` (case-insensitive) |
| `SD_3_5_BASE` | `None` | Configures device mesh for SD-3.5 in base configuration (2x4 mesh = 8 devices total) when set to `"true"` (case-insensitive) |
| `TP2` | `None` | Enables tensor parallelism across 2 devices (2x1 mesh) when set to `"true"` (case-insensitive). **Compatible with SDXL models only** |
| `SP_MESH_4X32` | `None` | Configures device mesh as 4x32 (sequence-parallel mesh) when set to `"true"` (case-insensitive). Used for very large mesh deployments |

### Usage Examples

#### Running SDXL with Tensor Parallelism (TP2)
```bash
# Enable TP2 for SDXL (requires 2 devices)
export TP2=true
export MODEL_RUNNER=tt-sdxl-trace
source run_uvicorn.sh
```

**Note:** TP2 configuration is currently supported only for SDXL models and requires exactly 2 TT devices.

#### Running Stable Diffusion 3.5 Base Configuration
```bash
# SD-3.5 base setup (2x4 mesh = 8 devices)
export SD_3_5_BASE=true
export MODEL=stable-diffusion-3.5-large
export DEVICE=galaxy
source run_uvicorn.sh
```

#### Running Stable Diffusion 3.5 Fast Configuration
```bash
# SD-3.5 fast setup (4x8 mesh = 32 devices)
export SD_3_5_FAST=true
export MODEL=stable-diffusion-3.5-large
export DEVICE=galaxy
source run_uvicorn.sh
```

**Important Notes:**
- These environment variables override the default `DEVICE_MESH_SHAPE` setting
- SD-3.5 configurations require Galaxy hardware with sufficient devices or T3K

## Configuration File

The server also supports configuration via a `.env` file in the project root. Environment variables take precedence over `.env` file settings.

## Configuration Examples

### Basic Configuration
```bash
# Set log level to debug
export LOG_LEVEL=DEBUG

# Configure for specific devices only
# Brackets represent chip pairs that will be grouped together
export DEVICE_IDS="(0,1),(2,3)"
```

### High-Throughput Configuration
```bash
# Increase queue size for high-throughput scenarios
export MAX_QUEUE_SIZE=128

# Set custom timeout for long-running inferences
export REQUEST_PROCESSING_TIMEOUT_SECONDS=300
```

### Production Configuration
```bash
# Configure for production environment
export ENVIRONMENT=production
export LOG_FILE="/var/log/tt-inference-server.log"
export LOG_LEVEL=WARNING
```

### Model and Device Override
```bash
# Use predefined model/device configuration
export MODEL="stable-diffusion-xl-base-1.0"
export DEVICE="n300"
```

### Audio Processing Configuration
```bash
# Configure for longer audio files
export MAX_AUDIO_DURATION_SECONDS=300.0
export MAX_AUDIO_SIZE_BYTES=104857600  # 100 MB
export DEFAULT_SAMPLE_RATE=22050
export ALLOW_AUDIO_PREPROCESSING=true
```

### Authentication Configuration
```bash
# Set custom API key for authentication
export API_KEY="my-secure-secret-key-123"

# For production, use a strong random key
export API_KEY="$(openssl rand -base64 32)"
```

When `API_KEY` is set, all API requests must include the authorization header:
```bash
# Example with custom API key
curl -H "Authorization: Bearer my-secure-secret-key-123" \
     ...
```

### Development Configuration
```bash
# Use mock devices for development
export MOCK_DEVICES_COUNT=2
export DEVICE_IDS="(0),(1)"
export ENVIRONMENT=development
```


# Steps for Onboarding a Model to the Inference Server

If you're integrating a new model into the inference server, here’s a suggested workflow to help guide the process:

1. **Implement a Model Runner** Create a model runner by inheriting the *base_runner* class and implementing its abstract methods. You can find the relevant codebase here: [tt-inference-server/tt-media-server/tt_model_runners at dev · tenstorrent/tt-inference-server ](https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-media-server/tt_model_runners)
(most likely a model runner is a *demo.py* file from a model in tt-metal broken down in methods of a class)
2. **Update Dependencies** If your runner relies on any additional libraries, please make sure to add them to the requirements.txt:  [tt-inference-server/tt-media-server/requirements.txt at dev · tenstorrent/tt-inference-server ](https://github.com/tenstorrent/tt-inference-server/blob/dev/tt-media-server/requirements.txt)
3. **Modify *runner_fabric.py*** Update *runner_fabric.py* to instantiate your runner based on the configuration: [tt-inference-server/tt-media-server/tt_model_runners/runner_fabric.py at dev · tenstorrent/tt-inference-server ](https://github.com/tenstorrent/tt-inference-server/blob/dev/tt-media-server/tt_model_runners/runner_fabric.py)
4. **Add a Dummy Config** Add a basic config entry to help instantiate your runner: [tt-inference-server/tt-media-server/config/settings.py at dev · tenstorrent/tt-inference-server ](https://github.com/tenstorrent/tt-inference-server/blob/dev/tt-media-server/config/settings.py)
Alternatively, you can use an environment variable:
```export MODEL_RUNNER=<your-model-runner-name>```
5. **Write a Unit Test** Please include a unit test in the *tests/* folder to verify your runner works as expected. This step is crucial—without it, it’s difficult to pinpoint issues if something breaks later
6. **Open an Issue for CI Coverage** Kindly submit a GitHub issue for Igor Djuric to review your PR and to help cover end to end running, CI integration, or any missing service steps: [https://github.com/tenstorrent/tt-inference-server/issuesConnect your Github account ](https://github.com/tenstorrent/tt-inference-server/issues)
7. **Share Benchmarks (if available)** If you’ve run any benchmarks or evaluation tests, please share them. They’re very helpful for understanding performance and validating correctness.

# Docker build and run

Docker build sample:

```bash
docker build -t sdxl-inf-server --platform=linux/amd64 -f tt-media-server/Dockerfile .
```

Docker image link:

https://github.com/tenstorrent/tt-inference-server/pkgs/container/tt-inference-server%2Ftt-server-dev-ubuntu-22.04-amd64

Docker run sample:

```bash
docker run \
  -e MODEL_RUNNER=forge \
  --rm -it \
  -p 8000:8000 \
  --user root \
  --entrypoint "/bin/bash" \
  --device /dev/tenstorrent/0 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  ghcr.io/tenstorrent/tt-inference-server/tt-server-dev-ubuntu-22.04-amd64
```

**Suggestion:** Always take the latest docker image

## Galaxy running settings

Running SDXL on Galaxy:

```bash
sudo docker run -d -it \
  -e MODEL_RUNNER=tt-sdxl-trace \
  -e DEVICE_IDS="(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15),(16),(17),(18),(19),(20),(21),(22),(23)" \
  --cap-add=sys_nice \
  --security-opt seccomp=unconfined \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --device /dev/tenstorrent \
  -p 8000:8000 \
  --user root \
  --device /dev/ipmi0 \
  ghcr.io/tenstorrent/tt-inference-server/tt-server-dev-ubuntu-22.04-amd64
```

**Note:** Sample above will run 24 devices with numbers 0 to 23. Please note it'd be a good practice to mount only the devices you are planning to use to avoid collisions.

Running Whisper on Galaxy:

```bash
sudo docker run -d -it \
  -e MODEL_RUNNER=tt-whisper \
  -e DEVICE_IDS="(24),(25),(26)" \
  --cap-add=sys_nice \
  --security-opt seccomp=unconfined \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --device /dev/tenstorrent \
  -p 8000:8000 \
  --user root \
  --device /dev/ipmi0 \
  ghcr.io/tenstorrent/tt-inference-server/tt-server-dev-ubuntu-22.04-amd64
```

**Note:** Sample above will run Whisper model on devices 24 to 26 - 3 devices.

# Profiling

We use [py-spy](https://github.com/benfred/py-spy) to profile the server.
To profile the server, first run the media server:

```bash
uvicorn main:app --lifespan on --port 8000
```

The console will print the PID of the server and the worker process PID:
```
INFO:     Started server process [1388662]
2025-12-11 11:58:49,925 - INFO - Started worker 0 with PID 1388679
```

Then run the profiler in two separate terminals, once for the server and once for the worker:
```bash
py-spy record -o profile_server.svg --pid <PID>
py-spy record -o profile_worker.svg --pid <PID>
```

Output is a flame chart [see interactive example](./docs/profiling-example.svg).

How to read the flame chart:

| Color | Width | Meaning | Interpretation | Action Needed |
|-------|-------|---------|----------------|---------------|
| **Light/Green** | **Narrow** | Fast function, quick execution | Efficient code, no issues | Perfect! Ignore it |
| **Light/Green** | **Wide** | I/O bound or coordinator function | Lots of waiting (network, disk, async) or delegates work to many children | Check if waiting is necessary. Optimize I/O if possible |
| **Yellow/Orange** | **Narrow** | Moderate CPU work, short duration | Some computation, but not critical | Monitor, usually okay |
| **Yellow/Orange** | **Wide** | Moderate CPU work, long duration | Doing noticeable work across time | Investigate if it can be optimized |
| **Red/Dark** | **Narrow** | CPU-intensive but quick | Hot code, but doesn't run long | Low priority - fast enough despite intensity |
| **Red/Dark** | **Wide** | CPU-intensive AND long-running | BOTTLENECK! | TOP PRIORITY - Optimize this first! |

# Remaining work:

1. Add unit tests
2. Add API tests
3. Cleanup unused things in runners
