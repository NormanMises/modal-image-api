# Modal Image Model API

[English](README.md) · [简体中文](README.zh-CN.md)

Deploy image generation models on [Modal](https://modal.com/) behind an OpenAI-compatible Images API.
Each model owns its inference code, runtime dependencies, and GPU settings.
Weights share one `image-model-cache` Volume, with a separate directory per model.
Add models through the catalog while reusing the shared API and deployment entry.

Included model integrations:

| `model` id | Model | Backend |
|---|---|---|
| `z-image-turbo` | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Z-Image native PyTorch |
| `kook-moss-guanlan` | [KOOK Guanlan](https://www.modelscope.cn/models/KookYan/Kook_Moss_SFT_guanlan) | MOSS-Image-SFT + LoRA |

Guanlan uses a separate L40S container and the `kook-moss-guanlan/` directory in the shared `image-model-cache` Volume.
ComfyUI and the [official `image_moss_image_t2i.json` template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_moss_image_t2i.json)
are pinned to commits. The container runs ComfyUI itself with custom nodes disabled.
The official API-export graph uses native loaders, text generation/encoding,
`KSampler`, VAE decoding, and image output nodes. The model and text encoder are
`moss_image_sft_bf16.safetensors` and `qwen3.5_9b_bf16.safetensors`; the VAE remains
`flux2-vae_small.safetensors`. The verified Guanlan LoRA is applied by the native
`LoraLoaderModelOnly` node at strength 1.

Template settings are retained: 25 steps, CFG 2.5, `er_sde / simple`, prompt
enhancement enabled, and skip-layer guidance disabled. The negative prompt is
`worst quality, bad anatomy, watermark, jpeg artifact` (without `nsfw`). The sampling
seed changes for each request. These are ComfyUI template settings, rather than
the standalone OpenMOSS Python settings.
Any positive integer dimensions are accepted. MOSS rounds its
generation dimensions up to multiples of 16 and returns the complete native image,
without cropping or resizing. The aspect ratio may change slightly due to alignment.
For example, `941x1672` generates and returns `944x1680`.
Aligned sizes pass through unchanged; non-positive sizes return HTTP 400 / `param: "size"`.
ComfyUI manages model memory. Modal resources, idle scaling, and tokenizer
parallelism use their defaults. ComfyUI listens only on the container loopback
interface; clients continue using the existing OpenAI-compatible API. Generated
PNG files and completed request history are deleted after returning the images.
The first request downloads the base model.

On macOS, use a local virtual environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install 'modal[api-proxy-support]' fastapi
.venv/bin/modal token new
.venv/bin/modal deploy images_app.py
```

Select `"model": "kook-moss-guanlan"` using the existing endpoint and API key.

The service provides:

- `GET /v1/models` — lists all registered models
- `POST /v1/images/generations` — generates one image per request, returned as Base64 JSON
- Model dispatch via the required `model` request field
- OpenAI-standard error format: `{"error": {"message", "type", "param", "code"}}`

## Requirements

- A Modal account with GPU access
- Modal CLI

The local Python version only needs to be supported by the Modal CLI. The deployed containers use Python 3.13 (Z-Image models) and 3.12 (MOSS) internally.

Z-Image-Turbo is public and does not require a Hugging Face token.

## Deploy

Install and authenticate the Modal CLI:

```bash
python -m pip install modal
modal setup
```

Create the Secret used by the API:

```bash
modal secret create image-api-auth API_KEY=replace-with-a-long-random-secret
```

The Secret must contain `API_KEY`. Keep the real value out of Git and out of browser code.

`API_KEY` is the key that API clients send in `Authorization: Bearer ...`. It is separate from your Modal login token.

Deploy from the repository root:

```bash
modal deploy images_app.py
```

Modal builds each model image from its own `deployment.py`; `images_app.py` composes the model Apps and the lightweight API and prints the application URL. Add `/v1` to that URL to get the API base URL:

```text
https://your-modal-app.modal.run/v1
```

All model weights live in one Modal Volume, `image-model-cache`, mounted at `/cache`.
Each model uses `/cache/<model_id>` (for example, `/cache/z-image-turbo` and
`/cache/kook-moss-guanlan`). Later containers reuse the cached weights.

## Structure

```text
images_app.py                   # Deploy entry: compose models and the CPU API
image_service/
  api.py                        # HTTP input, authentication, errors, responses
  contracts.py                  # ImageBackend abstract interface and metadata
  registry.py                   # Model metadata and callable in one registry
  deployment.py                 # ModelWorker base, App composition, remote calls
  catalog.py                    # Enabled model deployments
  config.py                     # Shared App and authentication Secret names
  encoding.py                   # PNG / Base64 serialization
  comfy.py                      # Internal ComfyUI server and workflow execution
  storage.py                    # One shared model-weight Volume
  models/
    z_image/
      config.py                 # Identity, source pin, generation defaults
      deployment.py             # Dependencies, GPU, Volume, load/generate lifecycle
      backend.py                # Native Z-Image loading and inference
    guanlan/
      config.py
      deployment.py
      backend.py                # Prepare weights and execute the native workflow
      lora.py                   # LoRA download and SHA256 verification
      workflow.py               # Official graph with BF16 loaders and LoRA wiring
```

Both backends initialize inside their GPU container startup hook. Z-Image loads
its model directly; Guanlan starts ComfyUI, which loads and caches models through
the native workflow nodes. The API image installs FastAPI only; each GPU image installs
its own inference dependencies. All imports outside container startup are lightweight.
Existing App, worker, and endpoint names remain stable.

## Adding a model

1. Add `image_service/models/<name>/` with an `__init__.py`, `config.py`, `backend.py`,
   and `deployment.py`. Keep inference imports inside backend loading/generation.
2. Subclass `ImageBackend` from `image_service.contracts`. Implement
   `load(cache_dir: str) -> None` and `generate(prompt: str, width: int, height: int) -> list[str]`.
   Loading happens once per container; generation returns Base64 PNGs. Reuse
   `image_service.encoding.image_to_base64` for serialization.
3. In `deployment.py`, define the model's `modal.App` and image. Subclass the shared
   `ModelWorker`, set `cache_dir`, and implement `create_backend() -> ImageBackend`
   with a lazy import and `return Backend()`. The base provides startup loading,
   cache commit, and the remote `generate` method. Export `ModelDeployment(app, MODEL_INFO, Worker)`. Use a unique worker class
   name and `with_source(image)` to include the local package. Limit each GPU
   container to one generation at a time; scale with more containers. Mount
   `image_service.storage.model_cache` at `CACHE_ROOT` and use `model_cache_dir(MODEL_INFO.id)`
   from the shared config for the model's cache directory.
4. Add the deployment to `MODEL_DEPLOYMENTS` in `image_service/catalog.py`.
   Metadata, `/v1/models`, and generation routing use that same registration.
5. Run the local check and `modal deploy images_app.py`.

Shared service names are in `image_service/config.py`; the shared authentication
Secret is `image-api-auth`, independent of the enabled models.

For model settings, edit its `config.py`; for dependencies, GPU resources, or idle
scaling, edit its `deployment.py`. No API or other model changes are needed.
The package composition follows [Modal's project structure guide](https://modal.com/docs/guide/project-structure).

## Authentication

Send the Secret value in the standard OpenAI header:

```http
Authorization: Bearer your-api-key
```

## API

Set the URL and key in your shell:

```bash
export BASE_URL="https://your-modal-app.modal.run/v1"
export API_KEY="your-api-key"
```

List available models:

```bash
curl "$BASE_URL/models" \
  -H "Authorization: Bearer $API_KEY"
```

Generate an image:

```bash
curl "$BASE_URL/images/generations" \
  -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-image-turbo",
    "prompt": "a bold typographic poster",
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

The request fields are:

- `model`: required, choose an ID returned by `GET /v1/models`
- `prompt`: required text prompt
- `size`: `WIDTHxHEIGHT`, default `1024x1024`
- `response_format`: `b64_json`

The endpoint always generates one image. It does not expose `n`, `steps`, `quality`, or `seed`. Size handling is delegated to each model's inference implementation.

Errors follow the OpenAI format; status codes are 400 `invalid_request_error`, 401 `authentication_error`, 500 `server_error`.

Use the OpenAI Python client if you prefer an SDK:

```bash
python -m pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://your-modal-app.modal.run/v1",
)

result = client.images.generate(
    model="z-image-turbo",
    prompt="a bold typographic poster",
    size="1024x1024",
    response_format="b64_json",
)
```

## Image and model weights

The repository defines the Modal Images in Python and pins the Z-Image source dependency to a tested commit. Each Modal workspace builds the images from those definitions.

The model weights stay in the shared Modal Volume rather than in the images. This keeps image builds smaller and lets new containers reuse the cached weights.

For other image workflows, see the [Modal existing images guide](https://modal.com/docs/guide/existing-images).

## Local check

```bash
python -m compileall -q images_app.py image_service
```

## Security

- Store `API_KEY` in the `image-api-auth` Modal Secret.
- Never commit `.env`, `.dev.vars`, Modal tokens, or other credentials.
- Rotate the key if it is exposed.
- CORS currently allows all origins. Restrict `allow_origins` before serving untrusted browser clients.

## Upstream

- [Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image)
- [Z-Image-Turbo model](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Modal documentation](https://modal.com/docs)
