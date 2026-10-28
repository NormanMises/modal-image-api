# Modal image API

[English](README.md) · [简体中文](README.zh-CN.md)

Deploy image generation models on [Modal](https://modal.com/) with an OpenAI-compatible Images API.
Each model has its own GPU container and inference dependencies. All models share one API and one weight cache Volume.

## Supported models

| API model ID | Model | Runtime |
| --- | --- | --- |
| `z-image-turbo` | [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Native PyTorch |
| `kook-moss-guanlan` | [KOOK Guanlan](https://www.modelscope.cn/models/KookYan/Kook_Moss_SFT_guanlan) | ComfyUI, MOSS-Image-SFT + Guanlan LoRA |

Both models use L40S GPUs. Each request generates one PNG and returns it as Base64 JSON.
For MOSS, `size` sets the aspect ratio. The service calculates generation dimensions near 1.5MP and rounds both dimensions up to multiples of 16, so the image has at least 1,500,000 pixels.
For example, `1080x1920` produces `928x1648`. Alignment may slightly change the aspect ratio. The service returns the native image without cropping or resizing it afterward.

## Deploy

Use Python 3.12 and a Modal account with GPU access. Run these commands from the repository root.

1. Install the dependencies and sign in to Modal.

   ```bash
   pip install -r requirements.txt
   modal token new
   ```

2. Create the API authentication Secret.

   ```bash
   modal secret create image-api-auth API_KEY=replace-with-a-long-random-secret
   ```

   Clients use this `API_KEY` as their Bearer token. It is separate from your Modal login token. Keep the real key out of Git and client-side browser code.

3. Deploy the service.

   ```bash
   modal deploy images_app.py
   ```

Add `/v1` to the URL printed by Modal to get your API base URL.
The first GPU container downloads its model weights. Later containers reuse the cached files.

## Generate an image

Set your deployment URL and key.

```bash
export BASE_URL="https://your-modal-app.modal.run/v1"
export API_KEY="your-api-key"
```

List the available models with `GET /v1/models`.

```bash
curl "$BASE_URL/models" -H "Authorization: Bearer $API_KEY"
```

Generate an image with `POST /v1/images/generations`. Change `model` to select a different model.

```bash
curl "$BASE_URL/images/generations" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kook-moss-guanlan",
    "prompt": "a red ceramic cup on a wooden table, natural daylight",
    "size": "1080x1920",
    "response_format": "b64_json"
  }'
```

The image is in `data[0].b64_json`. Successful generation responses include an `x-request-id` header for finding the request in the logs.

<details>
<summary>Request fields, responses, and errors</summary>

| Field | Required | Behavior |
| --- | --- | --- |
| `model` | Yes | Use an ID returned by `/v1/models`. |
| `prompt` | Yes | A non-empty text prompt. |
| `size` | No | `WIDTHxHEIGHT`, with positive integers. Defaults to `1024x1024`. MOSS converts this to its generation dimensions as described above. Z-Image passes it to the native inference code. |
| `response_format` | No | Only `b64_json` is supported. |

Each request generates one image. The API does not expose `n`, `steps`, `quality`, or `seed` controls.

```json
{
  "created": 1790000000,
  "data": [{"b64_json": "<base64-encoded PNG>"}]
}
```

Errors use the OpenAI format.

```json
{
  "error": {
    "message": "prompt must be a non-empty string.",
    "type": "invalid_request_error",
    "param": "prompt",
    "code": null
  }
}
```

Invalid inputs return HTTP 400, invalid credentials return 401, and server failures return 500.

</details>

<details>
<summary>Use the OpenAI Python client and save the image</summary>

The client is included in `requirements.txt`. The example uses the environment variables above.

```python
import base64
import os
from pathlib import Path

from openai import OpenAI

client = OpenAI(api_key=os.environ["API_KEY"], base_url=os.environ["BASE_URL"])
result = client.images.generate(
    model="kook-moss-guanlan",
    prompt="a red ceramic cup on a wooden table, natural daylight",
    size="1024x1024",
    response_format="b64_json",
)
Path("image.png").write_bytes(base64.b64decode(result.data[0].b64_json))
```

MOSS generates `1232x1232` for this square request.

</details>

## Model settings

Sampling settings live in each model's code. Clients select the model, prompt, and aspect ratio through the API.

<details>
<summary>MOSS Guanlan weights and workflow settings</summary>

Guanlan runs ComfyUI with custom nodes disabled. It uses the API-export graph of the official `image_moss_image_t2i.json` template, adapted for the following weights and settings. ComfyUI, the template, and the model files use the revisions in [config.py](image_service/models/guanlan/config.py).

| Setting | Value |
| --- | --- |
| Base model | `moss_image_sft_bf16.safetensors` |
| Text encoder | `qwen3.5_9b_bf16.safetensors` |
| VAE | `flux2-vae_small.safetensors` |
| Guanlan LoRA | Native `LoraLoaderModelOnly` node, strength `1.0`, SHA256-verified download |
| Steps | `25` |
| CFG | `3.0` |
| Sampler / scheduler | `er_sde` / `simple` |
| Skip-layer guidance | Enabled, layer `22`, range `0` to `0.5` |
| Prompt enhancement | Disabled. The original prompt goes to text encoding. |
| Negative prompt | `worst quality, bad anatomy, watermark, jpeg artifact` |
| Seed | Random for each request |
| Generation dimensions | Target `1,500,000` pixels, then round each dimension up to a multiple of `16` |

BF16 in the filenames describes the weights. ComfyUI chooses runtime precision through its default policy; the text encoder currently runs in FP16.
Workflow wiring and size conversion are in [workflow.py](image_service/models/guanlan/workflow.py). Some sampling values come from the pinned template rather than local configuration constants.

</details>

<details>
<summary>Z-Image-Turbo settings</summary>

Z-Image uses its native PyTorch inference code, with BF16 weights, 8 steps, guidance `0.0`, and `_native_flash` attention by default. Compilation is disabled.
See [config.py](image_service/models/z_image/config.py) and [backend.py](image_service/models/z_image/backend.py).

</details>

## Development and operation

<details>
<summary>Project structure and adding a model</summary>

```text
images_app.py                  Deployment entry for the CPU API and model Apps
image_service/
  api.py                       HTTP validation, authentication, and responses
  contracts.py                 ImageBackend interface and model metadata
  registry.py                  Model metadata and generation functions
  deployment.py                Shared ModelWorker lifecycle and Modal calls
  catalog.py                   Enabled model deployments
  config.py                    Shared service names and cache paths
  storage.py                   Shared model-weight Volume
  encoding.py                  PNG / Base64 encoding
  timing.py                    Request IDs and structured timing logs
  comfy.py                     Internal ComfyUI process and workflow execution
  models/
    z_image/                   Model config, deployment, and backend
    guanlan/                   Model config, deployment, backend, LoRA, and workflow
```

The CPU API does not import inference frameworks. GPU containers initialize their backends at startup. Z-Image loads its weights directly; Guanlan starts ComfyUI, whose workflow nodes load the models on demand.

To add a model:

1. Create `image_service/models/<name>/` with `__init__.py`, `config.py`, `backend.py`, and `deployment.py`.
2. Subclass `ImageBackend`. Implement `load(cache_dir: str) -> None` and `generate(prompt: str, width: int, height: int) -> list[str]`. Return Base64 PNGs and import inference frameworks inside the backend methods.
3. Define the model's Modal App and image. Subclass `ModelWorker`, set `cache_dir`, and implement `create_backend()`. Use a unique worker class name, `with_source(image)`, and the shared cache Volume. Export a `ModelDeployment`.
4. Register the deployment in `MODEL_DEPLOYMENTS` in `image_service/catalog.py`.
5. Check the code, deploy, and verify generation.

Edit a model's `config.py` for its settings and `deployment.py` for its dependencies and GPU. The shared API and other models do not need changes.

</details>

<details>
<summary>Weight cache, scaling, and runtime</summary>

All weights share the `image-model-cache` Modal Volume, mounted at `/cache`. Each model uses `/cache/<model_id>`. Weights stay in the Volume rather than the container image.

Each GPU container handles one generation at a time. Modal starts more containers for parallel requests. Resource and idle scaling settings use Modal defaults apart from the configured image, GPU, cache mount, and `HF_HOME`.

The API and MOSS images use Python 3.12; Z-Image uses Python 3.13. Local Python runs the CLI, not GPU inference.
ComfyUI listens on `127.0.0.1:8188` inside the container. The backend reads PNG files directly, then deletes those files and the completed workflow history before returning the result.

</details>

<details>
<summary>Logs and credentials</summary>

```bash
modal app logs images-api
```

Structured entries have `event: "image_timing"`, a `stage`, and elapsed `seconds`. Generation entries share a `request_id` across the API and GPU container. Startup entries have no request ID.

Logs measure backend loading, cache commit, worker generation, the API's remote call, and response serialization. MOSS also records weight preparation, ComfyUI startup, workflow preparation and queueing, execution, image reading and encoding, and cleanup. Workflow preparation logs contain the actual aligned dimensions.
The API handler timing excludes delivery of the response to the client. Stages overlap, so do not add all durations together.

Store `API_KEY` in the `image-api-auth` Modal Secret. Keep credentials out of Git and rotate exposed keys. CORS currently allows all origins; browser clients should call through a server that holds the key.

</details>

## Upstream projects

[Z-Image](https://github.com/Tongyi-MAI/Z-Image) · [ComfyUI](https://github.com/Comfy-Org/ComfyUI) · [MOSS-Image weights](https://huggingface.co/Comfy-Org/MOSS-Image) · [Official MOSS template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_moss_image_t2i.json) · [Modal documentation](https://modal.com/docs)
