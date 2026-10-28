# Modal 图像生成 API

[English](README.md) · [简体中文](README.zh-CN.md)

在 [Modal](https://modal.com/) 上部署图像生成模型，提供兼容 OpenAI Images API 的接口。
每个模型使用独立的 GPU 容器和推理依赖，共用一个 API 和一个权重缓存 Volume。

## 支持的模型

| API 模型 ID | 模型 | 运行方式 |
| --- | --- | --- |
| `z-image-turbo` | [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | 原生 PyTorch |
| `kook-moss-guanlan` | [KOOK 观澜](https://www.modelscope.cn/models/KookYan/Kook_Moss_SFT_guanlan) | ComfyUI，MOSS-Image-SFT + 观澜 LoRA |

两个模型都使用 L40S。每次请求生成一张 PNG，以 Base64 JSON 返回。
MOSS 的 `size` 用于指定宽高比。服务先按比例换算到 150 万像素，再将宽高向上对齐到 16 的倍数，确保至少 1.5MP。
例如，`1080x1920` 会生成 `928x1648`。对齐可能使比例有轻微偏差，生成后不裁切、不缩放。

## 部署

使用 Python 3.12，并准备有 GPU 使用权限的 Modal 账号。在仓库根目录执行以下步骤。

1. 安装依赖并登录 Modal。

   ```bash
   pip install -r requirements.txt
   modal token new
   ```

2. 创建 API 鉴权 Secret。

   ```bash
   modal secret create image-api-auth API_KEY=replace-with-a-long-random-secret
   ```

   客户端用这里的 `API_KEY` 作为 Bearer token，它与 Modal 登录 token 无关。不要把真实密钥提交到 Git 或写进浏览器端代码。

3. 部署服务。

   ```bash
   modal deploy images_app.py
   ```

在 Modal 输出的应用地址后加 `/v1`，就是 API Base URL。
GPU 容器首次启动时下载对应模型的权重，后续容器复用缓存。

## 生成图片

设置部署地址和密钥。

```bash
export BASE_URL="https://your-modal-app.modal.run/v1"
export API_KEY="your-api-key"
```

通过 `GET /v1/models` 获取可用模型。

```bash
curl "$BASE_URL/models" -H "Authorization: Bearer $API_KEY"
```

通过 `POST /v1/images/generations` 生成图片。修改 `model` 即可切换模型。

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

图片位于返回值的 `data[0].b64_json`。生成成功时，响应头还包含 `x-request-id`，可用它查找对应日志。

<details>
<summary>请求字段、返回值和错误格式</summary>

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `model` | 是 | 使用 `/v1/models` 返回的模型 ID。 |
| `prompt` | 是 | 非空文本提示词。 |
| `size` | 否 | `WIDTHxHEIGHT`，宽高为正整数，默认 `1024x1024`。MOSS 按前述规则换算生成尺寸，Z-Image 将尺寸传给原生推理代码。 |
| `response_format` | 否 | 只支持 `b64_json`。 |

每次请求生成一张图片。API 不提供 `n`、`steps`、`quality` 或 `seed` 控制项。

```json
{
  "created": 1790000000,
  "data": [{"b64_json": "<base64-encoded PNG>"}]
}
```

错误使用 OpenAI 格式。

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

输入错误返回 HTTP 400，鉴权失败返回 401，服务端错误返回 500。

</details>

<details>
<summary>使用 OpenAI Python 客户端并保存图片</summary>

`requirements.txt` 已包含客户端依赖。示例使用上面设置的环境变量。

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

这个正方形请求会让 MOSS 生成 `1232x1232` 的图片。

</details>

## 模型设置

采样参数在各模型的代码中配置。客户端通过 API 选择模型、提示词和宽高比。

<details>
<summary>MOSS 观澜的权重和工作流参数</summary>

观澜在 ComfyUI 中运行，禁用自定义节点。工作流使用官方 `image_moss_image_t2i.json` 的 API 导出图，并按下表调整权重和参数。
ComfyUI、模板和模型文件的固定版本见 [config.py](image_service/models/guanlan/config.py)。

| 设置 | 当前值 |
| --- | --- |
| 底模 | `moss_image_sft_bf16.safetensors` |
| 文本编码器 | `qwen3.5_9b_bf16.safetensors` |
| VAE | `flux2-vae_small.safetensors` |
| 观澜 LoRA | 原生 `LoraLoaderModelOnly` 节点，强度 `1.0`，下载后校验 SHA256 |
| 采样步数 | `25` |
| CFG | `3.0` |
| 采样器 / 调度器 | `er_sde` / `simple` |
| 跳层引导 | 开启，第 `22` 层，范围 `0` 至 `0.5` |
| 提示词增强 | 关闭，原始提示词直接进入文本编码 |
| 负面提示词 | `worst quality, bad anatomy, watermark, jpeg artifact` |
| 种子 | 每次请求随机生成 |
| 生成尺寸 | 先换算到 `1,500,000` 像素，再将宽高向上对齐到 `16` 的倍数 |

文件名中的 BF16 指权重格式。运行精度由 ComfyUI 默认策略决定，文本编码器当前以 FP16 运行。
节点接线和尺寸换算见 [workflow.py](image_service/models/guanlan/workflow.py)。部分采样参数沿用固定版本的模板，并非全部定义在本地配置常量中。

</details>

<details>
<summary>Z-Image-Turbo 参数</summary>

Z-Image 使用原生 PyTorch 推理代码，权重精度为 BF16，采样 8 步，guidance 为 `0.0`，默认 attention 为 `_native_flash`，关闭编译。
参数见 [config.py](image_service/models/z_image/config.py) 和 [backend.py](image_service/models/z_image/backend.py)。

</details>

## 开发与运行

<details>
<summary>项目结构和新增模型</summary>

```text
images_app.py                  部署入口，组合 CPU API 和模型 App
image_service/
  api.py                       HTTP 校验、鉴权和响应
  contracts.py                 ImageBackend 接口和模型元数据
  registry.py                  模型元数据和生成函数注册表
  deployment.py                公共 ModelWorker 生命周期和 Modal 调用
  catalog.py                   启用的模型清单
  config.py                    公共服务名称和缓存路径
  storage.py                   共享权重 Volume
  encoding.py                  PNG / Base64 编码
  timing.py                    请求 ID 和分阶段计时日志
  comfy.py                     内部 ComfyUI 进程和工作流执行
  models/
    z_image/                   模型配置、部署和推理后端
    guanlan/                   模型配置、部署、推理后端、LoRA 和工作流
```

CPU API 不导入推理框架，GPU 容器在启动时初始化后端。Z-Image 直接加载权重，观澜先启动 ComfyUI，再由工作流节点按需加载模型。

新增模型的步骤：

1. 创建 `image_service/models/<name>/`，添加 `__init__.py`、`config.py`、`backend.py` 和 `deployment.py`。
2. 后端继承 `ImageBackend`，实现 `load(cache_dir: str) -> None` 和 `generate(prompt: str, width: int, height: int) -> list[str]`。生成结果为 Base64 PNG，推理框架在后端方法内导入。
3. 定义模型的 Modal App 和镜像。容器继承 `ModelWorker`，设置 `cache_dir` 并实现 `create_backend()`。使用唯一的容器类名、`with_source(image)` 和共享缓存 Volume，导出 `ModelDeployment`。
4. 在 `image_service/catalog.py` 的 `MODEL_DEPLOYMENTS` 中注册。
5. 检查代码，部署并验证出图。

模型参数改对应的 `config.py`，依赖和 GPU 改对应的 `deployment.py`，无需修改公共 API 或其他模型。

</details>

<details>
<summary>权重缓存、并行和运行环境</summary>

权重共用 `image-model-cache` Modal Volume，挂载到 `/cache`。各模型使用 `/cache/<model_id>`。权重保存在 Volume 中，不打包进容器镜像。

每个 GPU 容器同时处理一次生成，Modal 通过增加容器处理并行请求。除镜像、GPU、缓存挂载和 `HF_HOME` 外，资源与空闲缩容配置使用 Modal 默认值。

API 和 MOSS 镜像使用 Python 3.12，Z-Image 使用 Python 3.13。本地 Python 只运行 CLI，不执行 GPU 推理。
ComfyUI 在容器内监听 `127.0.0.1:8188`。后端直接读取 PNG 文件，并在返回结果前删除生成文件和该工作流的执行历史。

</details>

<details>
<summary>日志和密钥管理</summary>

```bash
modal app logs images-api
```

分阶段日志包含 `event: "image_timing"`、阶段名 `stage` 和耗时 `seconds`。生成日志通过 `request_id` 关联 API 与 GPU 容器，容器启动日志没有请求 ID。

日志记录后端加载、缓存提交、容器内生成、API 远程调用和响应序列化的耗时。MOSS 还记录权重准备、ComfyUI 启动、工作流准备与排队、执行、图片读取编码和清理耗时。工作流准备日志包含对齐后的实际宽高。
API 处理耗时不包含响应传到客户端的时间，各阶段有包含关系，不能将所有耗时相加。

将 `API_KEY` 保存在 `image-api-auth` Modal Secret 中，不要将凭据提交到 Git，泄露后及时轮换。当前 CORS 允许所有来源，浏览器客户端应通过持有密钥的服务端调用。

</details>

## 上游项目

[Z-Image](https://github.com/Tongyi-MAI/Z-Image) · [ComfyUI](https://github.com/Comfy-Org/ComfyUI) · [MOSS-Image 权重](https://huggingface.co/Comfy-Org/MOSS-Image) · [MOSS 官方模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_moss_image_t2i.json) · [Modal 文档](https://modal.com/docs)
