# Modal 图像生成模型 API

[English](README.md) · [简体中文](README.zh-CN.md)

在 [Modal](https://modal.com/) 上部署图像生成模型，统一提供兼容 OpenAI Images API 的接口。
每个模型独立管理推理代码、运行依赖和 GPU 配置；权重共用一个 `image-model-cache` Volume，按模型分目录。通过模型清单接入，复用公共 API 和部署入口。

预置模型：

| `model` id | 模型 | 后端 |
|---|---|---|
| `z-image-turbo` | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Z-Image 原生 PyTorch |
| `kook-moss-guanlan` | [KOOK 观澜](https://www.modelscope.cn/models/KookYan/Kook_Moss_SFT_guanlan) | MOSS-Image-SFT + LoRA |

观澜使用独立 L40S 容器，权重放在共享 `image-model-cache` Volume 的 `kook-moss-guanlan/` 目录，底模为
MOSS-Image-SFT。ComfyUI 和[官方 `image_moss_image_t2i.json` 模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_moss_image_t2i.json)
固定到提交版本。容器实际运行 ComfyUI，禁用自定义节点，使用官方 API 导出图中的原生加载、
文本生成/编码、KSampler、VAE 解码和图片输出节点。

底模使用 `moss_image_sft_bf16.safetensors`，文本编码器使用 `qwen3.5_9b_bf16.safetensors`，
VAE 保留模板的 `flux2-vae_small.safetensors`。观澜 LoRA 按 SHA256 校验后，通过原生
`LoraLoaderModelOnly` 节点以强度 1 加载。

使用 25 步、CFG 3、`er_sde / simple`，开启第 22 层、0～0.5 范围的跳层引导。
关闭提示词增强，原始提示词直接进入文本编码。
负面词为 `worst quality, bad anatomy, watermark, jpeg artifact`（不含 `nsfw`）。
每次请求更换采样种子。在官方 ComfyUI 模板上调整了 CFG、跳层、提示词增强、模型精度和 LoRA 加载。
宽高可以是任意正整数。MOSS 将宽高向上对齐到 16 的倍数，直接返回生成的完整图片，
不裁切、不缩放；对齐可能使比例有轻微偏差。例如请求 `941x1672`，生成并返回
`944x1680`；已经对齐的尺寸直接返回。非正数仍返回 HTTP 400 / `param: "size"`。
ComfyUI 管理模型显存，Modal 资源、空闲缩容和分词并行使用默认值。ComfyUI 仅监听容器内部地址，
客户端继续调用原有 OpenAI 兼容 API。图片返回后清理生成文件及该请求的执行历史。

Mac 本地环境：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install 'modal[api-proxy-support]' fastapi
.venv/bin/modal token new
.venv/bin/modal deploy images_app.py
```

调用观澜时，把生成请求的 `model` 设置为 `kook-moss-guanlan`；API 地址和鉴权沿用现有服务。

服务提供：

- `GET /v1/models` —— 列出全部模型
- `POST /v1/images/generations` —— 每次请求生成一张图片，以 Base64 JSON 返回
- 通过请求里的 `model` 字段分发到对应模型（必填）
- OpenAI 标准错误格式：`{"error": {"message", "type", "param", "code"}}`

## 部署要求

- 已开通 GPU 的 Modal 账号
- Modal CLI

本地 Python 版本只需要满足 Modal CLI 的要求。部署到 Modal 的远端运行镜像内部使用 Python 3.13（Z-Image 模型）和 3.12（MOSS）。

Z-Image-Turbo 是公开模型，不需要 Hugging Face token。

## 部署

安装并登录 Modal CLI：

```bash
python -m pip install modal
modal setup
```

创建 API 使用的 Secret：

```bash
modal secret create image-api-auth API_KEY=替换为一个较长的随机密钥
```

该 Secret 必须包含 `API_KEY`。不要把真实密钥提交到 Git，也不要写进浏览器端代码。

`API_KEY` 就是客户端调用本 API 时放在 `Authorization: Bearer ...` 中的 key，与 Modal 登录 token 是两回事。

在仓库根目录执行部署：

```bash
modal deploy images_app.py
```

Modal 按各模型的 `deployment.py` 构建独立镜像；`images_app.py` 组合模型 App 和轻量 API，并输出应用地址。给这个地址加上 `/v1`，就是 API Base URL：

```text
https://your-modal-app.modal.run/v1
```

所有模型权重保存在同一个 `image-model-cache` Volume，挂载到 `/cache`。
各模型使用 `/cache/<model_id>`，例如 `/cache/z-image-turbo` 和 `/cache/kook-moss-guanlan`。
新容器直接复用已有权重。

## 项目结构

```text
images_app.py                   # 部署入口：组合模型与 CPU API
image_service/
  api.py                        # HTTP 输入、鉴权、错误和响应
  contracts.py                  # ImageBackend 抽象接口和模型元数据
  registry.py                   # 元数据与生成函数统一注册
  deployment.py                 # ModelWorker 基类、App 组合和远程调用
  catalog.py                    # 启用的模型清单
  config.py                     # 通用 App 和鉴权 Secret 名称
  encoding.py                   # PNG / Base64 序列化
  comfy.py                      # 内部 ComfyUI 服务与工作流执行
  storage.py                    # 唯一共享权重 Volume
  models/
    z_image/
      config.py                 # 模型标识、固定版本、生成默认值
      deployment.py             # 依赖、GPU、Volume、容器生命周期
      backend.py                # 原生 Z-Image 加载和推理
    guanlan/
      config.py
      deployment.py
      backend.py                # 准备权重并运行原生工作流
      lora.py                   # LoRA 下载和 SHA256 校验
      workflow.py               # 官方节点图的 BF16 模型与 LoRA 接线
```

两个后端统一在 GPU 容器启动钩子中初始化；Z-Image 直接加载模型，观澜启动 ComfyUI，
由原生工作流节点按需加载并缓存模型。API 镜像只安装
FastAPI；GPU 镜像各自安装推理依赖。容器启动前的模块导入均不加载推理框架。
App、模型容器类和接口名称沿用原配置。

## 新增模型

1. 在 `image_service/models/<name>/` 添加 `__init__.py`、`config.py`、`backend.py`
   和 `deployment.py`。推理框架在后端加载或生成时导入。
2. 后端继承 `image_service.contracts.ImageBackend`，实现统一接口：
   `load(cache_dir: str) -> None` 和 `generate(prompt: str, width: int, height: int) -> list[str]`。
   每个容器加载一次模型；生成返回 Base64 PNG，序列化复用 `image_service.encoding.image_to_base64`。
3. 在 `deployment.py` 定义模型的 `modal.App` 和镜像。容器继承公共 `ModelWorker`，
   设置 `cache_dir`，实现 `create_backend() -> ImageBackend`，在其中延迟导入并返回 `Backend()`。
   基类统一负责启动加载、缓存提交和远程 `generate`，子类不重复实现这些方法。
   导出 `ModelDeployment(app, MODEL_INFO, Worker)`。容器类名须唯一；用 `with_source(image)`
   打包本地代码。挂载 `image_service.storage.model_cache` 到 `CACHE_ROOT`，
   模型缓存路径使用公共配置的 `model_cache_dir(MODEL_INFO.id)`。每个 GPU 容器同时生成一张图，通过增加容器实现并行。
4. 在 `image_service/catalog.py` 的 `MODEL_DEPLOYMENTS` 中注册。
   模型列表、元数据和请求分发都取自这份注册表。
5. 执行本地检查，再运行 `modal deploy images_app.py`。

通用服务名称集中在 `image_service/config.py`，共享鉴权 Secret 为 `image-api-auth`，
与启用了哪些模型无关。

模型参数改对应的 `config.py`，依赖、GPU 资源和缩容时间改对应的 `deployment.py`。
无需修改 API 或其他模型。模块组合方式参见 [Modal 项目结构文档](https://modal.com/docs/guide/project-structure)。

## 认证

使用标准 OpenAI 请求头发送 Secret 中的值：

```http
Authorization: Bearer your-api-key
```

## API 调用

先设置地址和 API key：

```bash
export BASE_URL="https://your-modal-app.modal.run/v1"
export API_KEY="your-api-key"
```

获取模型列表：

```bash
curl "$BASE_URL/models" \
  -H "Authorization: Bearer $API_KEY"
```

生成图片：

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

请求字段：

- `model`：必填，选择 `GET /v1/models` 返回的模型 ID
- `prompt`：必填的文本提示词
- `size`：`WIDTHxHEIGHT`，默认 `1024x1024`
- `response_format`：只支持 `b64_json`

每次请求只生成一张图片，不提供 `n`、`steps`、`quality` 或 `seed`。尺寸处理交给各模型的推理实现。

错误遵循 OpenAI 格式：400 为 `invalid_request_error`，401 为 `authentication_error`，500 为 `server_error`。

也可以使用 OpenAI Python 客户端：

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

## 镜像和模型权重

仓库在 Python 文件中定义 Modal Image，并将 Z-Image 源码依赖固定到经过验证的提交。每个 Modal workspace 都会根据这份定义构建运行镜像。

模型权重保存在 Modal Volume 中，不打包进镜像。这样镜像更小，新实例也能复用已经下载的权重。

其他镜像用法参见 [Modal Existing Images 官方文档](https://modal.com/docs/guide/existing-images)。

## 本地检查

```bash
python -m compileall -q images_app.py image_service
```

## 安全

- 将 `API_KEY` 保存在 `image-api-auth` Modal Secret 中。
- 不要提交 `.env`、`.dev.vars`、Modal token 或其他凭据。
- 如果 key 泄露，请立即轮换。
- 当前 CORS 允许所有来源。面向不受信任的浏览器客户端时，请限制 `allow_origins`。

## 上游项目

- [Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image)
- [Z-Image-Turbo 模型](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Modal 文档](https://modal.com/docs)
