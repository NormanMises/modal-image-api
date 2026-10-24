# Repository Guidelines

## 项目结构

- `images_app.py`：唯一部署入口，组合模型 App 与轻量 CPU API。
- `image_service/api.py`：OpenAI Images 兼容 HTTP 接口、鉴权、参数校验、错误和响应。
- `image_service/contracts.py`：模型元数据与后端 `generate(prompt, width, height) -> list[str]` 接口。
- `image_service/registry.py`：元数据和生成函数统一注册；API 不依赖 Modal 或具体模型。
- `image_service/deployment.py`：Modal 组合、代码打包和远程调用适配。
- `image_service/catalog.py`：唯一的启用模型清单。
- `image_service/config.py`：App 名称、共享鉴权 Secret 等通用服务配置。
- `image_service/encoding.py`：PNG / Base64 序列化。
- `image_service/storage.py`：唯一共享权重 Volume `image-model-cache`，挂载到 `/cache`，各模型使用 `/cache/<model_id>`。
- `image_service/models/<name>/`：每个模型独立的 `config.py`、`deployment.py` 和 `backend.py`。观澜另有 LoRA 与采样模块。

新增模型只需新增模型目录并在 catalog 中注册，不改 API 或其他模型。模型配置放在对应
`config.py`；镜像依赖、GPU 和容器生命周期放在对应 `deployment.py`。所有模型共用 storage 中的 Volume，不单独创建模型 Volume。
所有模型均在各自容器启动钩子内导入后端并加载权重；禁止在包初始化或部署导入期间
加载 torch、diffusers 等推理依赖。`__init__.py` 保持轻量。

## 构建、调试与部署

- Windows 使用 `deep` conda/mamba 环境；Mac 使用仓库内 `.venv`。
- Mac 命令：`.venv/bin/python` 和 `.venv/bin/modal`。
- 语法检查：`.venv/bin/python -m compileall -q images_app.py image_service`
- 部署：`.venv/bin/modal deploy images_app.py`
- 保持 App `images-api`、函数 `fastapi_app` 以及现有容器类名稳定。
- 每个 GPU 容器同时处理一次生成；通过多个容器实现请求并行。
- 运行参数使用 Modal 和依赖库默认值，不额外覆盖 memory、timeout、scaledown_window、分词并行或下载性能参数。仅配置模型镜像、GPU、缓存挂载及必要的 HF_HOME。

## API 与 Secret

- `GET /v1/models`；`POST /v1/images/generations`。
- 客户端使用 `Authorization: Bearer <API_KEY>`；密钥放在 Modal Secret `image-api-auth`。
- `model` 必填，取值来自 catalog 注册的模型，具体列表由 `/v1/models` 返回。
- `prompt` 必填；`size` 默认 `1024x1024`，宽高为正整数。
- 观澜宽高向上对齐到 16 的倍数，返回完整原生图片，不裁切、不缩放。
- 每请求一张图；`response_format` 只支持 `b64_json`。不提供 n、steps、quality 或 seed。
- OpenAI 标准错误格式；400 `invalid_request_error`，401 `authentication_error`，500 `server_error`。

## 实现与验证

- 代码保持简洁高效易懂，直接按内部接口约定调用，不添加推测性的兜底、重复检查或备用路径。
- 保留外部输入校验、鉴权和下载权重的完整性校验。
- 改动后执行语法检查；涉及 API 时验证鉴权、错误和模型分发。
- 涉及模型时验证加载、采样和尺寸处理；涉及部署时实际部署并做小尺寸线上验证。
- 临时测试、对比、测量脚本和中间输出用完即删除，不提交；文档不保留已删除脚本的引用。
- 不提交 `.env`、HF token、Modal token 或密钥。
- 观澜 `sampling.py` 改编自 ComfyUI，保留 GPL-3.0 归属和根目录 `LICENSE-ComfyUI.txt`。
