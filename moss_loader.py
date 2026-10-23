"""MOSS-Image-SFT + 观澜 LoRA；严格校验后合并 AI Toolkit 的 PEFT 权重。"""

import base64
import hashlib
import io
from pathlib import Path
import tempfile
import urllib.request

MOSS_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
MOSS_REVISION = "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9"
LORA_REVISION = "f19ce490b7cee2c10700d44bda6bbbc8a3f0716d"
LORA_SHA256 = "52e41bc8ab3964b11104332bc7fc742571e5fd1eaf7bc046285a5b37c56d3020"
LORA_URL = (
    "https://modelscope.cn/api/v1/models/KookYan/Kook_Moss_SFT_guanlan/repo"
    f"?Revision={LORA_REVISION}&FilePath=KOOK-MOSS-SFT-%E8%A7%82%E6%BE%9C.safetensors"
)


def download_lora(cache_dir: str) -> Path:
    path = Path(cache_dir) / "kook-moss-guanlan.safetensors"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == LORA_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # 多个冷启动实例共享 Volume，各自下载到独立临时文件再原子替换。
    with tempfile.TemporaryDirectory(prefix=".guanlan-", dir=path.parent) as temp_dir:
        temporary = Path(temp_dir) / path.name
        with urllib.request.urlopen(LORA_URL, timeout=120) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != LORA_SHA256:
            raise ValueError("观澜 LoRA SHA256 校验失败")
        temporary.replace(path)
    return path


def merge_lora(transformer, state: dict) -> int:
    """AI Toolkit PEFT 导出不另带 alpha，使用 B @ A，强度为 1。"""
    import torch

    pairs = []
    consumed = set()
    for key, a in state.items():
        if not key.startswith("diffusion_model.") or not key.endswith(".lora_A.weight"):
            continue
        stem = key.removesuffix(".lora_A.weight")
        b_key = stem + ".lora_B.weight"
        if b_key not in state:
            raise ValueError(f"缺少 LoRA B: {stem}")
        b = state[b_key]
        module = transformer.get_submodule(stem.removeprefix("diffusion_model."))
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"无效 LoRA 形状: {stem}")
        if tuple(module.weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(f"LoRA 与底模形状不匹配: {stem}")
        pairs.append((module.weight, a, b))
        consumed.update((key, b_key))
    if not pairs or consumed != set(state):
        raise ValueError(f"LoRA 含未识别权重: {sorted(set(state) - consumed)}")
    with torch.no_grad():
        for weight, a, b in pairs:
            delta = b.float() @ a.float()
            weight.add_(delta.to(device=weight.device, dtype=weight.dtype))
    return len(pairs)


class MossBackend:
    def __init__(self, cache_dir: str):
        import torch
        from diffusers import DiffusionPipeline
        from safetensors.torch import load_file

        self.pipe = DiffusionPipeline.from_pretrained(
            MOSS_REPO,
            revision=MOSS_REVISION,
            custom_pipeline="pipeline_moss_image",
            custom_revision=MOSS_REVISION,
            trust_remote_code=True,
            torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
            cache_dir=cache_dir,
        )
        self.merged_layers = merge_lora(
            self.pipe.transformer, load_file(str(download_lora(cache_dir)))
        )
        self.pipe.scheduler = type(self.pipe.scheduler).from_config(self.pipe.scheduler.config)
        # 文本编码器、DiT、VAE 按顺序上 GPU，避免同时驻留导致显存溢出。
        self.pipe.enable_model_cpu_offload()

    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        import torch

        if width <= 0 or height <= 0 or width % 16 or height % 16:
            raise ValueError("MOSS 图片宽高必须是正数且为 16 的倍数")
        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=50,
                guidance_scale=3.0,
                uncond_skip_layers=[22],
            ).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return [base64.b64encode(buffer.getvalue()).decode("ascii")]
