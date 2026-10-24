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
MOSS_STEPS = 25
MOSS_GUIDANCE = 3.0
MOSS_NEGATIVE_PROMPT = "worst quality, bad anatomy, watermark, jpeg artifact"
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
        # 文本编码器、DiT、VAE 按顺序上 GPU，避免同时驻留导致显存溢出。
        self.pipe.enable_model_cpu_offload()

    def generate(
        self, prompt: str, width: int, height: int, *, sampler: str = "er_sde", seed: int | None = None,
        negative_prompt: str = MOSS_NEGATIVE_PROMPT,
    ) -> list[str]:
        import torch
        from comfy_moss_sampling import sample_er_sde, simple_sigmas, uncond_skip_active

        if width <= 0 or height <= 0:
            raise ValueError("MOSS 图片宽高必须是正数")
        # Generate directly on the supported grid and return the complete image.
        width = ((width + 15) // 16) * 16
        height = ((height + 15) // 16) * 16
        if sampler not in {"er_sde", "official_euler"}:
            raise ValueError(f"未知 MOSS 采样方案: {sampler}")
        try:
            with torch.inference_mode():
                device = self.pipe._execution_device
                positive, positive_mask = self.pipe.encode_prompt(prompt)
                negative, negative_mask = self.pipe.encode_prompt(negative_prompt)
                shape = (1, self.pipe.transformer.config.latent_channels, height // 16, width // 16)
                # ComfyUI starts from CPU fp32 noise; subsequent ER-SDE noise uses the GPU.
                generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None
                latents = torch.randn(shape, dtype=torch.float32, device="cpu", generator=generator).to(device)

                def denoise(x, sigma):
                    # ComfyUI's flow t is 0..1, but the standalone MOSS transformer
                    # consumes the training timestep 0..1000 (not normalized sigma).
                    timestep = sigma * self.pipe.scheduler.config.num_train_timesteps
                    cond = self.pipe.transformer(x, timestep, positive, positive_mask).sample.float()
                    skip = {"skip_layers": [22]} if uncond_skip_active(sigma[0]) else {}
                    uncond = self.pipe.transformer(
                        x, timestep, negative, negative_mask, **skip
                    ).sample.float()
                    velocity = uncond + MOSS_GUIDANCE * (cond - uncond)
                    return x - sigma.reshape(-1, 1, 1, 1) * velocity

                if sampler == "er_sde":
                    extra_args = {"seed": seed} if seed is not None else {}
                    latents = sample_er_sde(
                        lambda x, sigma, **_: denoise(x, sigma),
                        latents,
                        simple_sigmas(MOSS_STEPS).to(device),
                        extra_args=extra_args,
                    )
                else:
                    scheduler = type(self.pipe.scheduler).from_config(self.pipe.scheduler.config)
                    scheduler.set_timesteps(MOSS_STEPS, device=device)
                    for index, timestep in enumerate(self.pipe.progress_bar(scheduler.timesteps)):
                        sigma = scheduler.sigmas[index].to(device)
                        denoised = denoise(latents, sigma.expand(1))
                        velocity = (latents - denoised) / sigma
                        latents = scheduler.step(velocity, timestep, latents, return_dict=False)[0]
                image = self._decode_latents(latents)
        finally:
            self.pipe.maybe_free_model_hooks()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return [base64.b64encode(buffer.getvalue()).decode("ascii")]

    def _decode_latents(self, latents):
        """官方 MOSS 管线的 FLUX.2 latent 反归一化与解码。"""
        vae = self.pipe.vae
        z = latents.to(dtype=vae.dtype)
        bn = vae.bn
        z = z * (bn.running_var + vae.config.batch_norm_eps).sqrt().to(z).view(1, -1, 1, 1)
        z = z + bn.running_mean.to(z).view(1, -1, 1, 1)
        batch, channels, height, width = z.shape
        z = z.reshape(batch, channels // 4, 2, 2, height, width)
        z = z.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, 2 * height, 2 * width)
        images = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
        return self.pipe.image_processor.postprocess(images, output_type="pil")[0]
