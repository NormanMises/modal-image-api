"""MOSS-Image inference; weight preparation and sampling are separate modules."""

from ...encoding import image_to_base64
from .config import MOSS_GUIDANCE, MOSS_NEGATIVE_PROMPT, MOSS_REPO, MOSS_REVISION, MOSS_STEPS, SKIP_LAYER
from .lora import download_lora, merge_lora


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

    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        import torch
        from .sampling import sample_er_sde, simple_sigmas, uncond_skip_active

        # Generate directly on the supported grid and return the complete image.
        width = ((width + 15) // 16) * 16
        height = ((height + 15) // 16) * 16
        try:
            with torch.inference_mode():
                device = self.pipe._execution_device
                positive, positive_mask = self.pipe.encode_prompt(prompt)
                negative, negative_mask = self.pipe.encode_prompt(MOSS_NEGATIVE_PROMPT)
                shape = (1, self.pipe.transformer.config.latent_channels, height // 16, width // 16)
                # ComfyUI starts from CPU fp32 noise; subsequent ER-SDE noise uses the GPU.
                latents = torch.randn(shape, dtype=torch.float32, device="cpu").to(device)

                def denoise(x, sigma):
                    # ComfyUI's flow t is 0..1, but the standalone MOSS transformer
                    # consumes the training timestep 0..1000 (not normalized sigma).
                    timestep = sigma * self.pipe.scheduler.config.num_train_timesteps
                    cond = self.pipe.transformer(x, timestep, positive, positive_mask).sample.float()
                    skip = {"skip_layers": [SKIP_LAYER]} if uncond_skip_active(sigma[0]) else {}
                    uncond = self.pipe.transformer(
                        x, timestep, negative, negative_mask, **skip
                    ).sample.float()
                    velocity = uncond + MOSS_GUIDANCE * (cond - uncond)
                    return x - sigma.reshape(-1, 1, 1, 1) * velocity

                latents = sample_er_sde(denoise, latents, simple_sigmas(MOSS_STEPS).to(device))
                image = self._decode_latents(latents)
        finally:
            self.pipe.maybe_free_model_hooks()
        return [image_to_base64(image)]

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
