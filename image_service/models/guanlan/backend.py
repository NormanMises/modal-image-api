"""Guanlan LoRA inference using the official OpenMOSS Python pipeline."""

from ...contracts import ImageBackend
from ...encoding import image_to_base64
from .config import (
    MOSS_GUIDANCE,
    MOSS_NEGATIVE_PROMPT,
    MOSS_REPO,
    MOSS_REVISION,
    MOSS_SHIFT,
    MOSS_STEPS,
    SKIP_END,
    SKIP_LAYER,
    SKIP_START,
)
from .lora import download_lora, merge_lora


class MossBackend(ImageBackend):
    def load(self, cache_dir: str) -> None:
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
        self.pipe.scheduler = type(self.pipe.scheduler).from_config(
            self.pipe.scheduler.config, shift=MOSS_SHIFT
        )
        self.pipe.enable_model_cpu_offload()

    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        # Return the complete image generated on the model's 16-pixel grid.
        width = ((width + 15) // 16) * 16
        height = ((height + 15) // 16) * 16
        images = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=MOSS_STEPS,
            guidance_scale=MOSS_GUIDANCE,
            negative_prompt=MOSS_NEGATIVE_PROMPT,
            uncond_skip_layers=[SKIP_LAYER],
            slg_start=SKIP_START,
            slg_stop=SKIP_END,
        ).images
        return [image_to_base64(image) for image in images]
