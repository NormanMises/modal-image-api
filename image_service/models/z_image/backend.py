"""Z-Image-Turbo 权重加载、原生生成与图片序列化。"""

import os

from ...contracts import ImageBackend
from ...encoding import image_to_base64
from .config import GUIDANCE, MODEL_REPO, STEPS, WEIGHTS_SUBDIR


class ZImageBackend(ImageBackend):
    """Z-Image-Turbo 的加载与生成后端。"""

    def load(self, cache_dir: str) -> None:
        import torch
        from utils import ensure_model_weights, load_from_local_dir, set_attention_backend

        print("正在加载 base Z-Image-Turbo 组件...")
        model_path = ensure_model_weights(
            f"{cache_dir}/{WEIGHTS_SUBDIR}", repo_id=MODEL_REPO, verify=False
        )
        self.components = load_from_local_dir(
            model_path,
            device="cuda",
            dtype=torch.bfloat16,
            compile=False,
        )
        attention_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
        set_attention_backend(attention_backend)

        torch.cuda.empty_cache()
        print(f"Z-Image native model loaded; attention={attention_backend}")
        print("模型加载完成！")

    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        import torch
        from zimage import generate as native_generate

        images = native_generate(
            prompt=prompt,
            **self.components,
            height=height,
            width=width,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
        )
        torch.cuda.empty_cache()
        return [image_to_base64(image_obj) for image_obj in images]
