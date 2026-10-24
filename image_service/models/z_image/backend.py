"""Z-Image-Turbo 权重加载、原生生成与图片序列化。"""

import os

from ...encoding import image_to_base64
from .config import GUIDANCE, STEPS


class ZImageBackend:
    """Z-Image-Turbo 的加载与生成后端。"""

    def __init__(
        self,
        model_dir: str,
        repo_id: str,
        steps: int = STEPS,
        guidance: float = GUIDANCE,
    ):
        import torch
        from utils import ensure_model_weights, load_from_local_dir, set_attention_backend

        self._steps = steps
        self._guidance = guidance

        print("正在加载 base Z-Image-Turbo 组件...")
        model_path = ensure_model_weights(model_dir, repo_id=repo_id, verify=False)
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
            num_inference_steps=self._steps,
            guidance_scale=self._guidance,
        )
        torch.cuda.empty_cache()
        return [image_to_base64(image_obj) for image_obj in images]
