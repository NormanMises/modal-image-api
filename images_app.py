"""统一的 Modal 部署：一个 App、一个 FastAPI 应用、多个生图模型端点。

- ``images_api.create_images_app`` 提供共享的 OpenAI Images 兼容 web 接口。
- ``zimage_loader.ZImageBackend`` 封装 Z-Image-Turbo 的加载与生成。
- 部署：``modal deploy images_app.py``
"""

import modal

from images_api import ImagesMeta, create_images_app
from zimage_loader import ZImageBackend


CACHE_DIR = "/cache"
Z_IMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
Z_IMAGE_COMMIT = "26f23eda626ffadda020b04ff79488e1d72004cd"
Z_IMAGE_DIR = f"{CACHE_DIR}/Z-Image-Turbo"


app = modal.App("images-api")

z_image_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
moss_cache = modal.Volume.from_name("moss-image-sft-cache", create_if_missing=True)

moss_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi[standard]",
        "torch==2.9.1",
        "diffusers==0.38.0",
        "transformers>=5.3,<6",
        "accelerate>=1.10,<2",
        "safetensors>=0.6,<1",
        "Pillow>=10",
    )
    .env({"HF_HOME": CACHE_DIR, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file("images_api.py", remote_path="/root/images_api.py")
    .add_local_file("zimage_loader.py", remote_path="/root/zimage_loader.py")
    .add_local_file("moss_loader.py", remote_path="/root/moss_loader.py")
    .add_local_file("comfy_moss_sampling.py", remote_path="/root/comfy_moss_sampling.py")
)

z_image_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "fastapi[standard]",
        "tqdm",
        f"git+https://github.com/Tongyi-MAI/Z-Image.git@{Z_IMAGE_COMMIT}",
    )
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_HUB_CACHE": f"{CACHE_DIR}/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "ZIMAGE_ATTENTION": "_native_flash",
        }
    )
    .add_local_file("images_api.py", remote_path="/root/images_api.py")
    .add_local_file("zimage_loader.py", remote_path="/root/zimage_loader.py")
)

@app.cls(image=z_image_image, gpu="L40S", volumes={CACHE_DIR: z_image_cache})
class ZImageTurboModel:
    @modal.enter()
    def load_model(self):
        self.backend = ZImageBackend(model_dir=Z_IMAGE_DIR, repo_id=Z_IMAGE_REPO)

    @modal.method()
    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        return self.backend.generate(prompt, width, height)


@app.cls(
    image=moss_image,
    gpu="L40S",
    memory=65536,
    volumes={CACHE_DIR: moss_cache},
    timeout=1200,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=1)
class KookMossGuanlan:
    @modal.enter()
    def load_model(self):
        from moss_loader import MossBackend

        self.backend = MossBackend(CACHE_DIR)
        moss_cache.commit()

    @modal.method()
    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        return self.backend.generate(prompt, width, height)


web_app = create_images_app(
    models={
        "kook-moss-guanlan": lambda prompt, width, height: KookMossGuanlan().generate.remote.aio(
            prompt=prompt, width=width, height=height
        ),
        "z-image-turbo": lambda prompt, width, height: ZImageTurboModel().generate.remote.aio(
            prompt=prompt, width=width, height=height
        ),
    },
    metas={
        "kook-moss-guanlan": ImagesMeta(model="kook-moss-guanlan", owned_by="KookYan"),
        "z-image-turbo": ImagesMeta(model="z-image-turbo", owned_by="Tongyi-MAI"),
    },
)


@app.function(
    image=z_image_image,
    secrets=[modal.Secret.from_name("z-image-api")],
)
@modal.asgi_app()
def fastapi_app():
    return web_app
