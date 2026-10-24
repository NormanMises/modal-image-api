"""Z-Image runtime dependencies, GPU resources, and container lifecycle."""

import modal

from ...contracts import ImageBackend
from ...deployment import ModelDeployment, with_source
from .config import CACHE_DIR, CACHE_VOLUME, MODEL_DIR, MODEL_INFO, MODEL_REPO, SOURCE_COMMIT

app = modal.App()
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
image = with_source(
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install("tqdm", f"git+https://github.com/Tongyi-MAI/Z-Image.git@{SOURCE_COMMIT}")
    .env({
        "HF_HOME": CACHE_DIR,
        "HF_HUB_CACHE": f"{CACHE_DIR}/hub",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "ZIMAGE_ATTENTION": "_native_flash",
    })
)


@app.cls(image=image, gpu="L40S", volumes={CACHE_DIR: cache})
@modal.concurrent(max_inputs=1)
class ZImageTurboModel:
    @modal.enter()
    def load_model(self):
        from .backend import ZImageBackend

        self.backend: ImageBackend = ZImageBackend(model_dir=MODEL_DIR, repo_id=MODEL_REPO)
        cache.commit()

    @modal.method()
    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        return self.backend.generate(prompt, width, height)


deployment = ModelDeployment(app=app, info=MODEL_INFO, worker=ZImageTurboModel)
