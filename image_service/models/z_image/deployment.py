"""Z-Image runtime dependencies, GPU resources, and container lifecycle."""

import modal

from ...config import CACHE_ROOT
from ...contracts import ImageBackend
from ...deployment import ModelDeployment, with_source
from ...storage import model_cache
from .config import CACHE_DIR, MODEL_DIR, MODEL_INFO, MODEL_REPO, SOURCE_COMMIT

app = modal.App()
image = with_source(
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install("tqdm", f"git+https://github.com/Tongyi-MAI/Z-Image.git@{SOURCE_COMMIT}")
    .env({"HF_HOME": CACHE_DIR})
)


@app.cls(image=image, gpu="L40S", volumes={CACHE_ROOT: model_cache})
class ZImageTurboModel:
    @modal.enter()
    def load_model(self):
        from .backend import ZImageBackend

        self.backend: ImageBackend = ZImageBackend(model_dir=MODEL_DIR, repo_id=MODEL_REPO)
        model_cache.commit()

    @modal.method()
    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        return self.backend.generate(prompt, width, height)


deployment = ModelDeployment(app=app, info=MODEL_INFO, worker=ZImageTurboModel)
