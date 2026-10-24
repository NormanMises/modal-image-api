"""Z-Image runtime dependencies, GPU resources, and container lifecycle."""

import modal

from ...config import CACHE_ROOT
from ...contracts import ImageBackend
from ...deployment import ModelDeployment, ModelWorker, with_source
from ...storage import model_cache
from .config import CACHE_DIR, MODEL_INFO, SOURCE_COMMIT

app = modal.App()
image = with_source(
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install("tqdm", f"git+https://github.com/Tongyi-MAI/Z-Image.git@{SOURCE_COMMIT}")
    .env({"HF_HOME": CACHE_DIR})
)


@app.cls(image=image, gpu="L40S", volumes={CACHE_ROOT: model_cache})
class ZImageTurboModel(ModelWorker):
    cache_dir = CACHE_DIR

    def create_backend(self) -> ImageBackend:
        from .backend import ZImageBackend

        return ZImageBackend()


deployment = ModelDeployment(app=app, info=MODEL_INFO, worker=ZImageTurboModel)
