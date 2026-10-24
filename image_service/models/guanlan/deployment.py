"""Guanlan runtime dependencies, GPU resources, and container lifecycle."""

import modal

from ...config import CACHE_ROOT
from ...contracts import ImageBackend
from ...deployment import ModelDeployment, ModelWorker, with_source
from ...storage import model_cache
from .config import CACHE_DIR, MODEL_INFO

app = modal.App()
image = with_source(
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.9.1",
        "diffusers==0.38.0",
        "transformers>=5.3,<6",
        "accelerate>=1.10,<2",
        "safetensors>=0.6,<1",
        "Pillow>=10",
    )
    .env({"HF_HOME": CACHE_DIR})
)


@app.cls(image=image, gpu="L40S", volumes={CACHE_ROOT: model_cache})
class KookMossGuanlan(ModelWorker):
    cache_dir = CACHE_DIR

    def create_backend(self) -> ImageBackend:
        from .backend import MossBackend

        return MossBackend()


deployment = ModelDeployment(app=app, info=MODEL_INFO, worker=KookMossGuanlan)
