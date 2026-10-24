"""Guanlan runtime dependencies, GPU resources, and container lifecycle."""

import modal

from ...contracts import ImageBackend
from ...deployment import ModelDeployment, with_source
from .config import CACHE_DIR, CACHE_VOLUME, MODEL_INFO

app = modal.App()
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
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
    .env({"HF_HOME": CACHE_DIR, "TOKENIZERS_PARALLELISM": "false"})
)


@app.cls(
    image=image,
    gpu="L40S",
    memory=65536,
    volumes={CACHE_DIR: cache},
    timeout=1200,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=1)
class KookMossGuanlan:
    @modal.enter()
    def load_model(self):
        from .backend import MossBackend

        self.backend: ImageBackend = MossBackend(CACHE_DIR)
        cache.commit()

    @modal.method()
    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        return self.backend.generate(prompt, width, height)


deployment = ModelDeployment(app=app, info=MODEL_INFO, worker=KookMossGuanlan)
