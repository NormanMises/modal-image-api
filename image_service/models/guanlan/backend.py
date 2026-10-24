"""Guanlan inference through ComfyUI's official MOSS-Image template and native nodes."""

from pathlib import Path

from ...comfy import ComfyClient
from ...contracts import ImageBackend
from .config import MODEL_FILES, MODEL_REPO, MODEL_REVISION
from .lora import download_lora
from .workflow import generation_workflow, load_workflow


def prepare_models(cache_dir: str) -> Path:
    from huggingface_hub import hf_hub_download

    model_dir = Path(cache_dir) / "comfyui" / "models"
    for filename in MODEL_FILES:
        hf_hub_download(MODEL_REPO, filename, revision=MODEL_REVISION, local_dir=model_dir)
    download_lora(model_dir / "loras")
    return model_dir


class MossBackend(ImageBackend):
    def load(self, cache_dir: str) -> None:
        model_dir = prepare_models(cache_dir)
        self.workflow = load_workflow()
        self.client = ComfyClient()
        self.client.start(model_dir)

    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        return self.client.generate(generation_workflow(self.workflow, prompt, width, height))
