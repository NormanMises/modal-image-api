"""Guanlan runtime dependencies, GPU resources, and container lifecycle."""

import modal

from ...config import CACHE_ROOT
from ...contracts import ImageBackend
from ...deployment import ModelDeployment, ModelWorker, with_source
from ...storage import model_cache
from .config import CACHE_DIR, COMFY_COMMIT, MODEL_INFO, TEMPLATE_COMMIT, TEMPLATE_NAME

app = modal.App()
image = with_source(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "git init /opt/ComfyUI",
        "git -C /opt/ComfyUI remote add origin https://github.com/Comfy-Org/ComfyUI.git",
        f"git -C /opt/ComfyUI fetch --depth 1 origin {COMFY_COMMIT}",
        "git -C /opt/ComfyUI checkout FETCH_HEAD",
        "pip install -r /opt/ComfyUI/requirements.txt",
        "mkdir -p /opt/moss-workflow",
        "python -c \"import urllib.request; "
        f"urllib.request.urlretrieve('https://raw.githubusercontent.com/Comfy-Org/workflow_templates/{TEMPLATE_COMMIT}/templates/{TEMPLATE_NAME}', '/opt/moss-workflow/template.json'); "
        f"urllib.request.urlretrieve('https://raw.githubusercontent.com/Comfy-Org/workflow_templates/{TEMPLATE_COMMIT}/benchmarks/{TEMPLATE_NAME}', '/opt/moss-workflow/api.json')\"",
    )
    .uv_pip_install("huggingface_hub", "websocket-client")
    .env({"HF_HOME": CACHE_DIR})
)


@app.cls(image=image, gpu="L40S", volumes={CACHE_ROOT: model_cache})
class KookMossGuanlan(ModelWorker):
    cache_dir = CACHE_DIR

    def create_backend(self) -> ImageBackend:
        from .backend import MossBackend

        return MossBackend()


deployment = ModelDeployment(app=app, info=MODEL_INFO, worker=KookMossGuanlan)
