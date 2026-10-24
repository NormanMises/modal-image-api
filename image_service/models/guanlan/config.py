"""Guanlan identity, pinned weights, and generation defaults."""

from ...config import model_cache_dir
from ...contracts import ModelInfo

MODEL_INFO = ModelInfo(id="kook-moss-guanlan", owned_by="KookYan")
CACHE_DIR = model_cache_dir(MODEL_INFO.id)
COMFY_COMMIT = "7cebca52227c63b4180ceed448cbd7ad22903b6d"
TEMPLATE_COMMIT = "67984637eefcd4825dcf5c134e39034785be1682"
TEMPLATE_NAME = "image_moss_image_t2i.json"
MODEL_REPO = "Comfy-Org/MOSS-Image"
MODEL_REVISION = "24147db865fcdc207270f844777587223fed8d6e"
MODEL_FILES = (
    "diffusion_models/moss_image_sft_bf16.safetensors",
    "text_encoders/qwen3.5_9b_bf16.safetensors",
    "vae/flux2-vae_small.safetensors",
)
LORA_REVISION = "f19ce490b7cee2c10700d44bda6bbbc8a3f0716d"
LORA_SHA256 = "52e41bc8ab3964b11104332bc7fc742571e5fd1eaf7bc046285a5b37c56d3020"
LORA_URL = (
    "https://modelscope.cn/api/v1/models/KookYan/Kook_Moss_SFT_guanlan/repo"
    f"?Revision={LORA_REVISION}&FilePath=KOOK-MOSS-SFT-%E8%A7%82%E6%BE%9C.safetensors"
)
MOSS_NEGATIVE_PROMPT = "worst quality, bad anatomy, watermark, jpeg artifact"
LORA_STRENGTH = 1.0
