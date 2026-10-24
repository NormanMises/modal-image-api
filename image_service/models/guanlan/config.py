"""Guanlan identity, pinned weights, and generation defaults."""

from ...config import model_cache_dir
from ...contracts import ModelInfo

MODEL_INFO = ModelInfo(id="kook-moss-guanlan", owned_by="KookYan")
CACHE_DIR = model_cache_dir(MODEL_INFO.id)
MOSS_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
MOSS_REVISION = "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9"
LORA_REVISION = "f19ce490b7cee2c10700d44bda6bbbc8a3f0716d"
LORA_SHA256 = "52e41bc8ab3964b11104332bc7fc742571e5fd1eaf7bc046285a5b37c56d3020"
MOSS_STEPS = 25
MOSS_GUIDANCE = 3.0
MOSS_SHIFT = 5.0
SKIP_LAYER = 22
SKIP_START = 0.0
SKIP_END = 0.5
MOSS_NEGATIVE_PROMPT = "worst quality, bad anatomy, watermark, jpeg artifact"
LORA_URL = (
    "https://modelscope.cn/api/v1/models/KookYan/Kook_Moss_SFT_guanlan/repo"
    f"?Revision={LORA_REVISION}&FilePath=KOOK-MOSS-SFT-%E8%A7%82%E6%BE%9C.safetensors"
)
