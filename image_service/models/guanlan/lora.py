"""Download and verify the Guanlan LoRA for the native ComfyUI loader."""

import hashlib
from pathlib import Path
import tempfile
import urllib.request

from .config import LORA_SHA256, LORA_URL


def download_lora(cache_dir: str) -> Path:
    path = Path(cache_dir) / "kook-moss-guanlan.safetensors"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == LORA_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # 多个冷启动实例共享 Volume，各自下载到独立临时文件再原子替换。
    with tempfile.TemporaryDirectory(prefix=".guanlan-", dir=path.parent) as temp_dir:
        temporary = Path(temp_dir) / path.name
        with urllib.request.urlopen(LORA_URL, timeout=120) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != LORA_SHA256:
            raise ValueError("观澜 LoRA SHA256 校验失败")
        temporary.replace(path)
    return path
