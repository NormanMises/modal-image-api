"""Download validation and strict PEFT merging for the Guanlan LoRA."""

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


def merge_lora(transformer, state: dict) -> int:
    """AI Toolkit PEFT 导出不另带 alpha，使用 B @ A，强度为 1。"""
    import torch

    pairs = []
    consumed = set()
    for key, a in state.items():
        if not key.startswith("diffusion_model.") or not key.endswith(".lora_A.weight"):
            continue
        stem = key.removesuffix(".lora_A.weight")
        b_key = stem + ".lora_B.weight"
        if b_key not in state:
            raise ValueError(f"缺少 LoRA B: {stem}")
        b = state[b_key]
        module = transformer.get_submodule(stem.removeprefix("diffusion_model."))
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"无效 LoRA 形状: {stem}")
        if tuple(module.weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(f"LoRA 与底模形状不匹配: {stem}")
        pairs.append((module.weight, a, b))
        consumed.update((key, b_key))
    if not pairs or consumed != set(state):
        raise ValueError(f"LoRA 含未识别权重: {sorted(set(state) - consumed)}")
    with torch.no_grad():
        for weight, a, b in pairs:
            delta = b.float() @ a.float()
            weight.add_(delta.to(device=weight.device, dtype=weight.dtype))
    return len(pairs)
