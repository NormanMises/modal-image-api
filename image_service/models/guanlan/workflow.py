"""Adapt the pinned official image_moss_image_t2i template for API inputs and LoRA."""

import copy
import json
from pathlib import Path
import secrets

from .config import LORA_STRENGTH, MODEL_FILES, MOSS_NEGATIVE_PROMPT


def load_workflow() -> dict:
    workflow = json.loads(Path("/opt/moss-workflow/api.json").read_text())
    template = json.loads(Path("/opt/moss-workflow/template.json").read_text())
    nodes = {node["id"]: node for node in template["definitions"]["subgraphs"][0]["nodes"]}
    workflow["18:1"]["inputs"]["unet_name"] = Path(MODEL_FILES[0]).name
    workflow["18:2"]["inputs"]["clip_name"] = Path(MODEL_FILES[1]).name
    workflow["18:3"]["inputs"]["vae_name"] = Path(MODEL_FILES[2]).name
    workflow["18:14"]["inputs"]["value"] = nodes[14]["widgets_values"][0]
    workflow["18:5"]["inputs"]["text"] = MOSS_NEGATIVE_PROMPT
    workflow["18:lora"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["18:1", 0],
            "lora_name": "kook-moss-guanlan.safetensors",
            "strength_model": LORA_STRENGTH,
        },
    }
    workflow["18:15"]["inputs"]["model"] = ["18:lora", 0]
    workflow["18:24"]["inputs"]["on_false"] = ["18:lora", 0]
    return workflow


def generation_workflow(workflow: dict, prompt: str, width: int, height: int) -> dict:
    workflow = copy.deepcopy(workflow)
    workflow["18:17"]["inputs"]["value"] = prompt
    workflow["18:6"]["inputs"].update(
        width=((width + 15) // 16) * 16,
        height=((height + 15) // 16) * 16,
    )
    workflow["18:7"]["inputs"]["seed"] = secrets.randbits(63)
    return workflow
