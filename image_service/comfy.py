"""Run native ComfyUI workflows inside a model container."""

import base64
from contextlib import closing
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from .timing import log_timing


class ComfyClient:
    url = "http://127.0.0.1:8188"

    def start(self, model_dir: Path) -> None:
        started = time.perf_counter()
        paths = Path("/tmp/comfy-model-paths.json")
        paths.write_text(json.dumps({"image_models": {
            "base_path": str(model_dir),
            **{name: name for name in ("diffusion_models", "text_encoders", "vae", "loras")},
        }}))
        self.output_dir = Path("/tmp/comfy-output")
        self.process = subprocess.Popen([
            sys.executable, "main.py", "--listen", "127.0.0.1",
            "--disable-auto-launch", "--disable-all-custom-nodes",
            "--extra-model-paths-config", str(paths),
            "--output-directory", str(self.output_dir),
        ], cwd="/opt/ComfyUI")
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"ComfyUI exited with code {self.process.returncode}")
            try:
                self.request("/system_stats")
                break
            except urllib.error.URLError:
                time.sleep(0.1)
        log_timing("comfy_start", started)

    def request(self, path: str, body: dict | None = None):
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request) as response:
                data = response.read()
                return json.loads(data) if data else None
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from error

    def generate(self, workflow: dict) -> list[str]:
        import websocket

        started = time.perf_counter()
        client_id = uuid.uuid4().hex
        with closing(websocket.create_connection(f"ws://127.0.0.1:8188/ws?clientId={client_id}")) as socket:
            prompt_id = self.request("/prompt", {"prompt": workflow, "client_id": client_id})["prompt_id"]
            while True:
                message = socket.recv()
                if isinstance(message, bytes):
                    continue
                event = json.loads(message)
                data = event["data"]
                if data.get("prompt_id") != prompt_id:
                    continue
                if event["type"] == "execution_start":
                    log_timing("comfy_queue", started, prompt_id=prompt_id)
                    execution_started = time.perf_counter()
                if event["type"] in {"execution_error", "execution_interrupted"}:
                    raise RuntimeError(json.dumps(data))
                if event["type"] == "executing" and data["node"] is None:
                    break

        log_timing("comfy_execution", execution_started, prompt_id=prompt_id)
        started = time.perf_counter()
        history = self.request(f"/history/{prompt_id}")[prompt_id]
        log_timing("comfy_history", started, prompt_id=prompt_id)
        started = time.perf_counter()
        images = []
        image_bytes = 0
        for node in history["outputs"].values():
            for output in node.get("images", []):
                path = self.output_dir / output["subfolder"] / output["filename"]
                png = path.read_bytes()
                image_bytes += len(png)
                images.append(base64.b64encode(png).decode("ascii"))
                path.unlink()
        log_timing("image_read_encode", started, prompt_id=prompt_id, image_bytes=image_bytes, images=len(images))
        started = time.perf_counter()
        self.request("/history", {"delete": [prompt_id]})
        log_timing("comfy_cleanup", started, prompt_id=prompt_id)
        return images
