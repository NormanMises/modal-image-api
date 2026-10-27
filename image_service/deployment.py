"""Modal wiring shared by independent model deployments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter

import modal

from .contracts import ImageBackend, ModelInfo
from .registry import ModelEndpoint, ModelRegistry
from .storage import model_cache
from .timing import log_timing, request_id


def with_source(image: modal.Image) -> modal.Image:
    return image.add_local_python_source("image_service")


class ModelWorker(ABC):
    """Shared container lifecycle and generation endpoint for all models."""

    cache_dir: str
    backend: ImageBackend

    @abstractmethod
    def create_backend(self) -> ImageBackend:
        """Import and construct the backend inside its GPU container."""
        ...

    @modal.enter()
    def load_model(self):
        started = perf_counter()
        self.backend = self.create_backend()
        self.backend.load(self.cache_dir)
        log_timing("backend_load", started, cache_dir=self.cache_dir)
        started = perf_counter()
        model_cache.commit()
        log_timing("cache_commit", started, cache_dir=self.cache_dir)

    @modal.method()
    def generate(self, prompt: str, width: int, height: int, trace_id: str) -> list[str]:
        token = request_id.set(trace_id)
        started = perf_counter()
        try:
            return self.backend.generate(prompt, width, height)
        finally:
            log_timing("worker_generate", started, cache_dir=self.cache_dir, width=width, height=height)
            request_id.reset(token)


@dataclass(frozen=True)
class ModelDeployment:
    app: modal.App
    info: ModelInfo
    worker: modal.Cls

    def endpoint(self) -> ModelEndpoint:
        async def generate(prompt: str, width: int, height: int) -> list[str]:
            return await self.worker().generate.remote.aio(
                prompt=prompt, width=width, height=height, trace_id=request_id.get()
            )

        return ModelEndpoint(self.info, generate)


def register_models(app: modal.App, deployments: tuple[ModelDeployment, ...]) -> ModelRegistry:
    registry = ModelRegistry(tuple(deployment.endpoint() for deployment in deployments))
    for deployment in deployments:
        app.include(deployment.app)
    return registry
