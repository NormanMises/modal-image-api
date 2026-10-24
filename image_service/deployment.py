"""Modal wiring shared by independent model deployments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import modal

from .contracts import ImageBackend, ModelInfo
from .registry import ModelEndpoint, ModelRegistry
from .storage import model_cache


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
        self.backend = self.create_backend()
        self.backend.load(self.cache_dir)
        model_cache.commit()

    @modal.method()
    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        return self.backend.generate(prompt, width, height)


@dataclass(frozen=True)
class ModelDeployment:
    app: modal.App
    info: ModelInfo
    worker: modal.Cls

    def endpoint(self) -> ModelEndpoint:
        async def generate(prompt: str, width: int, height: int) -> list[str]:
            return await self.worker().generate.remote.aio(
                prompt=prompt, width=width, height=height
            )

        return ModelEndpoint(self.info, generate)


def register_models(app: modal.App, deployments: tuple[ModelDeployment, ...]) -> ModelRegistry:
    registry = ModelRegistry(tuple(deployment.endpoint() for deployment in deployments))
    for deployment in deployments:
        app.include(deployment.app)
    return registry
