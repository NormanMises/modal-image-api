"""Modal wiring shared by independent model deployments."""

from dataclasses import dataclass

import modal

from .contracts import ModelInfo
from .registry import ModelEndpoint, ModelRegistry


def with_source(image: modal.Image) -> modal.Image:
    return image.add_local_python_source("image_service")


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
