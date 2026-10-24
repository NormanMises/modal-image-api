"""One source of truth for model metadata and generation handlers."""

from dataclasses import dataclass

from .contracts import Generate, ModelInfo


@dataclass(frozen=True)
class ModelEndpoint:
    info: ModelInfo
    generate: Generate


class ModelRegistry:
    def __init__(self, endpoints: tuple[ModelEndpoint, ...]):
        self._endpoints = {endpoint.info.id: endpoint for endpoint in endpoints}

    def get(self, model_id: str) -> ModelEndpoint | None:
        return self._endpoints.get(model_id)

    def list_models(self) -> tuple[ModelInfo, ...]:
        return tuple(endpoint.info for endpoint in self._endpoints.values())
