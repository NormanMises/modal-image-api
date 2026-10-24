"""Model-independent contracts; no web, deployment, or inference dependencies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Awaitable, Callable

Generate = Callable[[str, int, int], Awaitable[list[str]]]


@dataclass(frozen=True)
class ModelInfo:
    id: str
    owned_by: str
    default_size: str = "1024x1024"


class ImageBackend(ABC):
    @abstractmethod
    def load(self, cache_dir: str) -> None:
        """Load the model once, using its persistent cache directory."""
        ...

    @abstractmethod
    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        """Return PNG images encoded as Base64 strings."""
        ...
