"""Model-independent contracts; no web, deployment, or inference dependencies."""

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

Generate = Callable[[str, int, int], Awaitable[list[str]]]


@dataclass(frozen=True)
class ModelInfo:
    id: str
    owned_by: str
    default_size: str = "1024x1024"


class ImageBackend(Protocol):
    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        """Return PNG images encoded as Base64 strings."""
        ...
