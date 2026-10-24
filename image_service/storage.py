"""One persistent model-weight Volume shared by all GPU containers."""

import modal

from .config import CACHE_VOLUME

model_cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
