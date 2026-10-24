"""Shared service names; model-specific settings live with each model."""

APP_NAME = "images-api"
API_SECRET_NAME = "image-api-auth"
CACHE_VOLUME = "image-model-cache"
CACHE_ROOT = "/cache"


def model_cache_dir(model_id: str) -> str:
    return f"{CACHE_ROOT}/{model_id}"
