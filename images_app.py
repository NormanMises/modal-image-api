"""Compose model deployments and expose the shared OpenAI-compatible API.

Deploy with: modal deploy images_app.py
"""

import modal

from image_service.catalog import MODEL_DEPLOYMENTS
from image_service.config import API_SECRET_NAME, APP_NAME
from image_service.deployment import register_models, with_source

app = modal.App(APP_NAME)
registry = register_models(app, MODEL_DEPLOYMENTS)
api_image = with_source(
    modal.Image.debian_slim(python_version="3.12").uv_pip_install("fastapi[standard]")
)


@app.function(image=api_image, secrets=[modal.Secret.from_name(API_SECRET_NAME)])
@modal.asgi_app()
def fastapi_app():
    from image_service.api import create_images_app

    return create_images_app(registry)
