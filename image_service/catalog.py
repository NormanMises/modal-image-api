"""Enabled models. Register a deployment here to expose it through the API."""

from .models.guanlan.deployment import deployment as guanlan
from .models.z_image.deployment import deployment as z_image

MODEL_DEPLOYMENTS = (guanlan, z_image)
