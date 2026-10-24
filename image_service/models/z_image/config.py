"""Z-Image-Turbo defaults and pinned source version."""

from ...config import model_cache_dir
from ...contracts import ModelInfo

MODEL_INFO = ModelInfo(id="z-image-turbo", owned_by="Tongyi-MAI")
CACHE_DIR = model_cache_dir(MODEL_INFO.id)
WEIGHTS_SUBDIR = "Z-Image-Turbo"
MODEL_REPO = "Tongyi-MAI/Z-Image-Turbo"
SOURCE_COMMIT = "26f23eda626ffadda020b04ff79488e1d72004cd"
STEPS = 8
GUIDANCE = 0.0
