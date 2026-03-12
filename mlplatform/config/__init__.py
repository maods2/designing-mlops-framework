"""Configuration schemas, Pydantic models, and YAML profile loading.

* :class:`TrainingConfig` — config for training; accepts kwargs dict.
* :class:`PredictionConfig` — config for prediction; accepts kwargs dict.
* :class:`RunConfig` — config for train/predict artifact runs.
* :func:`load_config_profiles` — load and merge YAML config profiles.
"""

from mlplatform.config.loader import load_config_profiles
from mlplatform.config.models import (
    PredictionConfig,
    RunConfig,
    TrainingConfig,
)

__all__ = [
    "TrainingConfig",
    "PredictionConfig",
    "RunConfig",
    "load_config_profiles",
]
