"""MLOps platform — public API.

Sub-packages:

* :mod:`mlplatform.config` — PipelineConfig, TrainingConfig, PredictionConfig,
  PipelineConfigBuilder, load_config_profiles, load_model_config.
* :mod:`mlplatform.storage` — Storage backends: LocalFileSystem, GCSStorage.
* :mod:`mlplatform.utils` — serialisation helpers and storage upload utilities.
* :mod:`mlplatform.artifacts` — Artifact, ArtifactRegistry, create_artifacts.
* :mod:`mlplatform.core` — BaseTrainer, BasePredictor, ExecutionContext.
* :mod:`mlplatform.tracking` — ExperimentTracker, NoneTracker, LocalJsonTracker.
* :mod:`mlplatform.profiles` — Profile, get_profile, register_profile.
* :mod:`mlplatform.inference` — InferenceStrategy, InProcessInference.
* :mod:`mlplatform.data` — load_prediction_input, write_prediction_output.
* :mod:`mlplatform.runner` — execute, dev_train, dev_predict.
* :mod:`mlplatform.spark` — build_root_zip, build_model_package.
* :mod:`mlplatform.cli` — CLI entry point.
"""

# ── v0.1.x public API ────────────────────────────────────────────────────────
from mlplatform._version import __version__
from mlplatform import artifacts, config, storage, utils

# Convenience: from mlplatform import Artifact
Artifact = artifacts.Artifact

__all__ = [
    "__version__",
    "Artifact",
    "artifacts",
    "config",
    "storage",
    "utils",
]
