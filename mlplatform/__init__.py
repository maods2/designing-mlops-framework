"""MLOps platform — public API for v0.1.x.

In this release the public API consists of four sub-packages:

* :mod:`mlplatform.config` — Pydantic-validated configuration models for
  training, prediction, and pipeline workflows.
* :mod:`mlplatform.storage` — Storage backends: ``LocalFileSystem`` and
  ``GCSStorage``.  GCS requires ``pip install mlplatform[storage]``.
* :mod:`mlplatform.utils` — serialisation helpers and storage upload utilities.
* :mod:`mlplatform.artifacts` — create ArtifactRegistry from config or explicit params.

Additional modules (tracking, inference strategies, Spark utilities) exist in
the repository and are importable directly, but they are **not part of the
supported public API** until a future release.

Install
-------
.. code-block:: bash

    # Only config (Pydantic schemas + YAML loading):
    pip install mlplatform[config]

    # Storage with GCS backend:
    pip install mlplatform[storage]

    # Utils + storage (serialisation, GCS artifact upload):
    pip install mlplatform[utils]

    # Everything in the current public API:
    pip install mlplatform[core]

Quick example
-------------
.. code-block:: python

    from mlplatform import Artifact, config

    cfg = config.TrainingConfig(model_name="m", feature="churn", version="v1")
    artifact = Artifact(**cfg.to_artifact_kwargs())
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

# ── Future full-platform API (not yet released) ───────────────────────────────
# Each sub-package requires its own pip extra:
#   pip install mlplatform[all]   # or select only what you need
#
# from mlplatform import (
#     core,         # mlplatform[core]    — BaseTrainer, BasePredictor,
#                   #                       ExecutionContext, ArtifactRegistry
#     tracking,     # mlplatform[tracking]— ExperimentTracker, LocalJsonTracker,
#                   #                       VertexAITracker, NoneTracker
#     spark,        # mlplatform[spark]   — build_root_zip, build_model_package
#     inference,    # (no extra needed)   — InferenceStrategy, InProcessInference
#     profiles,     # (no extra needed)   — Profile, get_profile, register_profile
#     serving,      # mlplatform[serving] — REST inference server (FastAPI)       [planned]
#     bigquery,     # mlplatform[bigquery]— BigQuery read/write helpers            [planned]
# )
