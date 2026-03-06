"""MLOps platform — public API for v0.1.x.

In this release the public API consists of two sub-packages:

* :mod:`mlplatform.utils` — serialisation helpers and storage upload utilities.
* :mod:`mlplatform.config` — Pydantic-validated configuration models for
  training, prediction, and pipeline workflows.

Additional modules (tracking, storage backends, invocation strategies, Spark
utilities) exist in the repository and are importable directly, but they are
**not part of the supported public API** until a future release.

Install
-------
.. code-block:: bash

    # Only utils (serialisation + GCS artifact upload):
    pip install mlplatform[utils]

    # Only config (Pydantic schemas + YAML loading):
    pip install mlplatform[config]

    # Everything in the current public API:
    pip install mlplatform[core]

Quick example
-------------
.. code-block:: python

    from mlplatform.utils import sanitize, save_plot
    from mlplatform.config import PipelineConfig

    pipeline = PipelineConfig.from_yaml("pipeline/train.yaml")
    print(pipeline.is_training, pipeline.model_count)
"""

# ── v0.1.x public API ────────────────────────────────────────────────────────
from mlplatform._version import __version__
from mlplatform import config, utils

__all__ = [
    "__version__",
    "config",
    "utils",
]

# ── Future full-platform API (not yet released) ───────────────────────────────
# The block below shows what this file will look like once all sub-packages
# graduate to the supported public API.  It is intentionally commented out so
# the intent is clear without affecting the current install.
#
# Each sub-package requires its own pip extra:
#   pip install mlplatform[all]   # or select only what you need
#
# from mlplatform import (
#     config,       # mlplatform[config]  — Pydantic pipeline/model config models
#     utils,        # mlplatform[utils]   — serialisation helpers + storage upload
#     storage,      # mlplatform[utils]   — Storage ABC, LocalFileSystem, GCSStorage
#     core,         # mlplatform[core]    — BaseTrainer, BasePredictor,
#                   #                       ExecutionContext, ArtifactRegistry
#     tracking,     # mlplatform[tracking]— ExperimentTracker, LocalJsonTracker,
#                   #                       VertexAITracker, NoneTracker
#     spark,        # mlplatform[spark]   — build_root_zip, build_model_package;
#                   #                       entry point: mlplatform/spark/main.py
#     invocation,   # (no extra needed)   — InvocationStrategy, InProcessInvocation
#     profiles,     # (no extra needed)   — Profile, get_profile, register_profile
#     serving,      # mlplatform[serving] — REST inference server (FastAPI)       [planned]
#     bigquery,     # mlplatform[bigquery]— BigQuery read/write helpers            [planned]
# )
#
# __all__ = [
#     "__version__",
#     # ── currently released ──────────────────────────────────────────────────
#     "config",
#     "utils",
#     "storage",
#     # ── graduating in the next release ──────────────────────────────────────
#     "core",
#     "tracking",
#     "spark",
#     "invocation",
#     "profiles",
#     # ── planned ─────────────────────────────────────────────────────────────
#     "serving",
#     "bigquery",
# ]
