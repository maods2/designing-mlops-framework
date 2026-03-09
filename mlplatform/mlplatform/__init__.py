"""MLOps platform — public API for v0.1.x.

In this release the public API consists of three sub-packages:

* :mod:`mlplatform.utils` — serialisation helpers and storage upload utilities.
* :mod:`mlplatform.config` — Pydantic-validated configuration models for
  training, prediction, and pipeline workflows.
* :mod:`mlplatform.storage` — Storage backends: ``LocalFileSystem`` and
  ``GCSStorage``.  Requires ``pip install mlplatform[utils]``.

Additional modules (tracking, invocation strategies, Spark utilities) exist in
the repository and are importable directly, but they are **not part of the
supported public API** until a future release.

Install
-------
.. code-block:: bash

    # utils + storage backends (serialisation, GCS artifact upload):
    pip install mlplatform[utils]

    # Only config (Pydantic schemas + YAML loading):
    pip install mlplatform[config]

    # Everything in the current public API:
    pip install mlplatform[core]

Quick example
-------------
.. code-block:: python

    from mlplatform import storage, utils
    from mlplatform.config import PipelineConfig

    store = storage.LocalFileSystem("./artifacts")
    utils.save_plot(fig, "plots/my_chart.png", store)

    pipeline = PipelineConfig.from_yaml("pipeline/train.yaml")
    print(pipeline.is_training, pipeline.model_count)
"""

# ── v0.1.x public API ────────────────────────────────────────────────────────
from mlplatform._version import __version__
from mlplatform import config, storage, utils

__all__ = [
    "__version__",
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
#     invocation,   # (no extra needed)   — InvocationStrategy, InProcessInvocation
#     profiles,     # (no extra needed)   — Profile, get_profile, register_profile
#     serving,      # mlplatform[serving] — REST inference server (FastAPI)       [planned]
#     bigquery,     # mlplatform[bigquery]— BigQuery read/write helpers            [planned]
# )
