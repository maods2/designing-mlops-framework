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

from mlplatform._version import __version__
from mlplatform import config, utils

__all__ = [
    "__version__",
    "config",
    "utils",
]
