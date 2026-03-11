"""Configuration schemas, Pydantic models, and YAML loading.

Public API
----------
The primary interface for users of ``mlplatform[config]``:

* :class:`TrainingConfig` — Pydantic model for a single training job.
* :class:`PredictionConfig` — Pydantic model for a single inference job.
* :class:`PipelineConfig` — Pydantic model for a full workflow; supports
  ``PipelineConfig.from_yaml(path)`` to load from a DAG YAML file.

Framework internals (used by loader and runners):

* :class:`ModelConfig` — Pydantic model used by the YAML loader.
* :class:`WorkflowConfig` — Pydantic model representing the raw loaded workflow.
* :func:`load_workflow_config` — low-level YAML loader (used by
  :meth:`PipelineConfig.from_yaml`).

Install
-------
    pip install mlplatform[config]
"""

from mlplatform.config.loader import load_workflow_config
from mlplatform.config.models import (
    ModelConfig,
    PipelineConfig,
    PredictionConfig,
    TrainingConfig,
    WorkflowConfig,
)

__all__ = [
    # ── Pydantic models (public API) ──────────────────────────────────────────
    "TrainingConfig",
    "PredictionConfig",
    "PipelineConfig",
    # ── Loader output types (framework internals / advanced use) ─────────────────
    "ModelConfig",
    "WorkflowConfig",
    "load_workflow_config",
]
